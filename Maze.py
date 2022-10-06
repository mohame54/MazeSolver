import copy
import numpy as np
import matplotlib.pyplot as plt

MARK = 0.6
st_fin_mark = 0.7
REDUNTDANT_MARK = .5


class MazeCell:
    def __init__(self, row_pos: int, col_pos: int, done: str = 'going'):
        self.row_pos = row_pos
        self.col_pos = col_pos
        self.done = done

    def unpack(self):
        return self.row_pos, self.col_pos, self.done

    def update(self, r: int, c: int, done='going'):
        self.row_pos = r
        self.col_pos = c
        self.done = done

    def __str__(self):
        return f"pos:{(self.row_pos, self.col_pos)} with status: {self.done}"


class MazeEnv:
    def __init__(self, maze, action_map, st_pos=(0, 0), end_pos=(8, 7)):
        if isinstance(maze, list):
            maze = np.array(maze, dtype=np.float32).squeeze()
        self._actions = {}
        for k, v in action_map.items():
            self._actions[int(k)] = v
        self._maze = maze
        self.st_pos = tuple(st_pos)
        self.end_pos = tuple(end_pos)
        self._check_pos()
        self._arrows = {"R": (0.25, 0), "L": (-0.25, 0), "U": (0, -0.25), "D": (0, 0.25)}
        self._min_reward = maze.size * -0.5
        self.reset()

    def reset(self):
        self._check_pos()
        # self._maze[self.st_pos[0],self.st_pos[1]] = st_fin_mark
        self._maze[self.end_pos[0], self.end_pos[1]] = st_fin_mark - 0.4
        self._states = copy.deepcopy(self._maze)
        self._cell = MazeCell(self.st_pos[0], self.st_pos[1])
        self._visited_cells = set()
        self._visited_cells.add(self.st_pos)
        self.visited = []  # to store the cell pos and the action occurred in it
        self.steps = 0
        self.total_reward = 0
        return self._states

    def _check_pos(self):
        for p in [self.st_pos, self.end_pos]:
            try:
                self._maze[p[0], p[1]]
            except IndexError:
                raise IndexError(f"the pos: {p} is out of the maze.")
            if self._maze[p[0], p[1]] == 0.:
                raise TypeError(f"Can not place this pos: {p} in a black grid")

    def _crossed(self, x, y):
        nrows, ncols = self._maze.shape
        x_crossed = x < 0 or x >= nrows  # the x pos is out of the maze
        y_crossed = y < 0 or y >= ncols  # the y pos is out of the maze
        if x_crossed or y_crossed:
            return True
        else:
            return self._states[x, y] == 0  # checking if it hit the black grid marked with 0.

    def step(self, act):
        possible_acts = self.compute_possible_actions()
        x_pos, y_pos, status = self._cell.unpack()

        self.visited.append((x_pos, y_pos, act))
        if possible_acts:
            if act == 0:
                y_pos += 1
            elif act == 1:
                y_pos -= 1
            elif act == 2:
                x_pos -= 1
            elif act == 3:
                x_pos += 1

            if (x_pos, y_pos) == self.end_pos:
                self._cell.update(x_pos, y_pos, 'win')
                reward = 10
            elif self._crossed(x_pos, y_pos):
                reward = -1
            elif (x_pos, y_pos) in self._visited_cells:
                self._cell.update(x_pos, y_pos, 'red')
                reward = -0.75
            else:
                self._cell.update(x_pos, y_pos)
                reward = -0.05
        else:
            self._cell.update(x_pos, y_pos, 'loss')
            reward = self._min_reward
        return reward

    def observe(self, act):
        reward = self.step(act)
        self.total_reward += reward
        self.steps += 1
        x, y, status = self._cell.unpack()
        self._visited_cells.add((x, y))

        if status == 'red':
            self._states[x, y] = REDUNTDANT_MARK
        else:
            self._states[x, y] = MARK
        done = 1 if (status == 'win' or status == 'loss') else 0
        return self._states, reward, done, status

    def compute_possible_actions(self):
        if self.total_reward < self._min_reward:
            return []
        row, col, _ = self._cell.unpack()
        actions = list(self._actions.keys())
        nrows, ncols = self._maze.shape
        if row == 0 or (row > 0 and self._states[row - 1, col] == 0.0):
            actions.remove(2)
        if row == (nrows - 1) or (row < nrows - 1 and self._states[row + 1, col] == 0.):
            actions.remove(3)
        if col == 0 or (col > 0 and self._states[row, col - 1] == 0.):
            actions.remove(1)
        if col == ncols - 1 or (col < ncols - 1 and self._states[row, col + 1] == 0.):
            actions.remove(0)
        return actions

        # to get an image of the environment

    @property
    def rendre(self):
        return self._states

    # plot the actions on the image
    def plot_with_arrows(self, visited=None, save_fig=None):
        nrow, ncol = self._maze.shape
        plt.grid('on')
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrow, 1))
        ax.set_yticks(np.arange(0.5, ncol, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        visited = visited if visited is not None else self.visited
        plt.imshow(self._states, cmap='gray', interpolation='none')
        for x, y, act in visited:
            arrow = self._arrows[self._actions[act]]
            plt.arrow(y, x, arrow[0], arrow[1], head_width=0.1)
        if save_fig:
            plt.savefig(save_fig)
        plt.show()
