import json


# you can modify the size of the maze and its shape  as you want
default_maze = [[1] * 8,
                         [1, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 1, 1, 1, 1, 1, 1],
                         [1, 0, 1, 0, 0, 1, 0, 1],
                         [1, 0, 1, 1, 0, 1, 0, 1],
                         [1, 0, 0, 0, 0, 1, 0, 1],
                         [1, 1, 1, 1, 0, 1, 0, 1],
                         [1, 0, 0, 0, 0, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 0, 1]],
                         # each 1 in the array represent vacant pos and 0 represent  a black wall

maze_dim = 8*9 # 8 rows 9 columns
action_map = {
    0: "R", 1: "L", 2: 'U', 3: "D"
}

maze_params = {
    'maze': default_maze,
    'action_map': action_map,
    'st_pos': (0, 0),
    'end_pos': (8, 7)
}
Maze_agent_params = {
    'batch_size': 64, 'cache_size': 10000, 'feat_dim': maze_dim, 'sync_every': 5, 'num_blcs': 4,
    'act_dim': 4, 'init_exp_rate': 1.0, 'min_exp_rate': 0.1, 'gamma': 0.999, 'discount': 0.95
}

train_params = {
    'Maze': maze_params,
    'Agent': Maze_agent_params
}

with open('config.json', 'w') as file:
    json.dump(train_params, file)

