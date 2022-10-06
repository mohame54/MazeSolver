# Maze solver project
**This a project to implement and train a  Deep Rl Agent to solve maze using Double deep q net.**


![](https://github.com/mohame54/MazeSolever/blob/main/Gifs/test.gif)            


![](https://github.com/mohame54/MazeSolever/blob/main/Gifs/test2.gif)


#***Content***:

1-***Objective and Goal***

2-***How to train our own Agent***

#objective and Goal :
*My goal here is to make an agent or a model that can start from **any valid position** in the grid maze and find the terminal or the goal point ***Marked with gray in the above grids*** the designed maze is flexible meaning we can change the start and  goal position ***whenever*** we want and let the agent find the best route to reach ***goal*** taking the least number of steps even we changed start and end position after training the agent would still try to find the best route without any additional training .*

## How to train your agent :
*you can use ***CLI*** to train the agent*
> python train.py -h   **To show the available command**

*For ***example :**** 
> python train.py --Episodes 15 save_net True

*This ***command*** will train the agent for 15 Episodes with saving the best net weights during training*


*The code below shows  how to train your agent after optimizing the config file ***config.json*** from ***config.py*** file or leave the default params*
```python
from Maze import *
from MazeSolver import *
import json
import time

with open('config.json','r') as file:
   params = json.load(file)
'''
the training params
'''   
Episodes=100 # num_episodes
show_maze= True
path_net = 'your_path.pb'
Agent = MazeAgent(**params['Agent'])
env = MazeEnv(**params['Maze']) 
best_reward = -1e4
wins_losses = []
for e in range(Episodes):
    start_time = time.time()
    state = env.reset()
    loss_hist = []
    while True:
        act = Agent.act(state)
        next_st, reward, done, statue = env.observe(act)
        Agent.cache(state, act, next_st, reward, done)
        loss, q_item = Agent.learn()
        state = next_st
        if loss is not None and q_item is not None:
             loss_hist.append(loss)
        if done:
            wins_losses.append(statue)
            break
    if best_reward < env.total_reward : # checking if total_reward is higher to save the net which made the biggest reward
        best_reward = env.total_reward
        Agent.save_network(path_net)
    if show_maze: # to show the path that the Agent made in the maze
        env.plot_with_arrows()
    if len(loss_hist) == 0:
        loss_hist.append(1) # to avoid crashes in the code we append 1 during the first episodes of training as there is no loss to measure in this time
    print(f"Episode : {e + 1} finished in :"
              f" {(time.time() - start_time):.3f} secs with loss:{np.mean(loss_hist):.3f}, total_reward:{env.total_reward:.3f} taking steps: {env.steps} step\n")

    num_wins = sum([i == 'win' for i in wins_losses])
    num_losses = sum([i == 'loss' for i in wins_losses])
    print(f"In {Episodes} episodes, the agent had won: {num_wins} times and lost: {num_losses} times.")
  
```

