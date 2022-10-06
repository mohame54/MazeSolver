import argparse
import json
from Maze import *
from MazeSolver import *
import numpy as np
import time


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Episodes', default=10, type=int, help='the number of episodes')
    parser.add_argument('--show_maze', default=False, type=bool, help='whether to show the agent path for every '
                                                                      'episode on the grid maze ')
    parser.add_argument('--save_best_net', default=True, type=bool, help='whether to save the best net during training')
    parser.add_argument('--path_net', default='best_net.pb', type=str, help='path to save the net')
    return parser.parse_args()


def get_agent_env():
    with open('config.json', 'r') as file:
        params = json.load(file)
    env = MazeEnv(**params['Maze'])
    Agent = MazeAgent(**params['Agent'])
    return env, Agent


if __name__ == '__main__':
    best_reward = -1e4
    args = options()
    env, agent = get_agent_env()
    # logs_episodes = {
    #   'avg_rewards': [], 'num_steps': [], 'avg_losses': []
    # }
    wins_losses = []
    for e in range(args.Episodes):
        start_time = time.time()
        state = env.reset()
        loss_hist = []
        while True:
            act = agent.act(state)
            next_st, reward, done, statue = env.observe(act)
            agent.cache(state, act, next_st, reward, done)
            loss, q_item = agent.learn()
            state = next_st
            if loss is not None and q_item is not None:
                loss_hist.append(loss)
            if done:
                wins_losses.append(statue)
                break
        if best_reward < env.total_reward and args.save_best_net:
            best_reward = env.total_reward
            agent.save_network(args.path_net)
        if args.show_maze:
            env.plot_with_arrows()
        if len(loss_hist) == 0:
            loss_hist.append(1)
        # logs_episodes['avg_rewards'].append(env.total_reward)
        # logs_episodes['num_steps'].append(env.steps)
        # logs_episodes['avg_losses'].append(np.mean(loss_hist))
        print(f"Episode : {e + 1} finished in :"
              f" {(time.time() - start_time):.3f} secs with loss:{np.mean(loss_hist):.3f}, total_reward:{env.total_reward:.3f} taking steps: {env.steps} step\n")

    num_wins = sum([i == 'win' for i in wins_losses])
    num_losses = sum([i == 'loss' for i in wins_losses])
    print(f"In {args.Episodes} episodes, the agent had won: {num_wins} times and lost: {num_losses} times.")
