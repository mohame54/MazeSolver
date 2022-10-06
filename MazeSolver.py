import torch
from torch import nn
import numpy as np
import copy


class MazeNet(nn.Module):
    def __init__(self, feat_dim, num_blocks=2, act_dim=4):
        super().__init__()
        self._flatten = nn.Flatten()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU())
                                     for _ in range(num_blocks)])
        self._fc = nn.Linear(feat_dim, act_dim)

    def forward(self, state):
        x = self._flatten(state)
        for blc in self.blocks:
            x = blc(x)
        return self._fc(x)


class DoubleMazeNet(nn.Module):
    def __init__(self, feat_dim, num_blocks, act_dim=4):
        super(DoubleMazeNet, self).__init__()
        self.online = MazeNet(feat_dim, num_blocks, act_dim)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state, net='online'):
        if net == 'online':
            return self.online(state)
        else:
            return self.target(state)


from collections import deque
import random


class MazeAgent:
    def __init__(self, batch_size, cache_size=10000, feat_dim=72, sync_every=5, num_blcs=4,
                 act_dim=4, init_exp_rate=1.0, min_exp_rate=0.1, gamma=0.999, discount=0.95):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DoubleMazeNet(feat_dim, num_blcs, act_dim).to(self._device)
        self._gamma = gamma
        self._discount = discount
        self._exp_rate = init_exp_rate
        self._min_rate = min_exp_rate
        self.batch_size = batch_size
        self.explore_before_training = 200
        self.sync_every = sync_every
        self.cache_size = cache_size
        self._cur_step = 0
        self._action_dim = act_dim
        self._model_mem = deque(maxlen=cache_size)

    def act(self, state):
        if np.random.randn() < self._exp_rate:
            act_idx = np.random.randint(self._action_dim)
        else:
            st = torch.tensor(state, dtype=torch.float32).to(self._device).unsqueeze(0)
            actions = self.net(st)
            act_idx = actions.argmax(dim=-1).item()
        self._exp_rate *= self._gamma
        self._exp_rate = min(self._exp_rate, self._min_rate)
        self._cur_step += 1
        return act_idx

    def sync_q_target(self):
        st_dict = self.net.online.state_dict()
        self.net.target.load_state_dict(st_dict)

    def cache(self, prev_state, act, state, reward, done):
        prev_state = np.array([prev_state]).astype(np.float32)
        state = np.array([state]).astype(np.float32)
        act = np.array([act]).astype(np.int64)

        reward = np.array([reward]).astype(np.float32)
        done = np.array([done]).astype(np.float32)
        self._model_mem.append((prev_state, state, act, reward, done))

    def recall(self):
        batch = random.sample(self._model_mem, self.batch_size)
        stack_cast = lambda x: torch.tensor(np.stack(x, axis=0)).to(self._device)
        prev_st, st, act, reward, done = map(stack_cast, zip(*batch))

        return prev_st.squeeze(), st.squeeze(), act.squeeze(), reward.squeeze(), done.squeeze()

    def compile(self, optim_name='Adam', loss_name='SmoothL1Loss', lr=1e-4):
        self._optim = getattr(torch.optim, optim_name)(self.net.parameters(), lr=lr)
        self._loss = getattr(nn, loss_name)()

    @torch.no_grad()
    def td_target(self, state, reward, done):
        q_vals = self.net(state, 'online')
        actions_index = q_vals.argmax(dim=-1)
        next_q = self.net(state, 'target')[np.arange(0, self.batch_size),
                                           actions_index].squeeze()

        return (reward + (1 - done).float() * self._discount * next_q).float()

    def td_estimate(self, prev_state, action):
        cur_q = self.net(prev_state)
        cur_q = torch.gather(cur_q, -1, action.long().unsqueeze(-1)).squeeze(-1)
        return cur_q

    def update(self, q_cur, q_target):
        self._optim.zero_grad()
        loss = self._loss(q_cur, q_target)
        loss.backward()
        self._optim.step()
        return loss

    def save_network(self, path):
        torch.save(self.net.state_dict(), path)

    def load_network(self, path):
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict)

    def learn(self):
        self.compile()
        if self._cur_step < self.explore_before_training:
            return None, None
        if self._cur_step % self.sync_every == 0:
            return None, None
        prev_st, st, act, reward, done = self.recall()
        q_tar = self.td_target(st, reward, done)
        q_cur = self.td_estimate(prev_st, action=act)
        loss = self.update(q_cur, q_tar)
        return loss.item(), q_cur.mean().item()
