import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from random import randint
from timeit import default_timer as timer

from models import Actor
from models import Critic
from rl_exp.rl_utils.noise_process import OUNoise
from rl_exp.rl_utils.replay_memory import ReplayBuffer

"""
From
https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial/blob/master/ddpg%20walker/ddpg_agent.py
"""


class Agent:
    """Interacts with and learns from the environment
    """

    def __init__(
        self,
        state_size,
        action_size,
        #  random_seed,
        device,
        LR_ACTOR,
        LR_CRITIC,
        BUFFER_SIZE,
        BATCH_SIZE,
        WEIGHT_DECAY,
        GAMMA,
        TAU,
    ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            #  random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        #  self.seed = random_seed
        self.device = device

        self.LR_ACTOR = LR_ACTOR
        self.LR_CRITIC = LR_CRITIC
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.GAMMA = GAMMA
        self.TAU = TAU

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.LR_ACTOR
        )

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=self.LR_CRITIC,
            weight_decay=self.WEIGHT_DECAY,
        )

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.memory = ReplayBuffer(
            self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.device,
        )

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
