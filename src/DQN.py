import random
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import defaultdict

DEVICE = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'


class DQNAgent(torch.nn.Module):
    def __init__(self, params: dict, state_len: int, action_len: int):
        super().__init__()
        self.reward = 0
        self.gamma = 0.9
        self.learning_rate = params["learning_rate"]
        self.epsilon = 1
        self.first_layer = params["first_layer_size"]
        self.second_layer = params["second_layer_size"]
        self.third_layer = params["third_layer_size"]
        self.memory = collections.deque(maxlen=params["memory_size"])
        self.weights = params["weights_path"]
        self.load_weights = params["load_weights"]
        self.optimizer = None
        self.input_size = state_len
        self.output_size = action_len
        self.network()

    def network(self):
        # Layers
        self.f1 = nn.Linear(self.input_size, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, self.output_size)
        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x

    def set_reward(self, reward):
        self.reward = max(self.reward, reward)

    def remember(self, state, action, reward, next_state):
        """
        Store the <state, action, reward, next_state> tuple in a
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state))

    def upsample(self):
        samples_count = self.count_samples()
        max_count = max(samples_count.values())
        for key in samples_count:
            samples_count[key] = samples_count[key] / max_count

        length = len(self.memory)

        for i in range(length):
            state, action, reward, next_state = self.memory[i]
            copies = int(1 / samples_count[action] - 1)
            for j in range(copies):
                self.remember(state, action, reward, next_state)

        self.count_samples()

    def count_samples(self):
        count = defaultdict(int)
        for state, action, reward, next_state in self.memory:
            count[action] += 1
        print(count)
        return count

    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = self.reward
            next_state_tensor = torch.tensor(
                np.expand_dims(next_state, 0), dtype=torch.float32
            ).to(DEVICE)
            state_tensor = torch.tensor(
                np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True
            ).to(DEVICE)
            target = self.reward + self.gamma * torch.max(
                self.forward(next_state_tensor)[0]
            )
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][action] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()
