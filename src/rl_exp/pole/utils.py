from collections import namedtuple
import random
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
# a named tuple representing a single transition in our environment. It
# essentially maps (state, action) pairs to their (next_state, reward) result,
# with the state being the screen difference image as described later on.


class ReplayMemory(object):
    """cyclic buffer of bounded size, holds the transitions observed recently

    It also implements a .sample() method for selecting a random batch of
    transitions for training.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        # if it is the first time filling the buffer, make it grow first
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # a convolutional neural network that takes in the difference between the
    # current and previous screen patches. It has two outputs, representing
    # Q(s,left) and Q(s,right) (where s is the input to the network). In effect,
    # the network is trying to predict the expected return of taking each action
    # given the current input.

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # TODO investigate what the hell this is returning
        return self.head(x.view(x.size(0), -1))
