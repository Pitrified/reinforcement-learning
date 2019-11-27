import pytest

from random import randint
from replay_memory import ReplayBuffer


@pytest.fixture
def replay_buffer():
    """
    """
    action_size = 10
    buffer_size = 20
    batch_size = 5
    seed = 42
    device = "cpu"
    rebuf = ReplayBuffer(action_size, buffer_size, batch_size, seed, device)
    return rebuf


def test_creation(replay_buffer):
    """create an empty replay_buffer
    """
    assert len(replay_buffer) == 0


def test_add_element(replay_buffer):
    """add an experience

    e = self.experience(state, action, reward, next_state, done)
    """
    state = 10
    action = 12
    reward = 14
    next_state = 16
    done = False
    replay_buffer.add(state, action, reward, next_state, done)
    assert len(replay_buffer) == 1


def add_element(rebuf):
    """
    """
    state = 10
    action = 12
    reward = 14
    next_state = 16
    done = False
    rebuf.add(state, action, reward, next_state, done)


def test_sample_elements(replay_buffer):
    """sample an experience
    """
    add_element(replay_buffer)
    add_element(replay_buffer)
    add_element(replay_buffer)
    add_element(replay_buffer)
    add_element(replay_buffer)
    add_element(replay_buffer)
    add_element(replay_buffer)
    states, actions, rewards, next_states, dones = replay_buffer.sample()
    assert len(states) == replay_buffer.batch_size
