import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from random import seed as rseed
from timeit import default_timer as timer
from collections import deque

from racer_agent import Agent
import gym
import gym_racer


def parse_arguments():
    """Setup CLI interface
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i",
        "--path_input",
        type=str,
        default="hp.jpg",
        help="path to input image to use",
    )

    parser.add_argument(
        "-s", "--rand_seed", type=int, default=-1, help="random seed to use"
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_logger(logLevel="DEBUG"):
    """Setup logger that outputs to console for the module
    """
    logroot = logging.getLogger("c")
    logroot.propagate = False
    logroot.setLevel(logLevel)

    module_console_handler = logging.StreamHandler()

    #  log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #  log_format_module = "%(name)s - %(levelname)s: %(message)s"
    #  log_format_module = '%(levelname)s: %(message)s'
    #  log_format_module = '%(name)s: %(message)s'
    log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logroot.addHandler(module_console_handler)

    logging.addLevelName(5, "TRACE")
    # use it like this
    # logroot.log(5, 'Exceedingly verbose debug')

    # example log line
    logg = logging.getLogger(f"c.{__name__}.setup_logger")
    logg.debug(f"Done setting up logger")


def setup_env():
    setup_logger()

    args = parse_arguments()

    # setup seed value
    if args.rand_seed == -1:
        myseed = 1
        myseed = int(timer() * 1e9 % 2 ** 32)
    else:
        myseed = args.rand_seed
    rseed(myseed)
    np.random.seed(myseed)

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = f"python3 racer_run.py"
    for a, v in args._get_kwargs():
        if a == "rand_seed":
            recap += f" --rand_seed {myseed}"
        else:
            recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def train_agent(env, agent, num_episodes, max_t, model_file_template):
    """
    """
    actor_file = model_file_template.format("actor")
    critic_file = model_file_template.format("critic")

    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print(
            "\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}".format(
                i_episode, np.mean(scores_deque), score
            ),
            end="",
        )
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )

    return scores


def run_racer_run(args):
    """
    """
    logg = logging.getLogger(f"c.{__name__}.run_racer_run")
    logg.debug(f"Running agent")

    BUFFER_SIZE = int(1e6)  # replay buffer size
    BATCH_SIZE = 128  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR_ACTOR = 1e-4  # learning rate of the actor
    LR_CRITIC = 3e-4  # learning rate of the critic
    WEIGHT_DECAY = 0.0001  # L2 weight decay

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mode = "console"
    sat = "lidar"
    sensor_array_params = {}
    sensor_array_params["ray_num"] = 7
    sensor_array_params["ray_step"] = 10
    sensor_array_params["ray_sensors_per_ray"] = 13
    sensor_array_params["ray_max_angle"] = 130
    racer_env = gym.make(
        "racer-v0",
        sensor_array_type=sat,
        render_mode=mode,
        sensor_array_params=sensor_array_params,
    )

    act_space = racer_env.action_space
    logg.debug(f"Action Space {act_space}")
    logg.debug(f"Action Space {act_space.shape}")
    logg.debug(f"Action Space {act_space.nvec}")
    action_size = np.prod(act_space.nvec)

    obs_space = racer_env.observation_space
    logg.debug(f"State Space {obs_space}")
    logg.debug(f"State Space {obs_space.shape}")
    state_size = obs_space.shape[0]

    random_seed = args.rand_seed

    model_file_template = "checkpoint_{}.pth"

    agent = Agent(
        state_size,
        action_size,
        random_seed,
        device,
        LR_ACTOR,
        LR_CRITIC,
        BUFFER_SIZE,
        BATCH_SIZE,
        WEIGHT_DECAY,
        GAMMA,
        TAU,
    )

    num_episodes = 2000
    max_t = 700
    scores = train_agent(racer_env, agent, num_episodes, max_t, model_file_template)


if __name__ == "__main__":
    args = setup_env()
    run_racer_run(args)
