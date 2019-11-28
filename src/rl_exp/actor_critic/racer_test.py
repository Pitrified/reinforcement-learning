import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from random import seed as rseed
from timeit import default_timer as timer
from collections import deque

import gym
import gym_racer

from racer_agent import Agent
from rl_exp.rl_utils.plots import plot_scores


def parse_arguments():
    """Setup CLI interface
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-mft",
        "--model_file_template",
        type=str,
        default="bestof/best_{}_46k.pth",
        help="path to input models to use, with template slots for actor/critic",
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
    recap = f"python3 racer_test.py"
    for a, v in args._get_kwargs():
        if a == "rand_seed":
            recap += f" --rand_seed {myseed}"
        else:
            recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def run_racer_test(args):
    """
    """
    logg = logging.getLogger(f"c.{__name__}.run_racer_test")
    logg.debug(f"Running agent")

    ##################
    # create the env #
    ##################

    dir_step = 3
    speed_step = 0.3
    malus_standing_still = -1
    reset_map = True

    mode = "human"
    sat = "lidar"
    sensor_array_params = {}
    sensor_array_params["ray_num"] = 10
    sensor_array_params["ray_step"] = 10
    sensor_array_params["ray_sensors_per_ray"] = 20
    sensor_array_params["ray_max_angle"] = 80

    racer_env = gym.make(
        "racer-v0",
        dir_step=dir_step,
        speed_step=speed_step,
        malus_standing_still=malus_standing_still,
        reset_map=reset_map,
        sensor_array_type=sat,
        render_mode=mode,
        sensor_array_params=sensor_array_params,
    )

    action_size = 2
    obs_space = racer_env.observation_space
    state_size = obs_space.shape[0]

    ##################
    # load the agent #
    ##################

    BUFFER_SIZE = int(1e6)  # replay buffer size
    BATCH_SIZE = 128  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR_ACTOR = 1e-4  # learning rate of the actor
    LR_CRITIC = 3e-4  # learning rate of the critic
    WEIGHT_DECAY = 0.0001  # L2 weight decay

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = Agent(
        state_size,
        action_size,
        device,
        LR_ACTOR,
        LR_CRITIC,
        BUFFER_SIZE,
        BATCH_SIZE,
        WEIGHT_DECAY,
        GAMMA,
        TAU,
    )

    model_file_template = args.model_file_template

    actor_file_best = model_file_template.format("actor")
    critic_file_best = model_file_template.format("critic")

    agent.actor_local.load_state_dict(torch.load(actor_file_best))
    agent.critic_local.load_state_dict(torch.load(critic_file_best))

    ##################
    # test the agent #
    ##################

    action_cutoff = 0.4
    max_frames = 1000
    show_rate = 10

    scores_deque = deque(maxlen=show_rate)
    scores = []
    scores_moving_average = []

    while True:
        state = racer_env.reset()
        agent.reset()
        score = 0

        for f in range(max_frames):
            action = agent.act(state)
            disc_action = np.zeros((2,), dtype=np.uint8)
            disc_action = np.where(action < -action_cutoff, 1, disc_action)
            disc_action = np.where(action > action_cutoff, 2, disc_action)
            next_state, reward, done, _ = racer_env.step(disc_action)

            racer_env.render(mode=mode, reward=reward)

            score += reward
            state = next_state

            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        scores_moving_average.append(np.mean(scores_deque))

        print(f"Score: {score:< 9.0f} survived {f+1:< 5d}")
        plot_scores(scores, p_title="Testing...")


if __name__ == "__main__":
    args = setup_env()
    run_racer_test(args)
