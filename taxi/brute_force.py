import argparse
import logging

import gym

import numpy as np
from random import seed
from timeit import default_timer as timer

from animate_frames import print_frames


def parse_arguments():
    """Setup CLI interface
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-s", "--seed", type=int, default=-1, help="random seed to use")

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
    log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logroot.addHandler(module_console_handler)

    logging.addLevelName(5, "TRACE")
    # use it like this
    # logroot.log(5, 'Exceedingly verbose debug')


def run_brute():
    env = gym.make("Taxi-v2").env

    env.s = 328  # set environment to illustration's state

    epochs = 0
    penalties, reward = 0, 0

    frames = []  # for animation

    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append(
            {
                "frame": env.render(mode="ansi"),
                "state": state,
                "action": action,
                "reward": reward,
            }
        )

        epochs += 1

    print(f"Timesteps taken: {epochs}")
    print(f"Penalties incurred: {penalties}")

    print_frames(frames)


def main():
    setup_logger()

    args = parse_arguments()

    # setup seed value
    if args.seed == -1:
        myseed = 1
        myseed = int(timer() * 1e9 % 2 ** 32)
    else:
        myseed = args.seed
    seed(myseed)
    np.random.seed(myseed)

    #  path_input = args.path_input

    recap = f"python3 brute_force.py"
    recap += f" --seed {myseed}"

    logmain = logging.getLogger(f"c.{__name__}.main")
    logmain.info(recap)

    run_brute()


if __name__ == "__main__":
    main()
