import argparse
import logging
import gym

import numpy as np

from random import seed
from random import uniform
from timeit import default_timer as timer

from animate_frames import print_frames


def parse_arguments():
    """Setup CLI interface
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-ne",
        "--num_episodes",
        type=int,
        default=40000,
        help="number of training episodes",
    )

    parser.add_argument(
        "-ip",
        "--path_input",
        type=str,
        #  default="qtable_out.txt",
        default="qtable_out.npy",
        help="path to text file to load the table from",
    )

    parser.add_argument(
        "-op",
        "--path_output",
        type=str,
        #  default="qtable_out.txt",
        default="qtable_out.npy",
        help="path to text file to save the table",
    )

    parser.add_argument(
        "-t",
        "--show",
        default=False,
        action="store_true",
        help="show just the animation, needs a valid input table",
    )

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


def run_qlearn(alpha, gamma, epsilon, num_episodes, path_output):
    """Setup an environment, train and evaluate
    """
    #  env = gym.make("Taxi-v2").env
    env = gym.make("Taxi-v3").env

    q_table = learn_qtable(env, alpha, gamma, epsilon, num_episodes)
    evaluate_qtable(env, q_table)
    animate_qtable(env, q_table)
    save_qtable(q_table, path_output)


def learn_qtable(env, alpha, gamma, epsilon, num_episodes):
    """Learn the whole q-table
    """

    obs_space = env.observation_space.n
    act_space = env.action_space.n
    print(f"Working on a {obs_space}x{act_space} observation-action space")

    q_table = np.zeros([obs_space, act_space])

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    start = timer()

    # just for good looking logs...
    for i in range(1, num_episodes + 1):
        # reset the env in a random state
        state = env.reset()

        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            # pick the next action
            if uniform(0, 1) < epsilon:
                # explore action space
                action = env.action_space.sample()
            else:
                # exploit learned values
                action = np.argmax(q_table[state])

            # update the env with the selected action
            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            # compute the new value
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            # update the table
            q_table[state, action] = new_value

            # keep track of errors
            if reward == -10:
                penalties += 1

            # update the env
            state = next_state
            epochs += 1

        #  if i % 1000 == 0:
        # it's interesting the speed up in the beginning, as the agent learns
        if i <= 4000 and i % 100 == 0:
            end = timer()
            print(f"At episode {i:5d}, took {end-start:.6f} s")
            start = end

    print("Done")
    return q_table


def evaluate_qtable(env, q_table):
    """Evaluate a q-table
    """
    total_epochs, total_penalties = 0, 0
    num_test_episodes = 100

    for i in range(num_test_episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

            # sometimes the taxi gets stuck
            #  if epochs % 1000 == 0:
            #  print(f"WTF i {i} epochs {epochs}")
            if epochs == 10000:
                print(f"WTF i {i} epochs {epochs}")
                break

        if i % 10 == 0:
            print(f"At test episode {i}")

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {num_test_episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / num_test_episodes}")
    print(f"Average penalties per episode: {total_penalties / num_test_episodes}")


def animate_qtable(env, q_table):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    frames = []  # for animation
    done = False

    while not done:
        action = np.argmax(q_table[state])
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
        if epochs == 10000:
            print(f"WTF i {i} epochs {epochs}")
            break

    print(f"Timesteps taken: {epochs}")
    print(f"Penalties incurred: {penalties}")

    print_frames(frames, 0.3)


def save_qtable(q_table, path_output):
    """Save the provided q-table in a file
    """

    #  np.savetxt(path_output, q_table)
    np.save(path_output, q_table)


def try_table(path_input):
    #  env = gym.make("Taxi-v2").env
    env = gym.make("Taxi-v3").env
    q_table = np.load(path_input)
    animate_qtable(env, q_table)


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
    num_episodes = args.num_episodes
    path_output = args.path_output
    path_input = args.path_input
    show = args.show

    recap = f"python3 qlearn_taxi.py"
    recap += f" --num_episodes {num_episodes}"
    recap += f" --path_output {path_output}"
    recap += f" --path_input {path_input}"
    if show:
        recap += f" --show"
    recap += f" --seed {myseed}"

    logmain = logging.getLogger(f"c.{__name__}.main")
    logmain.info(recap)

    # hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    #  num_episodes = 100000
    #  num_episodes = 20000
    #  num_episodes = 40000

    if show:
        try_table(path_input)
    else:
        run_qlearn(alpha, gamma, epsilon, num_episodes, path_output)


if __name__ == "__main__":
    main()
