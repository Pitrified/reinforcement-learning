from time import sleep


def print_frames(frames, frame_len=0.1):
    """Animate the frames provided

    TODO: clear terminal of previous frame
    TODO: output a gif
    """
    for i, frame in enumerate(frames):
        #  print(frame["frame"].getvalue())
        print(frame["frame"])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(frame_len)
