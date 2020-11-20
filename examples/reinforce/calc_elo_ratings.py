import argparse
import h5py

from dlgo import elo
from dlgo import reinforce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", "-g", type=int, default=10)
    parser.add_argument("--board-size", "-b", type=int, default=19)
    parser.add_argument("agents", nargs="+")

    args = parser.parse_args()

    agents = [
        # agent.load_policy_agent(h5py.File(filename))
        reinforce.load_q_agent(h5py.File(filename, "r"))
        for filename in args.agents
    ]
    for a in agents:
        a.set_temperature(0.02)

    ratings = elo.calculate_ratings(agents, args.num_games, args.board_size)

    for filename, rating in zip(args.agents, ratings):
        print("%s %d" % (filename, rating))


if __name__ == "__main__":
    main()
