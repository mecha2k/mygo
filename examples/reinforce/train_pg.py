import argparse
import h5py
import os
from dotenv import load_dotenv

from dlgo import agent
from dlgo import reinforce


def main():
    load_dotenv(verbose=True)
    DATA_DIR = os.getenv("DATA_DIR")
    AGENT_DIR = os.getenv("AGENT_DIR")
    agent_file = AGENT_DIR + "/my_deep_bot.h5"
    agent_out_file = DATA_DIR + "/reinforce/agent_out_v1.hdf5"
    experience_file = DATA_DIR + "/reinforce/experience_v1.hdf5"

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-agent", default=agent_file)
    parser.add_argument("--agent-out", default=agent_out_file)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--experience", nargs="+", default=[experience_file])
    args = parser.parse_args()

    learning_agent = agent.load_policy_agent(h5py.File(args.learning_agent, "r"))
    for exp_filename in args.experience:
        print("Training with %s..." % exp_filename)
        exp_buffer = reinforce.load_experience(h5py.File(exp_filename, "r"))
        learning_agent.train(exp_buffer, lr=args.lr, batch_size=args.bs)

    with h5py.File(args.agent_out, "w") as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == "__main__":
    main()
