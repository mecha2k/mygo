import argparse
import h5py
import os
from dotenv import load_dotenv

from dlgo import agent
from dlgo import httpfront
from dlgo import mcts
from dlgo import reinforce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-address", default="127.0.0.1")
    parser.add_argument("--port", "-p", type=int, default=5000)
    parser.add_argument("--pg-agent")
    parser.add_argument("--predict-agent")
    parser.add_argument("--q-agent")
    parser.add_argument("--ac-agent")

    args = parser.parse_args()

    bots = {"mcts": mcts.MCTSAgent(800, temperature=0.7)}
    if args.pg_agent:
        bots["pg"] = agent.load_policy_agent(h5py.File(args.pg_agent))
    if args.predict_agent:
        bots["predict"] = agent.load_prediction_agent(h5py.File(args.predict_agent))
    if args.q_agent:
        q_bot = reinforce.load_q_agent(h5py.File(args.q_agent))
        q_bot.set_temperature(0.01)
        bots["q"] = q_bot
    if args.ac_agent:
        ac_bot = reinforce.load_ac_agent(h5py.File(args.ac_agent))
        ac_bot.set_temperature(0.05)
        bots["ac"] = ac_bot

    load_dotenv(verbose=True)
    static_path = os.getenv("AGENT_DIR")
    bot_file = static_path + "/deep_bot.h5"
    model_file = h5py.File(bot_file, "r")
    bot_from_file = agent.load_prediction_agent(model_file)
    bots = {"predict": bot_from_file}

    web_app = httpfront.get_web_app(bots)
    web_app.run(host=args.bind_address, port=args.port, threaded=False, debug=True)


if __name__ == "__main__":
    main()
