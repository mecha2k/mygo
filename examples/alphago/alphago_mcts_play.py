from dlgo.agent import load_prediction_agent, load_policy_agent, AlphaGoMCTS
from dlgo.reinforce import load_value_agent
from dlgo.httpfront import get_web_app
from dlgo.kerasutil import init_gpus

from dotenv import load_dotenv
import h5py
import os


def alphago_mcts():
    load_dotenv(verbose=True)
    AlphaGo_dir = os.getenv("ALPHAGO_DIR")

    fast_policy = load_prediction_agent(h5py.File(AlphaGo_dir + "/my_alphago_sl_policy.h5", "r"))
    strong_policy = load_policy_agent(h5py.File(AlphaGo_dir + "/my_alphago_rl_policy.h5", "r"))
    value = load_value_agent(h5py.File(AlphaGo_dir + "/my_alphago_value.h5", "r"))

    return AlphaGoMCTS(strong_policy, fast_policy, value)


if __name__ == "__main__":
    init_gpus()

    alphago = alphago_mcts()
    web_app = get_web_app({"predict": alphago})
    web_app.run()
