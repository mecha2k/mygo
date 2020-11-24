from dlgo.agent.pg import PolicyAgent
from dlgo.agent.predict import load_prediction_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.reinforce.simulate import experience_simulation
from dlgo.kerasutil import init_gpus

from dotenv import load_dotenv
import h5py
import os


def main():
    load_dotenv(verbose=True)
    AlphaGo_dir = os.getenv("ALPHAGO_DIR")
    alphagofile = AlphaGo_dir + "/my_alphago_sl_policy.h5"

    encoder = AlphaGoEncoder()

    sl_agent = load_prediction_agent(h5py.File(alphagofile, "r"))
    sl_opponent = load_prediction_agent(h5py.File(alphagofile, "r"))
    sl_agent.model.summary()

    alphago_rl_agent = PolicyAgent(sl_agent.model, encoder)
    opponent = PolicyAgent(sl_opponent.model, encoder)

    num_games = 10
    experience = experience_simulation(num_games, alphago_rl_agent, opponent)

    alphago_rl_agent.train(experience)

    with h5py.File(AlphaGo_dir + "/my_alphago_rl_policy.h5", "w") as rl_agent_out:
        alphago_rl_agent.serialize(rl_agent_out)

    with h5py.File(AlphaGo_dir + "/my_alphago_rl_experience.h5", "w") as exp_out:
        experience.serialize(exp_out)


if __name__ == "__main__":
    init_gpus()
    main()
