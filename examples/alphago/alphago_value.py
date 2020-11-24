from dlgo.neuralnet.alphago import alphago_model
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.reinforce import ValueAgent, load_experience
from dlgo.kerasutil import init_gpus

from dotenv import load_dotenv
import h5py
import os


def main():
    load_dotenv(verbose=True)
    AlphaGo_dir = os.getenv("ALPHAGO_DIR")

    rows, cols = 19, 19
    encoder = AlphaGoEncoder()
    input_shape = (encoder.num_planes, rows, cols)
    alphago_value_network = alphago_model(input_shape)

    alphago_value = ValueAgent(alphago_value_network, encoder)
    experience = load_experience(h5py.File(AlphaGo_dir + "/my_alphago_rl_experience.h5", "r"))

    alphago_value.train(experience)

    with h5py.File(AlphaGo_dir + "/my_alphago_value.h5", "w") as value_agent_out:
        alphago_value.serialize(value_agent_out)


if __name__ == "__main__":
    init_gpus()
    main()
