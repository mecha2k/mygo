import argparse
import h5py
import os
from dotenv import load_dotenv

from keras.layers import Dense, Input
from keras.models import Model

import dlgo.neuralnet
from dlgo import reinforce
from dlgo import encoders


def main():
    load_dotenv(verbose=True)
    DATA_DIR = os.getenv("DATA_DIR")
    agent_out_file = DATA_DIR + "/reinforce/value_agent_v1.hdf5"

    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--network", default="large")
    parser.add_argument("--output_file", default=agent_out_file)
    args = parser.parse_args()

    encoder = encoders.get_encoder_by_name("simple", args.board_size)
    board_input = Input(shape=encoder.shape(), name="board_input")
    # action_input = Input(shape=(encoder.num_points(),), name='action_input')

    processed_board = board_input
    network = getattr(dlgo.neuralnet, args.network)
    for layer in network.layers(encoder.shape()):
        processed_board = layer(processed_board)

    value_output = Dense(1, activation="sigmoid")(processed_board)

    model = Model(inputs=board_input, outputs=value_output)

    new_agent = reinforce.ValueAgent(model, encoder)
    with h5py.File(args.output_file, "w") as outf:
        new_agent.serialize(outf)


if __name__ == "__main__":
    main()
