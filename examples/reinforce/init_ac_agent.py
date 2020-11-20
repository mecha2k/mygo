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
    agent_out_file = os.getenv("DATA_DIR") + "/reinforce/actor_critic_agent_v1.hdf5"

    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--network", default="large")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--output_file", default=agent_out_file)
    args = parser.parse_args()

    encoder = encoders.get_encoder_by_name("simple", args.board_size)
    board_input = Input(shape=encoder.shape(), name="board_input")

    processed_board = board_input
    network = getattr(dlgo.neuralnet, args.network)
    for layer in network.layers(encoder.shape()):
        processed_board = layer(processed_board)

    policy_hidden_layer = Dense(args.hidden_size, activation="relu")(processed_board)
    policy_output = Dense(encoder.num_points(), activation="softmax")(policy_hidden_layer)

    value_hidden_layer = Dense(args.hidden_size, activation="relu")(processed_board)
    value_output = Dense(1, activation="tanh")(value_hidden_layer)

    model = Model(inputs=[board_input], outputs=[policy_output, value_output])

    new_agent = reinforce.ACAgent(model, encoder)
    with h5py.File(args.output_file, "w") as outf:
        new_agent.serialize(outf)


if __name__ == "__main__":
    main()
