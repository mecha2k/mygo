import h5py
import os
import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from dotenv import load_dotenv

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.dataprocess.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.httpfront import get_web_app
from dlgo.neuralnet import large, small
from dlgo.agent.naive import RandomBot


def end_to_end():
    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())

    X, y = processor.load_go_data(num_samples=10)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = small.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(nb_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
    model.summary()

    model.fit(X, y, batch_size=128, epochs=1, verbose=1)

    load_dotenv(verbose=True)
    AGENT_DIR = os.getenv("AGENT_DIR")
    end_bot_name = AGENT_DIR + "/end_to_end.h5"

    deep_learning_bot = DeepLearningAgent(model, encoder)
    deep_learning_bot.serialize(h5py.File(end_bot_name, "w"))

    model_file = h5py.File(end_bot_name, "r")
    bot_from_file = load_prediction_agent(model_file)
    bot_from_file.model.summary()

    x_pred = X[100]
    x_pred = x_pred.reshape((1, encoder.num_planes, go_board_rows, go_board_cols))

    y_pred = bot_from_file.model.predict(x_pred, verbose=1)
    print(y_pred)

    web_app = get_web_app({"predict": bot_from_file})
    web_app.run()


if __name__ == "__main__":
    end_to_end()
