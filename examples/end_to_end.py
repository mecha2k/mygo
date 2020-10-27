import h5py
import tensorflow as tf
import os

from keras.models import Sequential
from keras.layers import Dense
from multiprocessing import freeze_support
from dotenv import load_dotenv

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.dataprocess.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.httpfront import get_web_app
from dlgo.neuralnet import large


def end_to_end():
    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())

    X, y = processor.load_go_data(num_samples=10)
    X = tf.transpose(X, perm=[0, 2, 3, 1])

    input_shape = (go_board_rows, go_board_cols, encoder.num_planes)
    model = Sequential()
    network_layers = large.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(nb_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
    )
    model.summary()

    model.fit(X, y, batch_size=128, epochs=1, verbose=1)

    load_dotenv(verbose=True)
    AGENT_DIR = os.getenv("AGENT_DIR")

    deep_learning_bot = DeepLearningAgent(model, encoder)
    deep_learning_bot.serialize(h5py.File(AGENT_DIR + "/deep_bot.h5", "w"))

    # model_file = h5py.File("../agents/deep_bot.h5", "r")
    # bot_from_file = load_prediction_agent(model_file)
    #
    # web_app = get_web_app({"predict": bot_from_file})
    # web_app.run()


if __name__ == "__main__":
    freeze_support()
    end_to_end()
