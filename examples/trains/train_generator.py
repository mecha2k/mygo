import h5py
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from multiprocessing import freeze_support
import os
import time

from dlgo.dataprocess.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.neuralnet import large


def train_generator():
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 1000

    start_time = time.time()

    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    encoder = AlphaGoEncoder()
    processor = GoDataProcessor(encoder=encoder.name())

    train_gen = processor.load_go_data("train", num_games, use_generator=True)
    test_gen = processor.load_go_data("test", num_games, use_generator=True)

    procpath = processor.data_dir + "/checkpoints"
    modelfile = procpath + "/my_deep_bot.h5"
    if os.path.isfile(modelfile):
        myagent = load_prediction_agent(h5py.File(modelfile, "r"))
        model = myagent.model
    else:
        input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
        network_layers = large.layers(input_shape)
        model = Sequential()
        for layer in network_layers:
            model.add(layer)
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model.summary()

    epochs = 5
    batch_size = 128

    train_num = train_gen.get_num_samples(batch_size, num_classes)
    print("train samples: ", train_num)

    if not os.path.isdir(procpath):
        os.makedirs(procpath)
    filepath = procpath + "/model_epoch_{epoch:02d}-{loss:.2f}.h5"
    callbacks = [
        ModelCheckpoint(
            filepath=filepath,
            monitor="accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            verbose=1,
        )
    ]

    model.fit(
        train_gen.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=train_gen.get_num_samples() / batch_size,
        validation_data=test_gen.generate(batch_size, num_classes),
        validation_steps=test_gen.get_num_samples() / batch_size,
        callbacks=callbacks,
    )

    model.evaluate(
        test_gen.generate(batch_size, num_classes),
        steps=test_gen.get_num_samples() / batch_size,
    )

    deep_learning_bot = DeepLearningAgent(model, encoder)
    deep_learning_bot.serialize(h5py.File(modelfile, "w"))

    print(f"model file saved: {modelfile}")
    print(f"elapsed time ({encoder.name()}): {time.time() - start_time} sec.")


if __name__ == "__main__":
    train_generator()
