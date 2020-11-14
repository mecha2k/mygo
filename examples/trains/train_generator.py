from dlgo.dataprocess.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.neuralnet import small

import os
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from multiprocessing import freeze_support
from dotenv import load_dotenv


def train_generator():
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 1000

    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())

    train_gen = processor.load_go_data("train", num_games, use_generator=True)
    test_gen = processor.load_go_data("test", num_games, use_generator=True)

    input_shape = (go_board_rows, go_board_cols, encoder.num_planes)
    network_layers = small.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model.summary()

    epochs = 100
    batch_size = 128

    train_num = train_gen.get_num_samples(batch_size, num_classes)
    print(train_num)

    load_dotenv(verbose=True)
    AGENT_DIR = os.getenv("AGENT_DIR")
    filepath = AGENT_DIR + "/checkpoints/model_epoch_{epoch:02d}-{loss:.2f}.h5"
    callbacks = [
        ModelCheckpoint(
            filepath=filepath,
            monitor="accuracy",
            save_best_only=True,
            save_weights_only=True,
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


if __name__ == "__main__":
    freeze_support()
    train_generator()
