from dlgo.dataprocess.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.neuralnet import small

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from multiprocessing import freeze_support


def train_generator():
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 10

    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())
    x_train, y_train = processor.load_go_data("train", num_games, use_generator=False)
    x_test, y_test = processor.load_go_data("test", num_games, use_generator=False)
    x_train = tf.transpose(x_train, perm=[0, 2, 3, 1])
    x_test = tf.transpose(x_test, perm=[0, 2, 3, 1])

    input_shape = (go_board_rows, go_board_cols, encoder.num_planes)
    network_layers = small.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
    )
    model.summary()

    epochs = 1
    batch_size = 128
    # gen = generator.generate(batch_size, num_classes)
    # print(generator.get_num_samples())
    #
    # model.fit_generator(
    #     generator=generator.generate(batch_size, num_classes),
    #     epochs=epochs,
    #     steps_per_epoch=generator.get_num_samples() / batch_size,
    #     validation_data=test_generator.generate(batch_size, num_classes),
    #     validation_steps=test_generator.get_num_samples() / batch_size,
    #     callbacks=[ModelCheckpoint("checkpoints/small_model_epoch_{epoch}.h5")],
    # )
    #
    # model.evaluate_generator(
    #     generator=test_generator.generate(batch_size, num_classes),
    #     steps=test_generator.get_num_samples() / batch_size,
    # )


if __name__ == "__main__":
    freeze_support()
    train_generator()
