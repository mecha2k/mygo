from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.encoders.simple import SimpleEncoder
from dlgo.dataprocess.myprocessor import GoDataProcessor
from dlgo.neuralnet import small

from keras.models import Sequential
from keras.layers.core import Dense


def train_consolidate():
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 10

    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    # encoder = SimpleEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())
    x_train, y_train = processor.load_go_data("train", num_games)
    x_test, y_test = processor.load_go_data("test", num_games)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = small.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model.summary()

    epochs = 1
    batch_size = 128
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    train_consolidate()
