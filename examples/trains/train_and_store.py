from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import to_categorical
import tensorflow as tf

from dlgo.dataprocess.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.neuralnet.large import layers

go_board_rows, go_board_cols = 19, 19
nb_classes = go_board_rows * go_board_cols

encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name(), data_directory="../data")

input_channels = encoder.num_planes
input_shape = (go_board_rows, go_board_cols, input_channels)

X, y = processor.load_go_data(num_samples=10)
X = tf.transpose(X, perm=[0, 2, 3, 1])
# y = to_categorical(y, nb_classes)

model = Sequential()
network_layers = layers(input_shape)
for layer in network_layers:
    model.add(layer)
model.add(Dense(nb_classes, activation="softmax"))
model.compile(
    loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
)
model.summary()

model.fit(X, y, batch_size=128, epochs=1, verbose=1)

weight_file = "../../agents/weights.hd5"
model.save_weights(weight_file, overwrite=True)
model_file = "../../agents/model.yml"
with open(model_file, "w") as yml:
    model_yaml = model.to_yaml()
    yml.write(model_yaml)