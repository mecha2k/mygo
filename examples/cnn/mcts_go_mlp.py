from __future__ import print_function

# TODO: tell readers where to put feature and label files in text

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(123)
X = np.load("../generated/features-40k.npy")
Y = np.load("../generated/labels-40k.npy")
samples = X.shape[0]
board_size = 9 * 9

X = X.reshape(samples, board_size)
Y = Y.reshape(samples, board_size)

train_samples = int(0.9 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

model = Sequential()
model.add(Dense(1000, activation="sigmoid", input_shape=(board_size,)))
model.add(Dense(500, activation="sigmoid"))
model.add(Dense(board_size, activation="sigmoid"))
model.summary()

model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
model.fit(X_train, Y_train, batch_size=64, epochs=15, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# fmt: off
test_board = np.array([[
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 1, -1,  1, -1, 0, 0, 0, 0,
    0, 1, -1,  1, -1, 0, 0, 0, 0,
    0, 0,  1, -1,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
]])
# fmt: on

move_probs = model.predict(test_board)[0]
i = 0
for row in range(9):
    row_formatted = []
    for col in range(9):
        row_formatted.append("{:.3f}".format(move_probs[i]))
        i += 1
    print(" ".join(row_formatted))
