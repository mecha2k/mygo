import importlib

__all__ = ["Encoder", "get_encoder_by_name"]


class Encoder:
    def name(self):
        raise NotImplementedError()

    def encode(self, game_state):
        raise NotImplementedError()

    def encode_point(self, point):
        raise NotImplementedError()

    def decode_point_index(self, index):
        raise NotImplementedError()

    def num_points(self):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()


# <1> Lets us support logging or saving the name of the encoder our model is using.
# <2> Turn a Go board into a numeric dataprocess.
# <3> Turn a Go board point into an integer index.
# <4> Turn an integer index back into a Go board point.
# <5> Number of points on the board, i.e. board width times board height.
# <6> Shape of the encoded board structure.


def get_encoder_by_name(name, board_size):
    if isinstance(board_size, int):
        board_size = (board_size, board_size)
    module = importlib.import_module("dlgo.encoders." + name)
    constructor = getattr(module, "create")
    return constructor(board_size)


# <1> We can create encoder instances by referencing their name.
# <2> If board_size is one integer, we create a square board from it.
# <3> Each encoder implementation will have to provide a "create" function that provides an instance.
