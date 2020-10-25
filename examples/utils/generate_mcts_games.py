import argparse
import numpy as np

from dlgo.encoders import get_encoder_by_name
from dlgo import goboard_fast as goboard
from dlgo import mcts
from dlgo.utils import print_board, print_move


def generate_game(board_size, rounds, max_moves, temperature):
    boards, moves = [], []
    encoder = get_encoder_by_name("oneplane", board_size)
    game = goboard.GameState.new_game(board_size)
    bot = mcts.MCTSAgent(rounds, temperature)

    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game)
        if move.is_play:
            boards.append(encoder.encode(game))
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot)

        print_move(game.next_player, move)
        game = game.apply_move(move)
        num_moves += 1
        if num_moves > max_moves:
            break

    return np.array(boards), np.array(moves)


# <1> In `boards` we store encoded board state, `moves` is for encoded moves.
# <2> We initialize a OnePlaneEncoder by name with given board size.
# <3> An new game of size `board_size` is instantiated.
# <4> A Monte Carlo tree search agent with specified number of rounds and temperature will serve as our bot.
# <5> The next move is selected by the bot.
# <6> The encoded board situation is appended to `boards`.
# <7> The one-hot-encoded next move is appended to `moves`.
# <8> Afterwards the bot move is applied to the board.
# <9> We continue with the next move, unless the maximum number of moves has been reached.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", "-b", type=int, default=9)
    parser.add_argument("--rounds", "-r", type=int, default=1000)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--max-moves", "-m", type=int, default=60, help="Max moves per game.")
    parser.add_argument("--num-games", "-n", type=int, default=10)
    parser.add_argument("--board-out", default="../generated/features.npy")
    parser.add_argument("--move-out", default="../generated/labels.npy")
    args = parser.parse_args()

    xs = []
    ys = []

    for i in range(args.num_games):
        print("Generating game %d/%d..." % (i + 1, args.num_games))
        x, y = generate_game(args.board_size, args.rounds, args.max_moves, args.temperature)
        xs.append(x)
        ys.append(y)

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    np.save(args.board_out, x)
    np.save(args.move_out, y)


if __name__ == "__main__":
    main()

# <1> This application allows some customization via command line arguments.
# <2> For the specified number of games we generate game dataprocess.
# <3> After all games have been generated, we concatenate features and labels, respectively.
# <4> We store feature and label dataprocess to separate files, as specified by the command line options.
