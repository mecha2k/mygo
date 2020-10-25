from __future__ import print_function

from dlgo import agent
from dlgo import goboard_slow as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords


def main():
    board_size = 7
    game = goboard.GameState.new_game(board_size)
    bot = agent.RandomBot()

    # while not game.is_over():
    #     print(chr(27) + "[2J")
    #     print_board(game.board)
    #     if game.next_player == gotypes.Player.black:
    #         human_move = input("stone: ")
    #         point = point_from_coords(human_move.strip())
    #         move = goboard.Move.play(point)
    #     else:
    #         move = bot.select_move(game)
    #
    #     print_move(game.next_player, move)
    #     game = game.apply_move(move)

    input_moves = ["C2", "B2", "C1", "B1", "E2", "D2", "D3", "C3", "D4", "C4", "E3"]
    input_moves = ["C3", "B3", "C4", "B4", "D4", "C5", "E3", "D3", "F3", "D5", "F4", "E4", "D2"]

    for ch in input_moves:
        print(chr(27) + "[2J")
        print_board(game.board)
        human_move = ch
        point = point_from_coords(human_move.strip())
        move = goboard.Move.play(point)
        print_move(game.next_player, move)
        game = game.apply_move(move)

    print(chr(27) + "[2J")
    print_board(game.board)


if __name__ == "__main__":
    main()
