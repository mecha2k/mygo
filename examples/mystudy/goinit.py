from dlgo import gotypes, goboard, agent
from dlgo.utils import print_board, print_move, point_from_coords


def main():
    point = gotypes.Point(row=3, col=2)
    print(point)
    print(point.row, point.col)

    board_size = 6
    game = goboard.GameState.new_game(board_size)
    # bot = agent.RandomBot()

    print(chr(27) + "[2J")
    print_board(game.board)


if __name__ == "__main__":
    main()
