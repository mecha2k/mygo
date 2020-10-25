from dlgo import minimax
from dlgo import tictactoe

COL_NAMES = "ABC"


def print_board(board):
    print("   A   B   C")
    for row in (1, 2, 3):
        pieces = []
        for col in (1, 2, 3):
            piece = board.get(tictactoe.Point(row, col))
            if piece == tictactoe.Player.x:
                pieces.append("X")
            elif piece == tictactoe.Player.o:
                pieces.append("O")
            else:
                pieces.append(" ")
        print("%d  %s" % (row, " | ".join(pieces)))


def point_from_coords(text):
    col_name = text[0]
    row = int(text[1])
    return tictactoe.Point(row, COL_NAMES.index(col_name) + 1)


def human_v_bot():
    game = tictactoe.GameState.new_game()

    human_player = tictactoe.Player.x
    bot = minimax.MinimaxAgent()

    while not game.is_over():
        print_board(game.board)
        if game.next_player == human_player:
            human_move = input("-- ")
            point = point_from_coords(human_move.strip())
            move = tictactoe.Move(point)
        else:
            move = bot.select_move(game)
        game = game.apply_move(move)

    print_board(game.board)
    winner = game.winner()
    if winner is None:
        print("It's a draw.")
    else:
        print("Winner: " + str(winner))


def bot_v_bot():
    game = tictactoe.GameState.new_game()
    bot = minimax.MinimaxAgent()

    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game)
        game = game.apply_move(move)

    print_board(game.board)
    winner = game.winner()
    if winner is None:
        print("It's a draw.")
    else:
        print("Winner: " + str(winner))


def main():
    human_v_bot()
    # bot_v_bot()


if __name__ == "__main__":
    main()
