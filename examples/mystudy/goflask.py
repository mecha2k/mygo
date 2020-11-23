import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from dlgo import agent
from dlgo import goboard_fast as goboard
from dlgo.utils import coords_from_point
from dlgo.utils import point_from_coords


def get_web_app(bot_map):
    load_dotenv(verbose=True)
    static_path = os.getenv("STATIC_DIR")
    app = Flask(__name__, static_folder=static_path, static_url_path="/static")
    CORS(app)

    @app.route("/user")
    def userhome():
        return jsonify(user="joe")

    @app.route("/select-move/<bot_name>", methods=["POST"])
    def select_move(bot_name):
        content = request.json
        board_size = content["board_size"]
        game_state = goboard.GameState.new_game(board_size)

        for move in content["moves"]:
            if move == "pass":
                next_move = goboard.Move.pass_turn()
            elif move == "resign":
                next_move = goboard.Move.resign()
            else:
                next_move = goboard.Move.play(point_from_coords(move))
            game_state = game_state.apply_move(next_move)
        bot_agent = bot_map[bot_name]
        bot_move = bot_agent.select_move(game_state)
        if bot_move.is_pass:
            bot_move_str = "pass"
        elif bot_move.is_resign:
            bot_move_str = "resign"
        else:
            bot_move_str = coords_from_point(bot_move.point)
        return jsonify({"bot_move": bot_move_str, "diagnostics": bot_agent.diagnostics()})

    return app


if __name__ == "__main__":
    myagent = agent.RandomBot()
    web_app = get_web_app({"random": myagent})
    web_app.run(debug=True)
