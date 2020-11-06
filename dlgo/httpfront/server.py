import os
import datetime

from flask import Flask
from flask import jsonify
from flask import request, render_template, make_response

from dlgo import goboard_fast as goboard
from dlgo.utils import coords_from_point
from dlgo.utils import point_from_coords

__all__ = ["get_web_app"]


def get_web_app(bot_map):
    here = os.path.dirname(__file__)
    static_path = os.path.join(here, "static")
    template_path = os.path.join(here, "templates")
    app = Flask(
        __name__, static_url_path="", static_folder=static_path, template_folder=template_path,
    )
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["EXPLAIN_TEMPLATE_LOADING"] = True
    print(app.config)
    print(here)
    print(static_path)
    print(template_path)

    @app.route("/")
    def home():
        mycookie = request.cookies.get("cookie")
        if mycookie is None:
            mycookie = "mecha2k"
        mycity = request.args.get("city")
        if mycity is None:
            mycity = "Seoul"

        response = make_response(render_template("play_random_99.html", city=mycity))

        expires = datetime.datetime.now() + datetime.timedelta(days=365)
        response.set_cookie("mycookie", mycookie, expires=expires)
        response.set_cookie("city", mycity, expires=expires)

        return response

    @app.route("/select-move/<bot_name>", methods=["POST"])
    def select_move(bot_name):
        content = request.json
        board_size = content["board_size"]
        game_state = goboard.GameState.new_game(board_size)
        # Replay the game up to this point.
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

        print(bot_move_str)
        # print(bot_agent.diagnostics())
        # return jsonify({"bot_move": bot_move_str, "diagnostics": bot_agent.diagnostics()})
        return jsonify({"bot_move": bot_move_str})

    return app
