from flask import Flask, request, render_template
import json

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/user")
def userhome():
    return "user home."


@app.route("/profile/<name>")
def userprofile(name):
    return "user profile: " + name


@app.route("/post/<int:postId>")
def userId(postId):
    return f"<h2>user post id: {postId} </h2>"


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return "login POST method"
    else:
        return "login GET method"


@app.route("/json", methods=["GET", "POST"])
def jsondata():
    # maybe = request.get_json(silent=True, cache=False, force=True)
    maybe = request.json
    if maybe:
        data = json.dumps(maybe)
    else:
        data = "no json"
    print(maybe)
    print(data)
    return "good job"


@app.route("/hello/")
@app.route("/hello/<name>")
def hello(name=None):
    return render_template("hello.html", name=name)


if __name__ == "__main__":
    app.run(debug=True)
