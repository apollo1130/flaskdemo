from flask import Flask, make_response, render_template
from flask_cors import CORS

app = Flask(__name__, static_url_path='');
CORS(app)
@app.route("/", methods=["GET"])
def getHome():
    return '<h1>Demo</h1>'