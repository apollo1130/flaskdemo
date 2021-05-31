import random
import json
from flask import Flask, jsonify,request
from flask_cors import CORS
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import time

app = Flask(__name__, static_url_path='');

CORS(app)
@app.route("/", methods=["GET"])
def home():
    return '<h1>Demo</h1>'
CORS(app)
@app.route("/bot", methods=["POST"])

#response
def response():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()    
    query = dict(request.form)['query']
    res = query 
    res =  tokenize(res)
    
    X = bag_of_words(res, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                
                return jsonify({"response" : random.choice(intent['responses'])})
    else:
        return jsonify({"response" : " I do not understand..."})