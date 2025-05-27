from flask import Flask, render_template, request, jsonify
import os
from model.chatbot import ChatBot
from model.preprocessing import input_features_dict, max_encoder_seq_length, num_encoder_tokens
from model.prediction import decode_response

# Initialize chatbot
chatbot = ChatBot(max_encoder_seq_length, num_encoder_tokens, input_features_dict, decode_response)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = chatbot.start_chat(msg)
    return response

if __name__ == '__main__':
    # Check for environment variable PORT for deployment platforms
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
