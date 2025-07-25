from flask import Flask, render_template, request, jsonify
from utils.rag_utils import initialize_rag_system, generate_response
import os

app = Flask(__name__)

# Initialize RAG system at startup
rag_system = initialize_rag_system()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = generate_response(user_message, rag_system['vector_db'], rag_system['memory'])
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)