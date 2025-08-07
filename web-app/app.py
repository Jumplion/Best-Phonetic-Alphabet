from flask import Flask, request, jsonify, render_template
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scoring import analyze_alphabet

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grade', methods=['POST'])
def grade_alphabet():
    data = request.get_json()
    result = analyze_alphabet(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
