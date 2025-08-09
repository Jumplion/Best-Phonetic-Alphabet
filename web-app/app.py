from flask import Flask, request, jsonify, render_template
import sys
import os

from BestPhoneticAlphabet import get_candidates_from_db
from scoring import analyze_alphabet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grade', methods=['POST'])
def grade_alphabet():
    data = request.get_json()
    result = analyze_alphabet(data)
    return jsonify(result)

# In your web app
@app.route('/api/candidates')
def get_candidates():
    metric = request.args.get('metric', 'score')
    limit = int(request.args.get('limit', 10))
    
    candidates = get_candidates_from_db(metric_type=metric, limit=limit)
    return candidates.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
