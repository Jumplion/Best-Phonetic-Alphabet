from flask import Flask, request, jsonify, render_template
import sys
import os

# Requires that this line comes before the following imports!!!!!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BestPhoneticAlphabet import get_candidates_from_db
from scoring import analyze_alphabet

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grade', methods=['POST'])
def grade_alphabet():
    data = request.get_json()
    print(f"Received data: {data}")
    result = analyze_alphabet(data)
    print(f"Analysis result: {result}")
    print(f"JSONIFIED: {jsonify(result)}")
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
