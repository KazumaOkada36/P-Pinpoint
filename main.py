from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai_parser import parse_business_message
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "p-pinpoint.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), filename)

@app.route("/parse-business", methods=["POST"])
def parse_business():
    data = request.get_json()
    message = data.get("message", "")

    if not message.strip():
        return jsonify({"error": "Message is required"}), 400

    parsed = parse_business_message(message)
    return jsonify(parsed)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    
    try:
        from src.pinpoint.recommender.features import build_feature_matrix
        from src.pinpoint.recommender.model import Recommender
        
        priorities = data.get("user_priorities", [])
        region = data.get("region", "")
        
        feature_matrix, state_names = build_feature_matrix()
        recommender = Recommender()
        recommender.train(feature_matrix, state_names)
        results = recommender.recommend(priorities, top_n=5)
        
        # Convert to lat/lng format
        STATE_COORDS = {
            "California": (34.05, -118.25),
            "Texas": (31.5, -99.0),
            "New York": (40.71, -74.0),
            "Florida": (27.5, -81.5),
            "Washington": (47.6, -122.33),
            "Colorado": (39.73, -104.98),
            "Massachusetts": (42.36, -71.06),
            "Illinois": (41.85, -87.65),
        }
        
        recommendations = []
        for state, score in results:
            coords = STATE_COORDS.get(state, (39.5, -98.35))
            recommendations.append({
                "name": state,
                "score": int(score * 100),
                "lat": coords[0],
                "lng": coords[1]
            })
        
        return jsonify({"received_profile": data, "recommendations": recommendations})
    
    except Exception as e:
        print("Recommend error:", e)
        return jsonify({
            "received_profile": data,
            "recommendations": [
                {"name": "Irvine, CA", "score": 91, "lat": 33.6846, "lng": -117.8265},
                {"name": "Pasadena, CA", "score": 87, "lat": 34.1478, "lng": -118.1445},
                {"name": "Costa Mesa, CA", "score": 84, "lat": 33.6411, "lng": -117.9187}
            ]
        })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
