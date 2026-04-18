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

    # Placeholder recommendation logic for now
    # Later you will replace this with your actual model / scoring logic
    return jsonify({
        "received_profile": data,
        "recommendations": [
            {"name": "Irvine, CA",     "score": 91, "lat": 33.6846, "lng": -117.8265},
            {"name": "Pasadena, CA",   "score": 87, "lat": 34.1478, "lng": -118.1445},
            {"name": "Costa Mesa, CA", "score": 84, "lat": 33.6411, "lng": -117.9187}
        ]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
