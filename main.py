from flask import Flask, request, jsonify
from flask_cors import CORS
from openai_parser import parse_business_message

app = Flask(__name__)
CORS(app)

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
            {"name": "Irvine", "score": 91},
            {"name": "Pasadena", "score": 87},
            {"name": "Costa Mesa", "score": 84}
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)
