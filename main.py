from flask import Flask, request, jsonify
from flask_cors import CORS
from openai_parser import parse_business_message
from pinpoint_mcp.engine.ranker import rank_locations
from pinpoint_mcp.data.business_map import normalize_business_type

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
    data = request.get_json() or {}

    business_type = data.get("business_type") or "cafe"
    region        = data.get("region") or None
    priorities    = data.get("user_priorities") or []
    top_n         = int(data.get("top_n", 10))

    result = rank_locations(
        business_type_raw=business_type,
        region_str=region,
        priorities=priorities,
        top_n=top_n,
    )

    # Shape the output for the frontend's map pin format
    recommendations = [
        {
            "name":        r["geoname"],
            "county":      r["county"],
            "state":       r["state"],
            "geofips":     r["geofips"],
            "score":       r["score"],
            "breakdown":   r["breakdown"],
            "key_metrics": r["key_metrics"],
            "explanation": r["explanation"],
            "lat":         r.get("lat"),
            "lng":         r.get("lng"),
            "rank":        r["rank"],
        }
        for r in result["results"]
    ]

    return jsonify({
        "business_type":          result["business_type"],
        "region_label":           result["region_label"],
        "total_counties_scored":  result.get("total_counties_scored", 0),
        "recommendations":        recommendations,
    })


if __name__ == "__main__":
    app.run(debug=True)
