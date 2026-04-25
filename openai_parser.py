from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv("secrets.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "business_type": {"type": ["string", "null"]},
        "region": {"type": ["string", "null"]},
        "employee_count": {"type": ["integer", "null"]},
        "valuation_usd": {"type": ["number", "null"]},
        "monthly_rent_budget_usd": {"type": ["number", "null"]},
        "priorities_known": {"type": "boolean"},
        "user_priorities": {
            "type": "array",
            "items": {"type": "string"}        },
        "suggested_priorities": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3
        },
        "follow_up_question": {"type": "string"}
    },
    "required": [
        "business_type",
        "region",
        "employee_count",
        "valuation_usd",
        "monthly_rent_budget_usd",
        "priorities_known",
        "user_priorities",
        "suggested_priorities",
        "follow_up_question"
    ]
}

def parse_business_message(message: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Pinpoint AI's intake parser. "
                        "Extract the user's business details and return ONLY a JSON object with these exact fields - never omit any: "
                        "business_type (string or null), "
                        "region (string or null, expand shorthand e.g. SoCal -> Southern California), "
                        "employee_count (integer or null), "
                        "valuation_usd (number or null), "
                        "monthly_rent_budget_usd (number or null), "
                        "priorities_known (boolean, true only if user explicitly states priorities), "
                        "user_priorities (array of strings, empty [] if not stated), "
                        "suggested_priorities (array of exactly 3 strings, always provide 3 relevant suggestions), "
                        "follow_up_question (string, ask what matters most if priorities unknown). "
                        "No extra fields. No markdown. No explanation. Just the JSON object."
                    )
                },
                {"role": "user", "content": message}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("Parser error:", e)

        return {
            "business_type": None,
            "region": None,
            "employee_count": None,
            "valuation_usd": None,
            "monthly_rent_budget_usd": None,
            "priorities_known": False,
            "user_priorities": [],
            "suggested_priorities": [
                "Affordable rent",
                "High foot traffic",
                "Target customer proximity"
            ],
            "follow_up_question": "I couldn't fully parse that. What matters most for your business?"
        }