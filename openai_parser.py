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
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are Pinpoint AI's intake parser. "
                        "Extract the user's business details into structured JSON. "
                        "If the user does not clearly state priorities, set priorities_known to false "
                        "and suggest exactly 3 likely priorities. "
                        "Normalize obvious shorthand when reasonable, such as SoCal to Southern California "
                        "and 700k to 700000. "
                        "Do not recommend locations yet."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "business_intake",
                    "strict": True,
                    "schema": SCHEMA
                }
            }
        )

        return json.loads(response.output_text)

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