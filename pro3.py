import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF-TOkEN")

API_URL = "https://router.huggingface.co/models/google/flan-t5-large"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def generate_email(product_name, target_audience):
    prompt = f"""
    Write a professional marketing email for launching a product called {product_name}.
    Target audience: {target_audience}.
    Keep it persuasive and under 200 words.
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 300
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    result = response.json()

    if isinstance(result, list):
        return result[0]["generated_text"]
    else:
        return "Error: " + str(result)

if __name__ == "__main__":
    product = input("Enter product name: ")
    audience = input("Enter target audience: ")

    email = generate_email(product, audience)

    print("\nGenerated Marketing Email:\n")
    print(email)
