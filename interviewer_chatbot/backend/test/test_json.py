import json

# Corrected JSON format
prompt_output = """
[
    {"answer_index": "1", "answer": "the actual 1st answer", "rating": 8, "feedback": "Great insight!"},
    {"answer_index": "2", "answer": "the actual 2nd answer", "rating": 4, "feedback": "Too brief."},
    {"answer_index": "3", "answer": "the actual 3rd answer", "rating": 10, "feedback": "Perfect."}
]
"""

try:
    raw_data = json.loads(prompt_output)
    processed_docs = []

    for entry in raw_data:
        # Directly grab the values since the keys are now consistent
        clean_doc = {
            "answer_index": entry.get("answer_index"),
            "answer_text": entry.get("answer"),
            "rating": entry.get("rating"),
            "feedback": entry.get("feedback"),
        }
        processed_docs.append(clean_doc)

    # Printing the results
    for item in processed_docs:
        number = item.get("answer_index")
        text = item.get("answer_text")
        print(f"answer # {number} is : {text}")

except json.JSONDecodeError:
    print("The LLM returned invalid JSON. You might need to clean the string first.")
