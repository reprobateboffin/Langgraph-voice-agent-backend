import os
from groq import Groq
from config.settings import settings

# Set the environment variable GROQ_API_KEY before running the script
client = Groq(
    api_key=settings.groq_api_key,
)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Explain how to test the Groq API in one sentence."}
    ],
    model="llama-3.1-8b-instant",
)

print(chat_completion.choices[0].message.content)
