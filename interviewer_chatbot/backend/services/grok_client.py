import os
from groq import Groq
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GrokClient:
    def __init__(self, client):
        self.client = client

    def generate_content(self, text: str, timeout: int = 15) -> str:
        messages = [{"role": "user", "content": text}]
        res = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",  # or any Groq model
            messages=messages,
            timeout=timeout,  # Add timeout parameter
            temperature=0.7,  # Consistent temperature
            max_tokens=500,  # Limit response length for faster responses
        )
        logger.info(f"Generated content for prompt (length {len(text)} chars)")
        return res.choices[0].message.content.strip()


# Usage
ai_client = Groq(api_key=settings.groq_api_key)
groq_client = GrokClient(ai_client)
