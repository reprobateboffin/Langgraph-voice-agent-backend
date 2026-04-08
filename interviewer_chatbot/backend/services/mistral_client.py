from mistralai.client import Mistral
import os
from config.settings import settings
from utils.logger import setup_logger

MISTRAL_API_KEY = settings.mistral_api_key

logger = setup_logger(__name__)


class MistralClient:
    def __init__(self):
        self.client = Mistral(api_key=MISTRAL_API_KEY)

    def generate_content(self, prompt: str) -> str:
        res = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            response_format={"type": "text"},
            temperature=0.7,  # Add temperature for faster responses
            max_tokens=500,  # Limit response length
        )

        content = res.choices[0].message.content
        logger.info(f"Generated content for prompt (length {len(prompt)} chars)")
        # Hard guarantee: always return string
        if not content:
            return ""

        return str(content).strip()


mistral_client = MistralClient()
