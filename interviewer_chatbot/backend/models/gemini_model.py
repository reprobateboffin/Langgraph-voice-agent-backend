from google import genai
from config.settings import settings

# genai.configure(api_key=settings.gemini_api_key)
client = genai.Client(api_key=settings.gemini_api_key)

# GeminiModel = genai.GenerativeModel(settings.gemini_model)
from google import genai
from config.settings import settings

# Create ONE client at import time (matches your current design)
_client = genai.Client(api_key=settings.gemini_api_key)


class _Response:
    """Small adapter so response.text always exists"""

    def __init__(self, text: str):
        self.text = text


class GeminiModel:
    @staticmethod
    def generate_content(prompt: str):
        result = _client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        return _Response(result.text or "")
