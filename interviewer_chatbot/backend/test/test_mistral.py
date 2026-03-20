from mistralai.client import Mistral
import os
from config.settings import settings

MISTRAL_API_KEY = settings.mistral_api_key
with Mistral(
    api_key=MISTRAL_API_KEY,
) as mistral:

    res = mistral.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": "Who is the best Pakistani painter? Answer in one short sentence.",
            },
        ],
        stream=False,
        response_format={
            "type": "text",
        },
    )

    # Handle response
    print(res)
