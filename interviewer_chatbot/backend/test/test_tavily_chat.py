# To install: pip install tavily-python

from tavily import TavilyClient
from backend.config.settings import settings

TAVILY_CHAT_KEY = settings.tavily_chat_key
client = TavilyClient(TAVILY_CHAT_KEY)
response = client.search(
    query="Hello, who is the current US president", search_depth="advanced"
)
print(response)
