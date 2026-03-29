from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings

embeddings = None


def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = GoogleGenerativeAIEmbeddings(model=settings.gemini_embedding_model)
    return embeddings
