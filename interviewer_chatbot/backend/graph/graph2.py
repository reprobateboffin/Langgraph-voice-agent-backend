import json
import os
import asyncio
import logging
from typing import Dict, List, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from services.vectorstore_service import load_vectorstore

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from livekit.agents import AgentServer
from contextlib import asynccontextmanager

from config.prompts import get_setup_prompt
from utils.generation import _safe_generate
from services.gemini_client import gemini_client  # assuming this exists
from contextlib import asynccontextmanager
from models.embedding_model import get_embeddings

logger = logging.getLogger("voice-agent")
load_dotenv()
server = AgentServer()

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging

print("AGENT MODULE IMPORTED", flush=True)

logger = logging.getLogger(__name__)


class State(TypedDict):
    user_response: str
    response: str
    step: int
    first_question: str
    topic: str
    user_id: str


# class State(TypedDict):
#     first_question: str
#     topic: str
#     content: List[str]
#     cv_content: str
#     user_response: str
#     robot_response: str
#     questions: List[str]
#     answers: List[str]
#     feedback: List[Dict]
#     current_question: Optional[str]
#     current_answer: Optional[str]
#     step: int
#     max_steps: int
#     final_evaluation: Optional[Dict]
#     messages: List[Dict]
#     question_type: str
#     needs_retrieval: bool
#     retrieved_context: Optional[str]
#     similarity_score: Optional[float]
#     user_id: str
#     tavily_snippets: List[str]
#     waiting_for_user: bool
#     feedback_text: str


# -----------------------
# Start node (runs once)
# -----------------------
def setup_node(state: State):
    if state["step"] == 0:
        logger.info("✅ Running setup_node")
        topic = state.get("topic", "gen ai engineer")
        question_type = state.get("question_type", "broad follow up").strip()
        user_id = state.get("user_id", "default_user")
        collection = load_vectorstore(user_id)
        retrieved_context = ""

        try:
            if collection:
                embeddings = get_embeddings()
                query_emb = embeddings.embed_query(topic)
                results = collection.query(query_embeddings=[query_emb], n_results=3)
                docs = results.get("documents", [[]])[0]
                retrieved_context = "\n\n".join(docs)
                logger.info("Retrieved setup context for topic: %s", topic)
        except Exception as e:
            logger.error("Setup retrieval failed for user '%s': %s", user_id, e)

        cv_content = state.get("cv_content", "Software engineer")
        prompt = get_setup_prompt(topic, question_type, retrieved_context, "RAG")

        first_question = _safe_generate(
            prompt, "Tell me about your experience with this technology."
        )
        return {
            **state,
            "first_question": first_question,
            "step": 1,
        }
    return state


# -----------------------
# Echo / LLM node
# -----------------------
def echo_node(state: State):
    try:
        response = gemini_client.generate_content(state["user_response"])
        logger.info(f"running echo node")
    except Exception as e:
        logger.exception("Error in echo_node")
        response = f"An internal error occurred: {e}"

    return {
        **state,
        "response": response,
        "step": state["step"] + 1,  # 🔑 advance conversation
        "user_response": "",  # 🔑 consume input
    }


# -----------------------
# Router
# -----------------------
def route_after_start(state: State):
    user_response = state.get("user_response", "")

    if isinstance(user_response, list):
        texts = []
        for part in user_response:
            if isinstance(part, dict):
                texts.append(part.get("text", ""))
            elif isinstance(part, str):
                texts.append(part)  # Handle plain strings directly
        user_response = " ".join(texts).strip()

    if state["step"] == 0:
        return "start"
    if isinstance(user_response, str) and user_response.strip():
        return "echo"
    logger.info(
        f"Routing: step={state['step']}, raw_user_response={state.get('user_response')}, cleaned_user_response={user_response}"
    )
    return END


def create_workflow():
    graph = StateGraph(State)

    graph.add_node("start", setup_node)
    graph.add_node("echo", echo_node)

    graph.set_entry_point("start")

    def router(state: State) -> str:
        ur = state.get("user_response", "")
        if isinstance(ur, list):
            texts = [p if isinstance(p, str) else p.get("text", "") for p in ur]
            cleaned = " ".join(texts).strip()
        else:
            cleaned = str(ur).strip()

        if state["step"] == 0:
            return "start"
        if cleaned:
            return "echo"
        return END

    # graph.add_conditional_edges(
    #     "start", router, {"start": "start", "echo": "echo", END: END}
    # )
    graph.add_edge("start", END)

    graph.add_conditional_edges("echo", router, {"echo": "echo", END: END})

    return graph.compile(checkpointer=MemorySaver())


compiled_graph2 = create_workflow()


async def response_stream(text: str | None):
    if text:
        yield text
    else:
        return
