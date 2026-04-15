import json
import os
from typing import Any, Dict, Mapping, List
from utils.logger import setup_logger
from utils.generation import (
    _safe_generate,
    parse_json_answer_feedback,
    parse_json_final_feedback,
)
from services.tavily_client import tavily_service
from models.embedding_model import get_embeddings
from models.final_evaluation import FinalEvaluation
from config.prompts import (
    get_answer_evaluation_prompt,
    get_setup_prompt,
    get_question_generation_prompt,
    get_evaluation_prompt,
    get_final_evaluation_prompt,
)
from utils.sanitizer import sanitize_state
from utils.generation import safe_parse_json
import textwrap
from services.vectorstore_service import load_vectorstore

logger = setup_logger(__name__)
embeddings = get_embeddings()


def decide_retrieval(query: str, user_id: str = "default_user") -> (bool, float):
    """
    Decide if retrieval is needed based on Chroma Cloud distances.

    Returns:
        Tuple[needs_retrieval: bool, min_distance: float]
    """
    try:
        collection = load_vectorstore(user_id)
        if not collection:
            logger.info("No collection for user '%s', forcing retrieval.", user_id)
            return True, 1.0

        query_emb = embeddings.embed_query(query)
        results = collection.query(query_embeddings=[query_emb], n_results=3)

        distances = results.get("distances", [[]])[0]

        if not distances:
            logger.info("No docs found in vectorstore, forcing retrieval.")
            return True, 1.0

        min_distance = float(min(distances))

        DISTANCE_THRESHOLD = 0.78
        needs_retrieval = min_distance > DISTANCE_THRESHOLD

        logger.info(
            "Decide retrieval -> min_distance: %.4f, needs_retrieval: %s",
            min_distance,
            needs_retrieval,
        )
        return needs_retrieval, min_distance

    except Exception as e:
        logger.error("Retrieval decision error: %s", e, exc_info=True)
        return True, 1.0


def setup_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    state = dict(state)
    if state.get("step", 0) > 0:
        return sanitize_state(state)

    logger.info("✅ Running setup_node")
    topic = state.get("topic", "").strip()
    question_type = state.get("question_type", "broad_followup").strip()
    user_id = state.get("user_id", "default_user")
    retrieved_context = ""

    try:
        collection = load_vectorstore(user_id)
        if collection:
            query_emb = embeddings.embed_query(topic)
            results = collection.query(query_embeddings=[query_emb], n_results=3)
            docs = results.get("documents", [[]])[0]
            retrieved_context = "\n\n".join(docs)
            logger.info("Retrieved setup context for topic: %s", topic)
    except Exception as e:
        logger.error("Setup retrieval failed for user '%s': %s", user_id, e)

    prompt = get_setup_prompt(topic, question_type, retrieved_context, "RAG")
    first_question = _safe_generate(
        prompt, "Tell me about your experience with this technology."
    )

    new_state = {
        **state,
        "topic": topic,
        "question_type": question_type,
        "content": [retrieved_context or "No content"],
        "messages": [
            {"role": "user", "content": f"Interview topic: {topic}"},
            {"role": "assistant", "content": first_question},
        ],
        "step": 1,
        "questions": [],
        "answers": [],
        "feedback": [],
        "first_question": first_question,
        "current_question": first_question,
        "robot_response": first_question,
        "max_steps": state.get("max_steps", 3),
        "waiting_for_user": True,
        "needs_retrieval": False,
        "retrieved_context": retrieved_context,
        "similarity_score": 0.0,
        "tavily_snippets": [],
    }

    return sanitize_state(new_state)


def get_answer_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Update state with the user's answer to the current question.

    Args:
        state (Mapping[str, Any]): Current state.

    Returns:
        Dict[str, Any]: Updated state including new messages and content.
    """
    state = dict(state)
    logger.info("✅ Running get_answer_node")

    current_q = state.get("current_question")
    if not current_q:
        raise ValueError("No current_question found in state.")

    answer = state.get("user_response", "")
    new_messages = list(state.get("messages", [])) + [
        {"role": "interviewer", "content": current_q},
        {"role": "candidate", "content": answer},
    ]
    content_list = list(state.get("content", [])) + [f"Q: {current_q}\nA: {answer}"]

    new_state = {
        **state,
        "current_answer": answer,
        "messages": new_messages,
        "questions": state.get("questions", []) + [current_q],
        "robot_response": current_q,
        "answers": state.get("answers", []) + [answer],
        "content": content_list,
    }
    return sanitize_state(new_state)


def retrieval_decision_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    logger.info("✅ Running retrieval_decision_node")
    state = dict(state)
    current_answer = state.get("current_answer", "")
    user_id = state.get("user_id", "default_user")

    try:
        needs_retrieval, similarity_score = decide_retrieval(current_answer, user_id)
    except Exception as e:
        logger.error("Retrieval decision node failed: %s", e)
        needs_retrieval, similarity_score = False, 1.0

    new_state = {
        **state,
        "needs_retrieval": needs_retrieval,
        "similarity_score": similarity_score,
    }
    return sanitize_state(new_state)


def retrieval_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Retrieve relevant documents from Chroma Cloud collection for the user.
    """
    logger.info("✅ Running retrieval_node")
    state = dict(state)

    if not state.get("needs_retrieval", False):
        return sanitize_state({**state, "retrieved_context": None})

    user_id = state.get("user_id", "default_user")
    query = state.get("current_answer", state.get("topic", ""))

    try:
        collection = load_vectorstore(user_id)
        if collection:
            query_emb = embeddings.embed_query(query)
            results = collection.query(query_embeddings=[query_emb], n_results=3)

            docs = results.get("documents", [[]])[0]
            retrieved_context = "\n\n".join(docs) if docs else None

            logger.info(
                "Retrieved %d docs for query '%s' (user: %s)",
                len(docs),
                query,
                user_id,
            )

            return sanitize_state({**state, "retrieved_context": retrieved_context})

    except Exception as e:
        logger.error("Retrieval failed for user '%s': %s", user_id, e, exc_info=True)

    return sanitize_state({**state, "retrieved_context": None})


def tavily_search_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Perform Tavily search to enrich context with relevant snippets.

    Args:
        state (Mapping[str, Any]): Current state.

    Returns:
        Dict[str, Any]: Updated state including Tavily snippets and enriched context.
    """
    logger.info("✅ Running tavily_search_node")

    state = dict(state)
    query = state.get("current_answer", state.get("topic", ""))
    if not query:
        return sanitize_state(state)

    try:
        snippets = tavily_service.search(query, top_k=5)
        if snippets:
            enriched_context = (
                (state.get("retrieved_context") or "") + "\n\n" + "\n\n".join(snippets)
            )
            return sanitize_state(
                {
                    **state,
                    "retrieved_context": enriched_context,
                    "tavily_snippets": snippets,
                }
            )
        return sanitize_state(state)
    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return sanitize_state(state)


def generate_question_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Generate the next interview question based on current state and context.

    Args:
        state (Mapping[str, Any]): Current state.

    Returns:
        Dict[str, Any]: Updated state with new question.
    """
    logger.info("✅ Running generate_question_node")

    state = dict(state)
    step = state.get("step", 0)
    max_questions = state.get("max_steps", 3)
    # if step >= max_questions:
    if len(state["answers"]) >= max_questions:
        return sanitize_state(state)

    topic = state.get("topic", "")
    content_list = state.get("content", [])
    context_text = []

    if state.get("needs_retrieval") and state.get("retrieved_context"):
        context_text.append(state["retrieved_context"])
        context_sources = ["RAG"]
    elif state.get("tavily_snippets"):
        context_text.extend(state["tavily_snippets"])
        context_sources = ["Tavily"]
    else:
        context_sources = ["None"]

    full_content = "\n".join(list(content_list[-4:]) + list(context_text))
    context_str = "\n".join(context_text)

    prompt = get_question_generation_prompt(
        content_text=full_content,
        topic=topic,
        step=step,
        tool_used=context_sources[0],
        context=context_str,
    )
    try:
        question = _safe_generate(
            prompt, f"Tell me more about your experience with {topic}."
        )
    except Exception as e:
        logger.error("Question generation failed: %s", e)
        question = f"Please elaborate more on {topic}."

    messages = list(state.get("messages", [])) + [
        {"role": "assistant", "content": question}
    ]

    new_state = {
        **state,
        "current_question": question,
        "messages": messages,
        "waiting_for_user": True,
        "step": step + 1,
    }

    return sanitize_state(new_state)


def evaluate_question_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    logger.info("✅ Running final evaluate_question_node")

    state = dict(state)
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    logger.info(f"QUESTIONS: {len(questions)}")
    logger.info(f"ANSWERS: {len(answers)}")
    if not answers or not questions:
        return sanitize_state(state)

    questions_text = "\n".join(f"{i+1}. Q: {q}" for i, q in enumerate(questions))
    answers_text = "\n".join(f"{i+1}. A: {a}" for i, a in enumerate(answers))

    full_content = "\n".join(state.get("content", []))

    parsed_eval = []
    prompt = get_answer_evaluation_prompt(
        kind="answer",
        answers=answers_text,
        questions=questions_text,
        full_content=full_content,
    )
    try:

        raw = _safe_generate(prompt)

        parsed_eval = parse_json_answer_feedback(raw)

    except Exception as e:
        logger.error("Evaluation failed: %s", e)

    feedback_text = "\n\n".join(
        f"Answer {item['answer_index']} (Rating {item['rating']}):\n{item['feedback']}"
        for item in parsed_eval
    )

    new_state = {
        **state,
        "feedback": parsed_eval,
        "feedback_text": feedback_text,
    }

    return sanitize_state(new_state)


# from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime, timezone

uri = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(uri)
db = client["interviews_db"]  # auto-created
collection = db["interviews"]  # auto-created


def final_evaluation_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    state = vars(state) if not isinstance(state, dict) else state

    questions, answers = (
        state.get("questions", []),
        state.get("answers", []),
    )
    if not questions or not answers:
        logger.warning("No data for final evaluation.")
        return state
    logger.info("✅ Running final_evaluation_node")

    transcript = ""
    transcript = "\n".join(
        f"{i+1}. Q: {q}\n{i+1}. A: {a}"
        for i, (q, a) in enumerate(zip(questions, answers))
    )
    final_prompt = get_final_evaluation_prompt(transcript)
    try:
        # raw_final = gemini_client.generate_content(final_prompt)
        raw_final = _safe_generate(final_prompt)
        print(f"what gemini returned: {raw_final}")
        parsed_final = parse_json_final_feedback(raw_final)
        print(f"AFTER PARSING: {parsed_final}", type(parsed_final))
    except Exception as e:
        logger.error("Final eval parse failed: %s", e)
        parsed_final = {}

    try:
        answer_eval = state.get("feedback", [])
        is_company = state.get("isCompany", False)

        interview_id = state.get("interview_id", "nnnooo")
        room_name = state.get("room_name", "NoNameFound")
        if is_company:
            company_name = state.get("company_name", "NoneFound")
        else:
            company_name = "personal_use"
        result = collection.insert_one(
            {
                "final_eval": parsed_final,
                "answer_eval": answer_eval,
                "questions": questions,
                "answers": answers,
                "user_id": state["user_id"],
                "interview_id": interview_id,
                "isCompany": is_company,
                "createdAt": datetime.now(timezone.utc),
                "company_name": company_name,
                "room_name": room_name,
            }
        )

        print("Inserted ID:", result.inserted_id)

        # Quick read
        print("\nAll interviews:")
        for note in collection.find().limit(5):
            print(note)
    except Exception as e:
        print(e)

    return {**state, "final_evaluation": parsed_final, "finished": True}


def display_results_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Display interview results using shared rendering utility.

    Args:
        state (Mapping[str, Any]): Current state.

    Returns:
        Dict[str, Any]: Sanitized state.
    """
    logger.info("✅ Running display_results_node")

    try:
        from utils.interview_results import render_interview_results

        render_interview_results(state)
    except Exception as e:
        logger.error("Display results failed: %s", e)
    return sanitize_state(state)
