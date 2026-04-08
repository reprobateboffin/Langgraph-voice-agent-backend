import logging
import textwrap
from typing import Any, Dict
import json
from services.gemini_client import gemini_client
from utils.prompt_template import safe_prompt
from google.genai.errors import ServerError  # adjust import if needed
import logging
from services.mistral_client import mistral_client
from services.grok_client import groq_client


logger = logging.getLogger(__name__)


def safe_text(text: str, max_len: int = 2000) -> str:
    """
    Sanitize and truncate user-provided or large text to avoid context overflow
    and unwanted characters.
    """
    if not text:
        return ""
    sanitized = str(text).replace("\r", "").replace("\t", "    ")
    return sanitized[:max_len]


# def _safe_generate(prompt: str, fallback: str, gemini_client=gemini_client) -> str:
#     try:
#         return mistral_client.generate_content(prompt) or fallback
#     except Exception as e:
#         logger.error("Generation failed: %s", e)
#         return fallback


# Corrected _safe_generate function
# def _safe_generate(prompt: str, fallback: str, gemini_client=None) -> str:
#     try:
#         # Mistral client returns string directly
#         response = groq_client.generate_content(prompt)
#         # Ensure we return a non-empty string
#         if response and response.strip():
#             return response.strip()
#         return fallback
#     except Exception as e:
#         logger.error("Generation failed: %s", e)
#         return fallback


def _safe_generate(
    prompt: str,
    fallback: str = "",
    gemini_client=gemini_client,
    mistral_client=mistral_client,
    groq_client=groq_client,
) -> str:
    if gemini_client:
        try:
            response = gemini_client.generate_content(prompt)

            if response:
                return response
        except Exception as e:
            logger.warning("Gemini failed: %s", e)

    if groq_client:
        try:
            response = groq_client.generate_content(prompt)
            if response:
                return response
        except Exception as e:
            logger.warning("Groq failed: %s", e)
    # 2️⃣ Mistral

    # 1️⃣ Gemini
    if mistral_client:
        try:
            response = mistral_client.generate_content(prompt)

            if response:
                return response
        except Exception as e:
            logger.warning("Mistral failed, falling back: %s", e)

    # 3️⃣ Groq

    # 4️⃣ Fallback
    logger.warning("All providers failed")
    return fallback


def safe_parse_json(response: Any) -> Dict[str, Any]:
    fallback = {"rating": 6, "feedback": "Good effort. Could elaborate more."}

    if not response:
        return fallback

    if isinstance(response, dict):
        return response

    text = None
    if hasattr(response, "text"):
        text = response.text
    elif hasattr(response, "content"):
        text = response.content
    elif isinstance(response, str):
        text = response

    if not text:
        return fallback

    try:
        return json.loads(text)
    except Exception:
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return {**fallback, "raw_text": text[:500]}


def parse_json_answer_feedback(response: Any):
    fallback = [
        {
            "answer_index": None,
            "answer_text": None,
            "rating": 6,
            "feedback": "Could not parse evaluation.",
        }
    ]

    if not response:
        return fallback

    # Gemini / OpenAI compatibility
    if hasattr(response, "text"):
        text = response.text
    elif hasattr(response, "content"):
        text = response.content
    elif isinstance(response, str):
        text = response
    else:
        return fallback

    # Try strict JSON
    try:
        raw_data = json.loads(text)
    except Exception:
        # Try extracting JSON block
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            raw_data = json.loads(text[start:end])
        except Exception:
            return [{**fallback[0], "raw_text": text[:500]}]

    if not isinstance(raw_data, list):
        raw_data = [raw_data]

    processed_answers = []

    for entry in raw_data:
        if not isinstance(entry, dict):
            continue

        clean_doc = {
            "answer_index": entry.get("answer_index"),
            "answer_text": entry.get("answer_text") or entry.get("answer"),
            "rating": entry.get("rating", 6),
            "feedback": entry.get("feedback", "No feedback provided"),
        }
        processed_answers.append(clean_doc)

    return processed_answers or fallback


def parse_json_final_feedback(response: Any):
    fallback = [
        {
            "overall_quality": 0,
            "strengths": [],
            "areas_for_improvement": [],
            "recommendation": "",
            "final_feedback": "",
        }
    ]

    if not response:
        return fallback

    # ✅ If already parsed object (VERY IMPORTANT)
    if isinstance(response, list):
        raw_data = response
    elif isinstance(response, dict):
        raw_data = [response]
    else:
        # Gemini / OpenAI wrapper handling
        if hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "content"):
            text = response.content
        elif isinstance(response, str):
            text = response
        else:
            return fallback

        # ✅ Remove markdown fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]

        try:
            raw_data = json.loads(text)
        except Exception:
            try:
                start = text.find("[")
                end = text.rfind("]") + 1
                raw_data = json.loads(text[start:end])
            except Exception:
                return [{**fallback[0], "raw_text": text[:500]}]

    if not isinstance(raw_data, list):
        raw_data = [raw_data]

    final_feedback = []

    for entry in raw_data:
        if not isinstance(entry, dict):
            continue

        clean_doc = {
            "overall_quality": int(entry.get("overall_quality", 0)),
            "strengths": entry.get("strengths", []),
            "areas_for_improvement": entry.get("areas_for_improvement", []),
            "recommendation": entry.get("recommendation", ""),
            "final_feedback": entry.get("final_feedback", ""),
        }
        final_feedback.append(clean_doc)

    return final_feedback or fallback


def build_prompt(role_desc: str, content: str, body: str) -> str:
    """
    Standard prompt builder to avoid duplication.
    Applies safe_text to content and strips extra whitespace.
    """
    return f"""
        You are {role_desc}.
        Using the following reference content:
        {safe_text(content)}

        {body}
    """.strip()
