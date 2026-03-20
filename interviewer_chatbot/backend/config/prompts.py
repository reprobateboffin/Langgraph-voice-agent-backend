import textwrap
from typing import Literal
from utils.generation import safe_text, build_prompt


def get_setup_prompt(
    topic: str, question_type: str, context: str, tool_used: str
) -> str:
    """
    Returns the initial interview question prompt.
    Uses retrieved context if tool_used == 'RAG'.
    """
    if tool_used == "RAG" and context:
        body = f"""
        You are conducting a technical interview for a {topic} position.

        Candidate Background:
        {context}

        Generate the first interview question considering the candidate's experience.
        """
    else:
        body = f"""
        You are conducting a technical interview for a {topic} position.

        Generate an opening question that assesses basic knowledge and experience.
        """
    return build_prompt("an expert interviewer", context, body)


def get_voice_setup_prompt(
    topic: str,
    question_type: str,
    cv_text: str,
):
    body = f""" 
          You are conducting a technical interview for a {topic} position.
        Candidate Background:
        {cv_text}
        Generate the first interview question considering the candidate's experience.with question type {question_type}
"""
    return build_prompt("an expert interviewer", cv_text, body)


def get_question_generation_prompt(
    content_text: str, topic: str, step: int, tool_used: str, context: str
) -> str:
    """
    Generates follow-up question prompts.
    Uses RAG or Tavily context depending on tool_used.
    """
    if tool_used == "RAG" and context:
        body = f"""
        Generate the next interview question for a {topic} position.

        Candidate Background:
        {context}

        Conversation so far:
        {content_text}

        Question number: {step + 1}
        """
    else:
        body = f"""
        Generate the next interview question for a {topic} position.

        Conversation so far:
        {content_text}

        Question number: {step + 1}
        """
    return build_prompt("an expert interviewer", content_text, body)


def get_evaluation_prompt(
    kind: Literal["question", "answer"],
    full_messages: str,
    full_content: str,
    transcript: str,
    last_question: str = "",
    last_answer: str = "",
) -> str:
    """
    Generic evaluation prompt for questions or answers.
    Returns JSON only.
    """
    kind_desc = "question" if kind == "question" else "candidate answer"
    body = f"""
        Evaluate the following {kind_desc} for clarity, relevance, depth, and alignment.
        Full Messages: {safe_text(full_messages)}
        Context: {safe_text(full_content)}
        Transcript: {safe_text(transcript)}
        Current Question: {safe_text(last_question)}
        Current Answer: {safe_text(last_answer)}
        Provide a rating (1-10) and detailed feedback.
        Return JSON only.
    """

    if kind == "answer":
        body += textwrap.dedent(
            """
        Return in JSON format:
        {
            "rating": 0,
            "feedback": "..."
        }
        """
        )
    return build_prompt("an expert interviewer", "", body)


def get_answer_evaluation_prompt(
    kind: Literal["question", "answer"],
    answers: str,
    questions: str,
    full_content: str,
) -> str:
    """
    Generic evaluation prompt for questions or answers.
    Returns JSON only.
    """
    kind_desc = "question" if kind == "question" else "candidate answer"
    body = f"""
        Evaluate the following {kind_desc}s for clarity, relevance, depth, and alignment.
        answers : {safe_text(answers)}
        questions: {safe_text(questions)}
        Context: {safe_text(full_content)}
        Provide a rating (1-10) and detailed feedback for each individual {kind_desc}. return a json with distinct answer feedback for each answer, it should be the same json format as given below
        Return JSON only, in the form  with attributes as below written. DO NOT RETURN ANY OTHER TEXT EXCEPT THIS JSON
    """

    if kind == "answer":
        body += textwrap.dedent(
            """
        Return in JSON format:
       [
            {
            "answer_index": 1,
            "answer_text": "the actual 1st answer" ,
            "rating": between 1-10,
            "feedback": "..."
             },
             {
            "answer_index": 2,
            "answer_text": "the actual 2nd answer",
            "rating": between 1-10,
            "feedback": "..."
             },
                {
            "answer_index": n,
            "answer_text": "the actual nth  answer", 
            "rating": between 1-10,
            "feedback": "..."
             },
            If multiple answers exist, include multiple objects.

            Do not include markdown.
            Do not include explanations.
            Do not include extra keys.
        ]
        """
        )
    return build_prompt("an expert interview evaluator", "", body)


def get_final_evaluation_prompt(transcript: str) -> str:
    """
    Produces final evaluation JSON for the entire interview.(answers only, do not evaluate the questions)
    """
    body = f"""    Based on the transcript, produce a JSON summary evaluation, JSON only, no other text with it(no markups etc) (answers only, do not evaluate the questions):
        {safe_text(transcript)} """
    body += textwrap.dedent(
        """
        Return JSON only Schema. If you want to give multiple strentgs or areas for improvement, give them as separate elements of the list:
        [{
            "overall_quality": 0,
            "strengths": ["..."],
            "areas_for_improvement": ["..."],
            "recommendation": "...",
            "final_feedback": "..."
        }]
  """
    )

    return build_prompt("an expert interviewer", "", body)
