from typing import List, Dict, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated


class InterviewState(TypedDict):
    first_question: str
    topic: str
    content: List[str]
    cv_content: str
    user_response: str
    robot_response: str
    questions: List[str]
    answers: List[str]
    feedback: List[Dict]
    current_question: Optional[str]
    current_answer: Optional[str]
    step: int
    max_steps: int
    final_evaluation: Optional[Dict]
    messages: List[Dict]
    messages2: Annotated[list[BaseMessage], add_messages]
    question_type: str
    needs_retrieval: bool
    retrieved_context: Optional[str]
    similarity_score: Optional[float]
    user_id: str
    tavily_snippets: List[str]
    waiting_for_user: bool
    feedback_text: str
    finished: bool
