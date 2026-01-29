# from langgraph.graph import StateGraph, END
# from graph.state import InterviewState
# from graph.nodes import (
#     setup_node,
#     get_answer_node,
#     evaluate_question_node,
#     generate_question_node,
#     final_evaluation_node,
#     display_results_node,
#     retrieval_decision_node,
#     retrieval_node,
#     tavily_search_node,
# )
# from utils.logger import setup_logger
# from langgraph.checkpoint.memory import MemorySaver
# import psycopg
# from langgraph.checkpoint.postgres import PostgresSaver
# from config.settings import settings

# logger = setup_logger(__name__)

# SETUP_NODE = "setup"
# GET_ANSWER_NODE = "get_answer"
# RETRIEVAL_DECISION_NODE = "retrieval_decision"
# RETRIEVAL_NODE = "retrieval"
# TAVILY_SEARCH_NODE = "tavily_search"
# GENERATE_QUESTION_NODE = "generate_question"
# EVALUATE_QUESTION_NODE = "evaluate_question"
# FINAL_EVALUATION_NODE = "final_evaluation"
# DISPLAY_RESULTS_NODE = "display_results"


# def should_retrieve(state: InterviewState) -> str:
#     """
#     Decide whether to use internal RAG retrieval or Tavily search based on the state.

#     Args:
#         state (InterviewState): Current interview state containing flags and context.

#     Returns:
#         str: The name of the next node to execute, either RETRIEVAL_NODE or TAVILY_SEARCH_NODE.
#     """
#     return RETRIEVAL_NODE if state["needs_retrieval"] else TAVILY_SEARCH_NODE


# def should_continue(state: InterviewState) -> str:
#     """
#     Determine whether the interview should continue, end, or move to evaluation.

#     Args:
#         state (InterviewState): Current state of the interview.

#     Returns:
#         str: The name of the next node.
#             - END if waiting for user input.
#             - EVALUATE_QUESTION_NODE if max steps reached.
#             - GET_ANSWER_NODE otherwise.
#     """
#     if state.get("waiting_for_user", False):
#         return END
#     if state.get("step", 0) >= state.get("max_steps", 10):
#         return EVALUATE_QUESTION_NODE
#     return GET_ANSWER_NODE


# def should_generate_question(state: InterviewState) -> str:
#     """
#     Decide whether to generate the next question or perform the final evaluation.

#     Args:
#         state (InterviewState): Current interview state.

#     Returns:
#         str: GENERATE_QUESTION_NODE if interview continues,
#              FINAL_EVALUATION_NODE if interview ends.
#     """
#     return (
#         GENERATE_QUESTION_NODE
#         if state["step"] < state["max_steps"]
#         else FINAL_EVALUATION_NODE
#     )


# def should_start_or_wait(state: InterviewState) -> str:
#     """
#     Determine if the interview should start immediately or wait for user input.

#     Args:
#         state (InterviewState): Current interview state.

#     Returns:
#         str: END if waiting for user input, GET_ANSWER_NODE otherwise.
#     """
#     return END if state.get("waiting_for_user", False) else GET_ANSWER_NODE


# def get_postgres_checkpointer():
#     """
#     Create PostgreSQL checkpointer with fallback to SQLite and then Memory.

#     Returns:
#         Checkpoint saver instance (PostgresSaver, SqliteSaver, or MemorySaver)
#     """
#     try:

#         conn = psycopg.connect(settings.database_url)
#         checkpointer = PostgresSaver(conn)

#         checkpointer.setup()

#         logger.info("✅ Using PostgreSQL checkpointer")
#         return checkpointer

#     except ImportError as e:
#         logger.warning("⚠️ PostgreSQL checkpoint package not available: %s", e)
#     except psycopg.OperationalError as e:
#         logger.warning("⚠️ Cannot connect to PostgreSQL database: %s", e)
#         logger.warning("⚠️ Make sure PostgreSQL is running and database exists")
#     except Exception as e:
#         logger.warning("⚠️ PostgreSQL setup failed: %s", e)

#     except Exception as e:
#         logger.warning("⚠️ SQLite saver failed: %s", e)

#     logger.info("✅ Using in-memory checkpointer (final fallback)")
#     return MemorySaver()


# def create_interview_graph() -> StateGraph:
#     """
#     Create and compile the interview state graph with PostgreSQL checkpoints.

#     The graph orchestrates the multi-step interview process, including:
#         - Setup and initialization.
#         - Retrieving or searching context (via RAG or Tavily).
#         - Generating and evaluating interview questions.
#         - Producing final evaluations and displaying results.

#     The flow includes checkpointing support using PostgreSQL (preferred),
#     with fallbacks to SQLite and in-memory storage.

#     Returns:
#         StateGraph: Compiled interview graph ready for execution.
#     """
#     logger.info("Initializing interview graph with RAG + Tavily search flow...")
#     builder = StateGraph(InterviewState)

#     builder.add_node(SETUP_NODE, setup_node)
#     builder.add_node(GET_ANSWER_NODE, get_answer_node)
#     builder.add_node(RETRIEVAL_DECISION_NODE, retrieval_decision_node)
#     builder.add_node(RETRIEVAL_NODE, retrieval_node)
#     builder.add_node(TAVILY_SEARCH_NODE, tavily_search_node)
#     builder.add_node(GENERATE_QUESTION_NODE, generate_question_node)
#     builder.add_node(EVALUATE_QUESTION_NODE, evaluate_question_node)
#     builder.add_node(FINAL_EVALUATION_NODE, final_evaluation_node)
#     builder.add_node(DISPLAY_RESULTS_NODE, display_results_node)

#     builder.set_entry_point(SETUP_NODE)

#     builder.add_conditional_edges(
#         SETUP_NODE,
#         should_start_or_wait,
#         {GET_ANSWER_NODE: GET_ANSWER_NODE, END: END},
#     )

#     builder.add_edge(GET_ANSWER_NODE, RETRIEVAL_DECISION_NODE)

#     builder.add_conditional_edges(
#         RETRIEVAL_DECISION_NODE,
#         should_retrieve,
#         {RETRIEVAL_NODE: RETRIEVAL_NODE, TAVILY_SEARCH_NODE: TAVILY_SEARCH_NODE},
#     )

#     builder.add_edge(RETRIEVAL_NODE, GENERATE_QUESTION_NODE)
#     builder.add_edge(TAVILY_SEARCH_NODE, GENERATE_QUESTION_NODE)

#     builder.add_conditional_edges(
#         GENERATE_QUESTION_NODE,
#         should_continue,
#         {
#             GET_ANSWER_NODE: GET_ANSWER_NODE,
#             EVALUATE_QUESTION_NODE: EVALUATE_QUESTION_NODE,
#             FINAL_EVALUATION_NODE: FINAL_EVALUATION_NODE,
#             END: END,
#         },
#     )

#     builder.add_edge(EVALUATE_QUESTION_NODE, FINAL_EVALUATION_NODE)
#     builder.add_edge(FINAL_EVALUATION_NODE, DISPLAY_RESULTS_NODE)
#     builder.add_edge(DISPLAY_RESULTS_NODE, END)

#     checkpointer = get_postgres_checkpointer()

#     logger.info("✅ Interview graph successfully compiled with PostgreSQL checkpoints.")
#     return builder.compile(checkpointer=checkpointer)


# compiled_graph = create_interview_graph()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


#
#
#
#
#
#
#
#


import json
import os
import asyncio
import logging
from typing import TypedDict

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    RoomInputOptions,
)
from livekit.plugins import elevenlabs, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.langchain import LLMAdapter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from livekit.agents import AgentServer
from contextlib import asynccontextmanager

from services.gemini_client import gemini_client  # assuming this exists
from contextlib import asynccontextmanager

logger = logging.getLogger("voice-agent")
load_dotenv()
server = AgentServer()

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging

logger = logging.getLogger(__name__)


class State(TypedDict):
    user_response: str
    response: str
    step: int
    first_question: str


# -----------------------
# Start node (runs once)
# -----------------------
def setup_node(state: State):
    if state["step"] == 0:
        return {
            **state,
            "first_question": (
                "Hello! Let's begin the interview. "
                "Can you tell me about your experience with LangGraph?"
            ),
            "step": 1,
        }
    return state


# -----------------------
# Echo / LLM node
# -----------------------
def echo_node(state: State):
    try:
        response = gemini_client.generate_content(state["user_response"])
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

    graph.add_conditional_edges(
        "start", router, {"start": "start", "echo": "echo", END: END}
    )
    graph.add_conditional_edges("echo", router, {"echo": "echo", END: END})

    return graph.compile(checkpointer=MemorySaver())


async def response_stream(text: str | None):
    if text:
        yield text
    else:
        return


class LangGraphLLM(LLMAdapter):
    def __init__(self, workflow, initial_state, thread_id):
        super().__init__(graph=workflow)
        self._workflow = workflow
        self._thread_id = thread_id
        self._config = {"configurable": {"thread_id": self._thread_id}}

        # Run once to initialize state and generate first question
        self._workflow.invoke(initial_state, config=self._config)

    def get_current_response(self) -> str:
        """Retrieve the latest response from the checkpointed state."""
        try:
            state = self._workflow.get_state(self._config)
            return state.values.get("first_question") or "No response available"
        except Exception as e:
            logger.exception("Failed to get current state")
            return f"Error retrieving response: {e}"

    def normalize_user_text(self, content) -> str:
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            ).strip()
        if isinstance(content, str):
            return content.strip()
        return ""

    @asynccontextmanager
    async def chat(self, chat_ctx, **kwargs):
        response: str | None = None

        try:
            if chat_ctx.items:
                last_msg = chat_ctx.items[-1]
                user_text = getattr(last_msg, "content", None) or str(last_msg)
                # Inside chat(), just before invoke
                state_before = self._workflow.get_state(self._config).values
                logger.info(
                    f"Before invoke: step={state_before.get('step')}, "
                    f"response_preview={state_before.get('response')[:80]}, "
                    f"user_response incoming={user_text[:80]}"
                )
                if user_text:
                    result = await asyncio.to_thread(
                        lambda: self._workflow.invoke(
                            {"user_response": user_text},
                            config=self._config,
                        )
                    )
                    # After invoke

                    logger.info(
                        f"After invoke: step={result.get('step')}, "
                        f"response_preview={result.get('response')[:80]}"
                    )

                    response = result.get("response")
            else:
                response = self.get_current_response()

        except Exception as e:
            logger.exception("Error in chat")
            response = f"An internal error occurred: {e}"

        # 🔑 Yield an ASYNC ITERATOR — not a context manager
        yield response_stream(response)


class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a professional technical interviewer.")

    async def on_user_turn_completed(self, turn_ctx, new_message):
        async def filler_gen():
            yield "Okay, let me think about that..."

        try:
            await self.session.say(filler_gen(), add_to_chat_ctx=False)
        except Exception as e:
            logger.exception("Failed to send filler")


@server.rtc_session(agent_name="voice-agent")
async def entrypoint(ctx: JobContext):
    try:
        await ctx.connect()
    except Exception as e:
        logger.exception("Failed to connect")
        raise

    initial_state = {
        "user_response": "",
        "response": "",
        "step": 0,
    }

    workflow = create_workflow()
    llm = LangGraphLLM(
        workflow=workflow,
        initial_state=initial_state,
        thread_id="voice-agent-session-1234",
    )

    try:
        session = AgentSession(
            llm=llm,
            stt="assemblyai/universal-streaming:en",
            tts=elevenlabs.TTS(
                model="eleven_v2_flash",
                voice_id="CwhRBWXzGAHq8TQ4Fs17",
                api_key=os.getenv("ELEVENLABS_API_KEY"),
            ),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
            preemptive_generation=True,
        )
    except Exception as e:
        logger.exception("Failed to initialize AgentSession")
        raise

    try:
        await session.start(
            agent=VoiceAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )

        # ────────────────────────────────────────────────
        # This is the key part: speak the first question right after session start
        async def speak_first_question():
            # Get the response that was generated during __init__
            first_question = llm.get_current_response()
            yield first_question

        await session.say(speak_first_question(), add_to_chat_ctx=False)
        # ────────────────────────────────────────────────

        logger.info(f"Agent started in room: {ctx.room.name} — first question spoken")

    except Exception as e:
        logger.exception("Failed to start session")
        raise


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="voice-agent"))
