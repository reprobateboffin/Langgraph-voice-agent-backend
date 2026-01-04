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
from services.gemini_client import gemini_client
from contextlib import asynccontextmanager
from livekit.agents import AgentServer

logger = logging.getLogger("voice-agent")
load_dotenv()
server = AgentServer()


class State(TypedDict):
    """Workflow state for interview."""

    step: int
    max_steps: int
    current_question: str
    user_response: str
    completed: bool


def start_node(state: State):
    """Generate the first interview question."""
    question = "Hello, welcome to the interview. Please start by introducing yourself."

    return {"current_question": question, "step": 1, "completed": False}


def continue_interview_node(state: State):
    """Generate subsequent interview questions."""
    try:
        if state.get("completed", False):
            return {
                "current_question": "Thank you! Interview complete.",
                "completed": True,
            }

        if state["step"] >= state["max_steps"]:
            return {
                "current_question": "That's all the questions. Thank you!",
                "completed": True,
            }

        # Use Gemini for dynamic questions
        prompt = f"""Generate interview question number {state['step']} for a job interview.
        Previous answer was: {state.get('user_response', 'No answer yet')}
        Make it a follow-up question."""

        question = gemini_client.generate_content(prompt)

    except Exception as e:
        logger.exception("Error in continue_interview_node")
        question = f"Question {state['step']}: Please continue."

    return {"current_question": question, "step": state["step"] + 1, "completed": False}


def create_start_workflow():
    """Workflow for first question only."""
    try:
        graph = StateGraph(State)
        graph.add_node("start", start_node)
        graph.set_entry_point("start")
        graph.add_edge("start", END)
        return graph.compile()
    except Exception as e:
        logger.exception("Failed to create start workflow")
        raise


def create_continue_workflow():
    """Workflow for follow-up questions."""
    try:
        graph = StateGraph(State)
        graph.add_node("continue", continue_interview_node)
        graph.set_entry_point("continue")
        graph.add_edge("continue", END)
        return graph.compile()
    except Exception as e:
        logger.exception("Failed to create continue workflow")
        raise


def extract_user_text(chat_ctx):
    """
    Safely extract user text from chat context.
    Handles different message formats.
    """
    if not chat_ctx.items:
        return ""

    # Get the last message
    last_msg = chat_ctx.items[-1]

    # Try different ways to extract text
    if hasattr(last_msg, "content"):
        text = last_msg.content
    elif hasattr(last_msg, "text"):
        text = last_msg.text
    elif hasattr(last_msg, "message"):
        text = last_msg.message
    else:
        text = str(last_msg)

    # If text is a list, join it
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)

    # If text is still not a string, convert it
    if not isinstance(text, str):
        text = str(text)

    return text.strip()


class InterviewLLM(LLMAdapter):
    """
    LLM adapter for interview workflow.
    """

    def __init__(self, max_steps=5):
        # Create a simple dummy workflow for the parent class
        dummy_graph = create_start_workflow()
        super().__init__(graph=dummy_graph)

        self.max_steps = max_steps
        self.state = {
            "step": 0,
            "max_steps": max_steps,
            "current_question": "",
            "user_response": "",
            "completed": False,
        }
        self.start_workflow = create_start_workflow()
        self.continue_workflow = create_continue_workflow()
        self.has_asked_first = False

    @asynccontextmanager
    async def chat(self, chat_ctx, **kwargs):
        """
        Handle conversation flow.
        """
        response = ""

        try:
            # First question
            if not self.has_asked_first:
                logger.info("Getting first question")
                result = self.start_workflow.invoke(self.state)
                self.state.update(result)
                response = result.get("current_question", "Welcome.")
                self.has_asked_first = True

            else:
                # Get user input using safe extraction
                user_text = extract_user_text(chat_ctx)

                if user_text:  # Now user_text is guaranteed to be a string
                    logger.info(f"User said: {user_text[:50]}...")

                    # Store user response
                    self.state["user_response"] = user_text

                    # Check if interview is complete
                    if self.state["step"] >= self.state["max_steps"]:
                        response = "Thank you! The interview is now complete."
                        self.state["completed"] = True
                    else:
                        # Get next question
                        result = self.continue_workflow.invoke(self.state)
                        self.state.update(result)
                        response = result.get("current_question", "Next question.")
                else:
                    logger.info("No user input detected")
                    response = ""  # Don't respond to empty input

        except Exception as e:
            logger.exception("Error in LLM chat")
            response = f"Error: {str(e)[:50]}"

        async def generator():
            if response:
                yield response

        yield generator()


class InterviewAgent(Agent):
    """Interview agent."""

    def __init__(self):
        super().__init__(instructions="You are conducting a job interview.")

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """
        Optional: Send brief acknowledgment.
        """

        async def ack_gen():
            yield "Okay."

        try:
            await self.session.say(ack_gen(), add_to_chat_ctx=False)
        except Exception as e:
            logger.debug(f"Failed to send acknowledgment: {e}")


@server.rtc_session(agent_name="voice-agent")
async def entrypoint(ctx: JobContext):
    """
    Main entrypoint.
    """
    try:
        await ctx.connect()
        logger.info(f"Connected to room: {ctx.room.name}")

    except Exception as e:
        logger.exception("Failed to connect")
        return

    try:
        # Get metadata
        metadata = {}
        if ctx.room.metadata:
            try:
                metadata = json.loads(ctx.room.metadata)
                logger.info(f"Metadata: {metadata}")
            except:
                logger.warning("Could not parse metadata")

        max_steps = int(metadata.get("max_steps", 5))

        # Create LLM
        llm = InterviewLLM(max_steps)

        # Create session
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
            preemptive_generation=False,
        )

        # Start agent
        await session.start(
            agent=InterviewAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )

        logger.info("Agent session started")

        # Get and speak first question
        logger.info("Triggering first question...")

        # Create a minimal mock context
        class MockCtx:
            items = []

        # Get first question
        async with llm.chat(MockCtx()) as generator:
            async for chunk in generator:
                first_question = chunk
                logger.info(f"First question: {first_question[:50]}...")

                # Speak it
                async def speak():
                    yield first_question

                await session.say(speak())
                break

        logger.info("Agent spoke first question successfully")

    except Exception as e:
        logger.exception("Failed to start agent")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="voice-agent"))
