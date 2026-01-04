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
import uuid

logger = logging.getLogger("voice-agent")
load_dotenv()
server = AgentServer()


class State(TypedDict):
    """Interview workflow state - SIMPLIFIED"""

    step: int
    current_question: str


# SIMPLE NODE that just increments and returns question
def question_node(state: State):
    step = state["step"]

    # Define actual questions (not just numbers)
    questions = [
        "Hello, welcome to our company. Let's start with your qualifications.",
        "Tell me about your work experience.",
        "What are your technical skills?",
        "Why do you want to work here?",
        "Where do you see yourself in 5 years?",
        "What's your experience with version control?",
        "How do you handle tight deadlines?",
        "Do you have any questions for me?",
    ]

    # Get question based on step
    if step < len(questions):
        question = questions[step]
    else:
        question = f"Question {step + 1}: Please continue."

    return {"current_question": question, "step": step + 1}  # Increment for next time


def create_simple_workflow():
    """
    SIMPLE workflow - no checkpointer complexity.
    """
    try:
        graph = StateGraph(State)
        graph.add_node("question", question_node)
        graph.set_entry_point("question")
        graph.add_edge("question", END)
        return graph.compile()
    except Exception as e:
        logger.exception("Failed to create workflow")
        raise


class InterviewLLM(LLMAdapter):
    """
    SIMPLE LLM adapter - manual state tracking, no checkpointer bugs.
    """

    def __init__(self, max_steps=5):
        # Create simple workflow
        self.workflow = create_simple_workflow()
        super().__init__(graph=self.workflow)

        # MANUAL state tracking - SIMPLE and RELIABLE
        self.step = 0
        self.max_steps = max_steps
        self.completed = False
        self.has_spoken_first = False

        logger.info(
            f"InterviewLLM initialized: step={self.step}, max_steps={max_steps}"
        )

    def _extract_user_text(self, chat_ctx):
        """Safely extract user text."""
        if not chat_ctx.items:
            return ""

        last_msg = chat_ctx.items[-1]

        # Try different message formats
        if hasattr(last_msg, "content"):
            text = last_msg.content
        elif hasattr(last_msg, "text"):
            text = last_msg.text
        elif hasattr(last_msg, "message"):
            text = last_msg.message
        else:
            text = str(last_msg)

        # Handle lists
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)

        return str(text).strip()

    @asynccontextmanager
    async def chat(self, chat_ctx, **kwargs):
        """
        SIMPLE chat handler - no checkpoint complexity.
        """
        response = ""

        try:
            # FIRST QUESTION: Agent speaks immediately
            if not self.has_spoken_first:
                logger.info("FIRST QUESTION - Agent initiating")

                # Run workflow with current step (0)
                state = {"step": self.step, "current_question": ""}
                result = await asyncio.to_thread(lambda: self.workflow.invoke(state))

                response = result.get("current_question", "Welcome to the interview.")
                self.step = result.get("step", 1)  # Update step manually
                self.has_spoken_first = True

                logger.info(f"First question spoken. New step: {self.step}")

            # SUBSEQUENT QUESTIONS: Process user response
            else:
                # Get user input
                user_text = self._extract_user_text(chat_ctx)

                if user_text:
                    logger.info(f"User responded: {user_text[:50]}...")

                    # Check if interview is complete
                    if self.step >= self.max_steps:
                        response = "Thank you! The interview is now complete."
                        self.completed = True
                    else:
                        # Generate next question
                        state = {"step": self.step, "current_question": ""}
                        result = await asyncio.to_thread(
                            lambda: self.workflow.invoke(state)
                        )

                        response = result.get(
                            "current_question", "Next question please."
                        )
                        self.step = result.get("step", self.step + 1)

                        logger.info(f"Next question generated. New step: {self.step}")
                else:
                    # No user input yet
                    logger.debug("Waiting for user input...")
                    response = ""

        except Exception as e:
            logger.exception("Error in simple LLM")
            response = f"Sorry, I encountered an error."

        async def generator():
            if response:
                yield response

        yield generator()


class InterviewAgent(Agent):
    """
    Simple interview agent.
    """

    def __init__(self):
        super().__init__(
            instructions="You are conducting a job interview. Ask one question at a time."
        )


@server.rtc_session(agent_name="voice-agent")
async def entrypoint(ctx: JobContext):
    """
    Main entrypoint - SIMPLE and RELIABLE.
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

        # Create SIMPLE LLM (no checkpoint complexity)
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

        # TRIGGER FIRST QUESTION
        await asyncio.sleep(0.5)

        # Create empty context
        class EmptyCtx:
            items = []

        logger.info("Triggering first question...")

        # Get and speak first question
        async with llm.chat(EmptyCtx()) as generator:
            async for chunk in generator:
                first_question = chunk
                logger.info(f"Speaking: {first_question[:50]}...")

                async def speak():
                    yield first_question

                await session.say(speak())
                break

        logger.info("First question spoken successfully")

    except Exception as e:
        logger.exception("Failed to start agent")
        raise


if __name__ == "__main__":
    # Clear logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="voice-agent"))
