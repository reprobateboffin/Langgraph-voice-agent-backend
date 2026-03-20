import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from livekit import api

import httpx
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    RoomInputOptions,
    cli,
)
from livekit.agents import AgentServer
from livekit.agents.llm import LLM
from livekit.plugins import elevenlabs, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from contextlib import asynccontextmanager

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)

server = AgentServer()

BACKEND_URL = "http://localhost:8000"


# -------------------------------------------------------------------
# Backend API helpers
# -------------------------------------------------------------------


async def run_graph_start(initial_state: dict, thread_id: str) -> dict:
    """
    Start a new interview session by calling the backend /start_interview endpoint.

    Args:
        initial_state: Dictionary containing job_title, question_type, cv_text, max_step, etc.
        thread_id: Unique identifier for this interview thread

    Returns:
        Parsed JSON response from the backend (should contain the first question, etc.)

    Raises:
        httpx.HTTPStatusError: If the backend returns 4xx/5xx status
    """
    async with httpx.AsyncClient(timeout=240) as client:
        r = await client.post(
            f"{BACKEND_URL}/start_interview",
            data={
                "job_title": initial_state.get("topic"),
                "question_type": initial_state.get("question_type"),
                "cv_text": initial_state.get("cv_text"),
                "max_step": initial_state.get("max_step", "5"),
                "thread_id": thread_id,
            },
        )
        print(r.status_code)
        print(r.text)
        r.raise_for_status()
        return r.json()


from pydantic import BaseModel


class ContinueRequest(BaseModel):
    """Request model for continuing an existing interview session."""

    user_response: str
    thread_id: str


async def run_graph_continue(req: ContinueRequest) -> dict:
    """
    Send user's answer to backend and get the next question / evaluation step.

    Args:
        req: ContinueRequest object containing user's response and thread identifier

    Returns:
        Parsed JSON response from /continue_interview endpoint

    Raises:
        httpx.HTTPStatusError: On non-2xx responses from backend
    """
    async with httpx.AsyncClient(timeout=240) as client:
        r = await client.post(
            f"{BACKEND_URL}/continue_interview",
            json=req.dict(),
            timeout=240,
        )
        print(r.status_code)
        print(r.text)
        r.raise_for_status()
        return r.json()


class SimpleAPILLM(LLM):
    """
    Custom LLM adapter that delegates question generation and continuation
    to an external backend HTTP API instead of using a real language model.
    """

    def __init__(self, initial_state: dict, thread_id: str):
        super().__init__()
        self.initial_state = initial_state
        self.thread_id = thread_id
        self.finished = False
        self.interview_started = False

    async def get_first_question(self) -> str:
        """
        Retrieve the very first interview question from backend.

        Returns:
            The first question text (or fallback message if failed)
        """
        if not self.interview_started:
            result = await run_graph_start(self.initial_state, self.thread_id)
            print(
                f'resulting new message is {result.get("message","None retrieved la")}'
            )
            self.interview_started = True
            return result.get("message", "No question generated.")
        return ""

    @asynccontextmanager
    async def chat(self, chat_ctx, **kwargs):
        """
        Main chat turn handler — sends user response to backend and streams
        the next interviewer message (or finish message).
        """

        async def stream(msg: str):
            yield msg

        try:
            if not chat_ctx.items:
                yield stream("")
                return

            user_text_raw = chat_ctx.items[-1].content
            if self.interview_started:
                if isinstance(user_text_raw, list):
                    user_text = " ".join(user_text_raw)
                else:
                    user_text = str(user_text_raw)

                req = ContinueRequest(
                    user_response=user_text,
                    thread_id=self.thread_id,
                )

                result = await run_graph_continue(req)
                current_step = result.get("current_step", 1)
                current_status = result.get("status", "unfinished")
                max_step = result.get("max_steps", 5)

                if not result.get("finished"):
                    message = result.get("message", "I didn't get a response.")
                    yield stream(message)
                else:
                    yield stream(
                        "The interview has finished, you will be redirected to results page soon"
                    )
                    await asyncio.sleep(20)
                    self.finished = True

        except Exception as e:
            logger.exception("LLM chat failed")
            yield stream("Sorry, something went wrong. Please try again.")


class VoiceAgent(Agent):
    """
    Minimal agent subclass that adds a thinking filler phrase
    before the real answer is generated.
    """

    def __init__(self):
        super().__init__(instructions="You are a professional technical interviewer.")

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """
        Called after user finishes speaking → plays a short "thinking" filler.
        """

        async def filler():
            yield "Okay, let me think about that."

        try:
            await self.session.say(filler(), add_to_chat_ctx=False)
        except Exception:
            logger.exception("Failed to send filler response")


async def destroy_room(room_name: str) -> None:
    """
    Delete the LiveKit room using the management API.

    Args:
        room_name: Name of the room to delete
    """
    lk = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )

    await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))


@server.rtc_session(agent_name="voice-agent")
async def entrypoint(ctx: JobContext):
    """
    Main entry point for each LiveKit room / interview session.

    Responsibilities:
    - Connect to room
    - Parse job metadata
    - Initialize custom LLM + voice pipeline
    - Ask first question
    - Monitor interview completion and clean up room
    """
    await ctx.connect()

    raw_meta = getattr(ctx.job, "metadata", None)

    if raw_meta:
        meta = json.loads(raw_meta)
        initial_state = meta.get("initial_state", {})
    else:
        initial_state = {
            "topic": "water on mars",
            "question_type": "broad follow up",
            "cv_text": "",
            "max_step": 5,
        }

    llm = SimpleAPILLM(
        initial_state=initial_state, thread_id=f"interview-{ctx.room.name}"
    )

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

    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    async def speak_first():
        question = await llm.get_first_question()
        yield question

    await session.say(speak_first(), add_to_chat_ctx=False)

    logger.info(f"Voice agent started in room: {ctx.room.name}")

    async def monitor_interview():
        """
        Background task that watches for interview completion
        and destroys the room when finished.
        """
        while True:
            await asyncio.sleep(1)
            if llm.finished:
                await destroy_room(ctx.room.name)
                break

    asyncio.create_task(monitor_interview())


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="voice-agent",
        )
    )
