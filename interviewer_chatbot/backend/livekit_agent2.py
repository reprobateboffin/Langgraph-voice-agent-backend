import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from livekit import api

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

load_dotenv()
logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)

server = AgentServer()
BACKEND_URL = "http://localhost:8000"


# -------------------- API CALLS --------------------


async def run_graph_start(initial_state: dict, thread_id: str) -> dict:
    async with httpx.AsyncClient(timeout=240) as client:
        r = await client.post(
            f"{BACKEND_URL}/start_interview",
            data={
                "job_title": str(initial_state.get("topic", "")),
                "question_type": str(initial_state.get("question_type", "")),
                "cv_text": str(initial_state.get("cv_text", "")),
                "max_step": str(initial_state.get("max_step", "5")),
                "thread_id": str(thread_id),
                "username": str(initial_state.get("username", "Muhammad")),
            },
        )
        print("START:", r.status_code, r.text)
        r.raise_for_status()
        return r.json()


async def run_graph_continue(user_response: str, thread_id: str) -> dict:
    async with httpx.AsyncClient(timeout=240) as client:
        r = await client.post(
            f"{BACKEND_URL}/continue_interview",
            json={"user_response": user_response, "thread_id": thread_id},
        )
        print("CONTINUE:", r.status_code, r.text)
        r.raise_for_status()
        return r.json()


# -------------------- LLM --------------------


class SimpleAPILLM(LLM):
    def __init__(self, initial_state: dict, thread_id: str):
        super().__init__()
        self.initial_state = initial_state
        self.thread_id = thread_id
        self.started = False
        self.finished = False

    async def get_first_question(self):
        try:
            result = await run_graph_start(self.initial_state, self.thread_id)
            self.started = True
            return result.get("message", "No question generated.")
        except Exception as e:
            print("START ERROR:", e)
            return "Failed to start interview."

    @asynccontextmanager
    async def chat(self, chat_ctx, **kwargs):
        async def stream(msg: str):
            yield msg

        try:
            if not chat_ctx.items or not self.started:
                yield stream("")
                return

            user_text = str(chat_ctx.items[-1].content)
            result = await run_graph_continue(user_text, self.thread_id)

            if result.get("finished"):
                yield stream("Interview finished.")
                self.finished = True
            else:
                yield stream(result.get("message", "No response."))

        except Exception as e:
            logger.exception("CHAT ERROR")
            yield stream("Something went wrong.")


# -------------------- AGENT --------------------


class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a technical interviewer.")

    async def on_user_turn_completed(self, turn_ctx, new_message):
        async def filler():
            yield "Let me think..."

        await self.session.say(filler(), add_to_chat_ctx=False)


# -------------------- ENTRYPOINT --------------------


@server.rtc_session(agent_name="voice-agent")
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    raw_meta = getattr(ctx.job, "metadata", None)

    if raw_meta:
        meta = json.loads(raw_meta)
        initial_state = meta.get("initial_state", {})
        thread_id = initial_state.get("thread_id", f"room-{ctx.room.name}")
    else:
        thread_id = f"room-{ctx.room.name}"
        initial_state = {
            "topic": "fallback",
            "question_type": "general",
            "cv_text": "",
            "max_step": "5",
            "thread_id": thread_id,
        }

    llm = SimpleAPILLM(initial_state, thread_id)

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
    )

    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    async def speak_first():
        yield await llm.get_first_question()

    await session.say(speak_first(), add_to_chat_ctx=False)

    async def monitor():
        while True:
            await asyncio.sleep(1)
            if llm.finished:
                lk = api.LiveKitAPI(
                    url=os.getenv("LIVEKIT_URL"),
                    api_key=os.getenv("LIVEKIT_API_KEY"),
                    api_secret=os.getenv("LIVEKIT_API_SECRET"),
                )
                await lk.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
                break

    asyncio.create_task(monitor())


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="voice-agent"))
