import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
import random

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
from livekit.plugins import elevenlabs, silero, noise_cancellation, simli
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from config.settings import settings

load_dotenv()
logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)

server = AgentServer()
BACKEND_URL = settings.backend_url


# -------------------- API CALLS --------------------


async def run_graph_start(initial_state: dict, thread_id: str) -> dict:
    interview_id = str(
        thread_id[5]
        + initial_state.get("topic", "no topic")
        + initial_state.get("thread_id", "thread1")[5]
        + initial_state.get("username", "John Doe")
    )

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
                "isCompany": bool(initial_state.get("isCompany", False)),
                "interview_id": interview_id,
                "company_name": str(initial_state.get("company_name", "None")),
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
        FILLERS = [
            "Alright, let me think about the next question.",
            "That’s helpful context, let me build on that.",
            "Interesting, let me take a second to frame the next part.",
            "Got it, let me think through where to go next.",
            "Thanks for sharing that, let me come up with the next question.",
            "That’s a good overview, let me dig a bit deeper.",
            "Nice, let me think of a follow-up here.",
            "Okay, that gives me a good picture, let me continue.",
            "I see, let me explore that area a bit more.",
            "Good, let me take a moment to structure the next question.",
            "That’s insightful, let me think about the next angle.",
            "Alright, let me push a bit further on this topic.",
            "Got it, let me come up with something more specific.",
            "That helps, let me refine the next question.",
            "Interesting approach, let me think of a deeper follow-up.",
            "Okay, let me consider the next step in this discussion.",
            "Thanks, let me take a second to think this through.",
            "That makes sense, let me expand on this direction.",
            "Alright, let me challenge that a little further.",
            "Good point, let me think of a related question.",
        ]

        async def filler():
            yield f"{random.choice(FILLERS)} Now, here's something I'd like to ask..."

        await self.session.say(filler(), add_to_chat_ctx=False)


# -------------------- ENTRYPOINT --------------------


# @server.rtc_session(agent_name="voice-agent")
# async def entrypoint(ctx: JobContext):
#     await ctx.connect()

#     raw_meta = getattr(ctx.job, "metadata", None)

#     if raw_meta:
#         meta = json.loads(raw_meta)
#         initial_state = meta.get("initial_state", {})
#         thread_id = initial_state.get("thread_id", f"room-{ctx.room.name}")
#     else:
#         thread_id = f"room-{ctx.room.name}"
#         initial_state = {
#             "topic": "fallback",
#             "question_type": "general",
#             "cv_text": "",
#             "max_step": "5",
#             "thread_id": thread_id,
#         }

#     llm = SimpleAPILLM(initial_state, thread_id)

#     voice_id = initial_state.get("voice_id", "CwhRBWXzGAHq8TQ4Fs17")
#     face_id = initial_state.get("face_id", "0c2b8b04-5274-41f1-a21c-d5c98322efa9")
#     session = AgentSession(
#         llm=llm,
#         stt="assemblyai/universal-streaming:en",
#         tts=elevenlabs.TTS(
#             model="eleven_v2_flash",
#             voice_id=voice_id,
#             api_key=os.getenv("ELEVENLABS_API_KEY"),
#         ),
#         vad=silero.VAD.load(),
#         turn_detection=MultilingualModel(),
#     )
#     simli_avatar = simli.AvatarSession(
#         simli_config=simli.SimliConfig(
#             api_key=os.getenv("SIMLI_API_KEY"),
#             face_id=face_id,
#         ),
#     )
#     try:
#         await simli_avatar.start(session, room=ctx.room)

#         # Add right after starting the avatar
#         async def simple_keep_alive():
#             while True:
#                 await asyncio.sleep(4)
#                 try:
#                     # This sends a tiny update to keep connection alive
#                     await simli_avatar.update_expression("neutral")
#                 except:
#                     pass

#         asyncio.create_task(simple_keep_alive())
#         logger.info("✅ Simli avatar started successfully")
#     except Exception as e:
#         logger.error(f"❌ Simli avatar failed to start: {type(e).__name__}: {e}")
#         logger.warning("Continuing with voice-only agent (no avatar video)")
#         # Do NOT raise — so the interview can still work with audio    await asyncio.sleep(2)  # ← Add this line (1.5–3 seconds)
#     await session.start(
#         agent=VoiceAgent(),
#         room=ctx.room,
#         room_input_options=RoomInputOptions(
#             noise_cancellation=noise_cancellation.BVC(),
#         ),
#         # Correct way to disable agent's audio output (Simli will publish audio instead)
#     )
#     await asyncio.sleep(2.0)

#     # Speak the first question reliably
#     try:
#         first_message = await llm.get_first_question()
#         logger.info(f"First question: {first_message[:150]}...")

#         await session.say(first_message, add_to_chat_ctx=False)
#         logger.info("✅ First message delivered to user")
#     except Exception as e:
#         logger.error(f"Error speaking first message: {e}")

#     async def monitor():
#         while True:
#             await asyncio.sleep(1)
#             if llm.finished:
#                 ctx.shutdown()

#                 # await simli_avatar.stop()  # clean up
#                 lk = api.LiveKitAPI(
#                     url=os.getenv("LIVEKIT_URL"),
#                     api_key=os.getenv("LIVEKIT_API_KEY"),
#                     api_secret=os.getenv("LIVEKIT_API_SECRET"),
#                 )
#                 await lk.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
#                 break

#     asyncio.create_task(monitor())


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

    voice_id = initial_state.get("voice_id", "CwhRBWXzGAHq8TQ4Fs17")
    face_id = initial_state.get("face_id", "0c2b8b04-5274-41f1-a21c-d5c98322efa9")

    session = AgentSession(
        llm=llm,
        stt="assemblyai/universal-streaming:en",
        tts=elevenlabs.TTS(
            model="eleven_v2_flash",
            voice_id=voice_id,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    simli_avatar = simli.AvatarSession(
        simli_config=simli.SimliConfig(
            api_key=os.getenv("SIMLI_API_KEY"),
            face_id=face_id,
        ),
    )

    tasks = []
    running = True
    is_cleaned = False

    # -------------------- CLEANUP + SHUTDOWN --------------------
    async def cleanup_and_shutdown():
        nonlocal running, is_cleaned

        if is_cleaned:
            return
        is_cleaned = True
        running = False

        logger.info("🧹 Cleaning up and shutting down...")

        # Cancel background tasks
        for t in tasks:
            t.cancel()

        # Stop avatar
        try:
            await simli_avatar.stop()
            logger.info("✅ Avatar stopped")
        except Exception:
            pass

        # Delete room safely
        try:
            lk = api.LiveKitAPI(
                url=os.getenv("LIVEKIT_URL"),
                api_key=os.getenv("LIVEKIT_API_KEY"),
                api_secret=os.getenv("LIVEKIT_API_SECRET"),
            )
            await lk.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
            logger.info("🗑️ Room deleted")
        except Exception as e:
            logger.warning(f"Room deletion failed: {e}")

        ctx.shutdown()

    # -------------------- DISCONNECT HANDLER --------------------
    def on_disconnect(participant):
        logger.info(f"⚠️ Participant disconnected: {participant.identity}")
        asyncio.create_task(cleanup_and_shutdown())

    ctx.room.on("participant_disconnected", on_disconnect)

    # -------------------- START AVATAR --------------------
    try:
        await simli_avatar.start(session, room=ctx.room)
        logger.info("✅ Simli avatar started")

        async def keep_alive():
            while running:
                await asyncio.sleep(4)
                try:
                    if ctx.room.connection_state == "connected":
                        await simli_avatar.update_expression("neutral")
                    else:
                        break
                except Exception:
                    break

        tasks.append(asyncio.create_task(keep_alive()))

    except Exception as e:
        logger.error(f"❌ Avatar failed: {e}")
        logger.warning("Continuing with voice-only mode")

    # -------------------- START SESSION --------------------
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await asyncio.sleep(2.0)

    # -------------------- FIRST QUESTION --------------------
    try:
        first_message = await llm.get_first_question()
        logger.info(f"First question: {first_message[:150]}...")

        await session.say(first_message, add_to_chat_ctx=False)
        logger.info("✅ First message sent")

    except Exception as e:
        logger.error(f"❌ First message error: {e}")

    # -------------------- MONITOR INTERVIEW --------------------
    async def monitor():
        while running:
            await asyncio.sleep(1)

            if llm.finished:
                logger.info("🎯 Interview finished normally")
                await cleanup_and_shutdown()
                break

    tasks.append(asyncio.create_task(monitor()))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="voice-agent"))
