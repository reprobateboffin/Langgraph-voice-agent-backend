import asyncio
import json
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from pydantic import BaseModel
import uuid
from typing import Optional
from livekit.api import (
    AccessToken,
    VideoGrants,
    RoomConfiguration,
    RoomAgentDispatch,
)
from utils.cv_tools import extract_text_from_pdf_bytes, chunk_cv_text
from services.vectorstore_service import create_vectorstore, delete_vectorstore
from graph.graph import compiled_graph
from config.settings import settings

router = APIRouter(tags=["Interview"])

LIVEKIT_API_KEY = settings.livekit_api_key
LIVEKIT_API_SECRET = settings.livekit_api_secret
LIVEKIT_URL = settings.livekit_url


# -------------------- START INTERVIEW --------------------


@router.post("/start_interview")
async def start_interview(
    job_title: str = Form(...),
    question_type: str = Form(...),
    cv_text: str = Form(""),
    max_step: str = Form("5"),
    thread_id: str = Form(...),
    username: str = Form(...),
):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        user_id = username

        # only create vectorstore if cv_text exists
        if cv_text:
            document = chunk_cv_text(cv_text, user_id=user_id)
            create_vectorstore(document, user_id=user_id)

        initial_state = {
            "topic": job_title,
            "content": [],
            "cv_content": cv_text[:1000] if cv_text else "",
            "questions": [],
            "answers": [],
            "user_response": None,
            "feedback": [],
            "current_question": None,
            "current_answer": None,
            "step": 0,
            "max_steps": int(max_step),
            "final_evaluation": None,
            "messages": [],
            "question_type": question_type,
            "needs_retrieval": False,
            "retrieved_context": None,
            "similarity_score": None,
            "user_id": user_id,
            "finished": False,
        }

        final_state = compiled_graph.invoke(initial_state, config=config)

        return {
            "thread_id": thread_id,
            "status": "question",
            "message": final_state["messages"][-1]["content"],
            "current_step": final_state["step"],
            "max_steps": final_state["max_steps"],
        }

    except Exception as e:
        print("START ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- CONTINUE --------------------


class ContinueRequest(BaseModel):
    user_response: str
    thread_id: str


@router.post("/continue_interview")
async def continue_interview(req: ContinueRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        existing_state = compiled_graph.get_state(config)
        if not existing_state:
            raise HTTPException(status_code=400, detail="No ongoing interview.")

        state_dict = (
            dict(existing_state.values)
            if hasattr(existing_state, "values")
            else dict(existing_state)
        )

        state_dict["user_response"] = req.user_response
        state_dict["waiting_for_user"] = False

        final_state = compiled_graph.invoke(state_dict, config=config)
        messages = final_state.get("messages", [])

        if final_state.get("feedback"):
            delete_vectorstore(final_state.get("user_id", "default_user"))
            return {
                "thread_id": req.thread_id,
                "status": "completed",
                "message": final_state["feedback"],
                "final_evaluation": final_state["final_evaluation"],
                "current_step": final_state.get("step", 1),
                "max_steps": final_state.get("max_steps", 3),
                "finished": True,
            }

        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return {
                    "thread_id": req.thread_id,
                    "status": "question",
                    "message": msg["content"],
                    "current_step": final_state.get("step", 1),
                    "max_steps": final_state.get("max_steps", 3),
                }

        raise HTTPException(status_code=500, detail="No response generated.")

    except Exception as e:
        print("CONTINUE ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- JOIN --------------------
# In a new or existing FastAPI file
rooms_db = {}  # Use a real DB or Redis in production
from motor.motor_asyncio import AsyncIOMotorClient
import os

uri = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(uri)
db = client["interviews_db"]  # auto-created
rooms_collection = db["rooms"]  # auto-created


@router.post("/register-room")
async def register_room(
    username: str = Form(...),
    job_title: str = Form(...),
    room_name: str = Form(...),
    question_type: str = Form(...),
    cv: UploadFile | None = File(None),
    max_step: str = Form(...),
):
    # Store the settings for this specific room
    try:
        cv_text = ""

        if cv:
            cv_bytes = await cv.read()
            if cv_bytes:
                try:
                    cv_text = extract_text_from_pdf_bytes(cv_bytes)
                except Exception as e:
                    print("CV ERROR:", e)
    except Exception as e:
        print("CV ERROR:", e)
    result = await rooms_collection.insert_one(
        {
            "job_title": job_title,
            "question_type": question_type,
            "max_step": max_step,
            "username": username,
            "cv_text": cv_text,
            "room_name": room_name,
        }
    )
    return {"status": "registered in database"}


from fastapi import HTTPException
from bson import ObjectId


@router.get("/get-room-info/{room_name}")
async def get_room_info(room_name: str):
    try:
        room = await rooms_collection.find_one({"room_name": room_name})

        if not room:
            raise HTTPException(status_code=404, detail="Room not found")

        # Convert ObjectId to string
        room["_id"] = str(room["_id"])

        return {
            "username": room["username"],
            "job_title": room["job_title"],
            "question_type": room["question_type"],
            "max_step": room["max_step"],
            "room_name": room["room_name"],
            "cv_text": room.get("cv_text", ""),
        }

    except Exception as e:
        print("GET ROOM ERROR:", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/join")
async def join_meeting(
    username: str = Form(...),
    job_title: str = Form(...),
    room_name: str = Form(...),
    question_type: str = Form(...),
    cv: UploadFile | None = File(None),
    max_step: str = Form(...),
):
    try:
        cv_text = ""

        if cv:
            cv_bytes = await cv.read()
            if cv_bytes:
                try:
                    cv_text = extract_text_from_pdf_bytes(cv_bytes)
                except Exception as e:
                    print("CV ERROR:", e)

        thread_id = str(uuid.uuid4())

        if cv_text:
            document = chunk_cv_text(cv_text, user_id=username)
            create_vectorstore(document, user_id=username)

        metadata = {
            "initial_state": {
                "topic": job_title,
                "question_type": question_type,
                "cv_text": cv_text,
                "max_step": max_step,
                "thread_id": thread_id,
            }
        }

        token = (
            AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
            .with_identity(username)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .with_room_config(
                RoomConfiguration(
                    agents=[
                        RoomAgentDispatch(
                            agent_name="voice-agent",
                            metadata=json.dumps(metadata),
                        )
                    ]
                )
            )
            .to_jwt()
        )

        return {
            "token": token,
            "url": LIVEKIT_URL,
            "room_name": room_name,
        }

    except Exception as e:
        print("JOIN ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/join-meeting")
async def join_meeting(
    room_name: str = Form(...),
):
    room = await rooms_collection.find_one({"room_name": room_name}, {"_id": 0})
    if not room:
        raise HTTPException(status_code=404, detail="room not found")

    try:
        cv_text = ""
        cv_text = room["cv_text"]

        username = room["username"]
        job_title = room["job_title"]
        question_type = room["question_type"]
        max_step = room["max_step"]

        thread_id = str(uuid.uuid4())

        if cv_text:
            document = chunk_cv_text(cv_text, user_id=username)
            create_vectorstore(document, user_id=username)

        metadata = {
            "initial_state": {
                "topic": job_title,
                "question_type": question_type,
                "cv_text": cv_text,
                "max_step": max_step,
                "thread_id": thread_id,
                "username": username,
            }
        }

        token = (
            AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
            .with_identity(username)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .with_room_config(
                RoomConfiguration(
                    agents=[
                        RoomAgentDispatch(
                            agent_name="voice-agent",
                            metadata=json.dumps(metadata),
                        )
                    ]
                )
            )
            .to_jwt()
        )

        return {
            "token": token,
            "url": LIVEKIT_URL,
            "room_name": room_name,
            "job_title": job_title,
            "username": username,
            "question_type": question_type,
            "question_no": max_step,
        }

    except Exception as e:
        print("JOIN ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
