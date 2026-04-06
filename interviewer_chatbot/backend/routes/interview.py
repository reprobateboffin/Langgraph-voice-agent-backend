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
    interview_id: str = Form(...),
    isCompany: bool = Form(...),
    company_name: Optional[str] = Form(...),
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
            "isCompany": isCompany,
            "interview_id": interview_id,
            "company_name": company_name,
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

from datetime import datetime, timedelta, timezone


@router.post("/register-room")
async def register_room(
    username: str = Form(...),
    job_title: str = Form(...),
    room_name: str = Form(...),
    question_type: str = Form(...),
    cv: UploadFile | None = File(None),
    max_step: str = Form(...),
    company_name: Optional[str] = Form(None),
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
    isCompany = bool(company_name)
    result = await rooms_collection.insert_one(
        {
            "job_title": job_title,
            "question_type": question_type,
            "max_step": max_step,
            "username": username,
            "cv_text": cv_text,
            "room_name": room_name,
            "isCompany": isCompany,
            "company_name": company_name,
            "createdAt": datetime.now(timezone.utc),
        }
    )
    return {"status": "registered in database"}


@router.delete("/cleanup-old-rooms")
async def cleanup_old_rooms():
    try:
        # Calculate cutoff time (2 days ago)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=2)

        # Delete all rooms older than 2 days
        result = await rooms_collection.delete_many({"createdAt": {"$lt": cutoff_date}})

        return {"status": "success", "deleted_count": result.deleted_count}

    except Exception as e:
        return {"status": "error", "message": str(e)}


class DeleteRoomRequest(BaseModel):
    room_name: str


@router.post("/delete-room")
async def delete_room(payload: DeleteRoomRequest):
    try:
        if not payload.room_name:
            raise HTTPException(status_code=400, detail="room_name is required")

        result = await rooms_collection.delete_one({"room_name": payload.room_name})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Room not found")

        return {
            "message": "Room deleted successfully",
            "room_name": payload.room_name,
        }

    except Exception as e:
        print("Error deleting room:", e)
        raise HTTPException(status_code=500, detail="Internal server error")


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
            "isCompany": room.get("isCompany", True),
            "company_name": room.get("company_name", ""),
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
    is_company: str = Form(...),
):
    # Validate room exists
    isCompany = is_company
    is_company_bool = isCompany.lower() == "true"

    # Or with better validation
    if isCompany.lower() not in ["true", "false"]:
        raise HTTPException(status_code=400, detail="isCompany must be true or false")
    is_company_bool = isCompany.lower() == "true"

    room = await rooms_collection.find_one({"room_name": room_name})
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    company_name = "None"
    # Extract fields with defaults and validation
    cv_text = room.get("cv_text", "")
    username = room.get("username")
    job_title = room.get("job_title")
    question_type = room.get("question_type")
    max_step = room.get("max_step")
    if is_company_bool:
        company_name = room.get("company_name")
    # Validate required fields
    required_fields = [username, job_title, question_type, max_step]
    if not all(required_fields):
        missing = ["username", "job_title", "question_type", "max_step"][
            required_fields.count(None) :
        ]
        raise HTTPException(
            status_code=400,
            detail=f"Room missing required fields: {', '.join(missing)}",
        )

    try:
        # Process CV if present
        if cv_text and cv_text.strip():
            document = chunk_cv_text(cv_text, user_id=username)
            create_vectorstore(document, user_id=username)

        thread_id = str(uuid.uuid4())

        metadata = {
            "initial_state": {
                "topic": job_title,
                "question_type": question_type,
                "max_step": max_step,
                "thread_id": thread_id,
                "username": username,
                "isCompany": is_company_bool,
                "room_name": room_name,
                "company_name": company_name,
            }
        }

        # Optionally exclude cv_text from metadata if not needed
        # Only add cv_text if necessary
        if cv_text:
            metadata["initial_state"]["cv_text"] = cv_text

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

    except KeyError as e:
        print(f"Missing key in room data: {e}")
        raise HTTPException(
            status_code=400, detail=f"Invalid room data: missing {str(e)}"
        )
    except Exception as e:
        print(f"JOIN ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Failed to join meeting")
