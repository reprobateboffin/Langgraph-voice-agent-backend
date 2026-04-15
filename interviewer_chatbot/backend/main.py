from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from uuid import uuid4
from fastapi import FastAPI
from routes.interview import router as interview_router
from routes.user import router as auth_router
from fastapi.middleware.cors import CORSMiddleware
from graph.graph2 import compiled_graph2
from langserve import add_routes

rooms_db = {}  # Use a real DB or Redis in production
from motor.motor_asyncio import AsyncIOMotorClient
import os

uri = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(uri)
db = client["interviews_db"]

rooms_collection = db["rooms"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Clearing old index and setting new 3-day rule...")
    try:
        # This deletes the 6-hour rule currently stuck in your database
        await rooms_collection.drop_index("createdAt_1")
    except Exception:
        pass  # Ignore if it's already gone

    # This sets your new 3-day rule
    await rooms_collection.create_index("createdAt", expireAfterSeconds=259200)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # NO trailing slash, NO wildcard "*"
    allow_credentials=True,  # This MUST be True for cookies
    allow_methods=["POST", "GET", "OPTIONS", "DELETE"],  # Explicitly list or use ["*"]
    allow_headers=["Content-Type", "Set-Cookie"],  # Explicitly list or use ["*"]
)
app.include_router(interview_router)
app.include_router(auth_router)


@app.get("/")
async def root():
    return {"message": "Interview API is running 🚀"}
