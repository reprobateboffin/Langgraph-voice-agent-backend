from typing import Any, Dict, Optional
from uuid import uuid4
from fastapi import FastAPI
from routes.interview import router as interview_router
from routes.user import router as auth_router
from fastapi.middleware.cors import CORSMiddleware
from graph.graph2 import compiled_graph2
from langserve import add_routes

# from langserve import add_routes

app = FastAPI()
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
