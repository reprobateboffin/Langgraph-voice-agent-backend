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
    allow_origins=["*"],  # Or ["http://localhost:5173"] for your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(interview_router)
app.include_router(auth_router)


@app.get("/")
async def root():
    return {"message": "Interview API is running 🚀"}
