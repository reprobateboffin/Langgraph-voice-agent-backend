from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, Response, HTTPException, APIRouter, Body
from utils.auth import hash_password, verify_password, create_token
from pydantic import BaseModel, Field
from config.settings import settings
from motor.motor_asyncio import AsyncIOMotorClient
import os
from fastapi import (
    FastAPI,
    Response,
    HTTPException,
    APIRouter,
    Request,
    Depends,
)
from bson import ObjectId
from utils.auth import (
    hash_password,
    verify_password,
    create_token,
    SECRET,
)  # Import SECRET from auth.py
from jose import jwt, JWTError

from typing import List, Optional
from pydantic import BaseModel, Field

SECRET = "supersecret"
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig


class EmailSchema(BaseModel):
    email: str
    link: str
    username: str
    job_title: str


class OrgDetails(BaseModel):
    companyName: str
    companyAddress: str
    industry: str


class OrgRegister(BaseModel):
    username: str
    email: str
    password: str
    organization: OrgDetails
    isCompany: bool


class UserLogin(BaseModel):
    email: str
    password: str


class InterviewByIdRequest(BaseModel):
    interview_id: str


class InterviewRequest(BaseModel):
    user_id: str
    interview_id: Optional[str] = None
    type: str


router = APIRouter(tags=["user"])
LIVEKIT_API_KEY = settings.livekit_api_key
LIVEKIT_API_SECRET = settings.livekit_api_secret
LIVEKIT_URL = settings.livekit_url
MAIL_USERNAME = settings.mail_username
MAIL_PASSWORD = settings.mail_password


class UserRegister(BaseModel):
    username: str
    email: str
    password: str


uri = os.getenv("MONGODB_URI")

client = AsyncIOMotorClient(uri)
db = client["interviews_db"]
users_collection = db["user"]
interview_collection = db["interviews"]
rooms_collection = db["rooms"]


async def get_current_user(request: Request):
    token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await users_collection.find_one({"_id": ObjectId(user_id)})

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@router.post("/register")
async def register(user: UserRegister, response: Response):
    existing_user = await users_collection.find_one({"email": user.email})

    if existing_user:
        raise HTTPException(status_code=400, detail="Email exists")

    hashed_pw = hash_password(user.password)
    result = await users_collection.insert_one(
        {"username": user.username, "email": user.email, "hashed_password": hashed_pw}
    )
    user_id = str(result.inserted_id)
    token = create_token(user_id)

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=36000,
    )
    return {"message": "User registered"}


@router.post("/register-org")
async def register_org(user: OrgRegister, response: Response):
    # 1. Check if user already exists
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = hash_password(user.password)

    org_document = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_pw,
        "isCompany": True,
        "company_info": {
            "name": user.organization.companyName,
            "address": user.organization.companyAddress,
            "industry": user.organization.industry,
        },
    }

    result = await users_collection.insert_one(org_document)
    user_id = str(result.inserted_id)

    token = create_token(user_id)

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=36000,
    )

    return {
        "message": "Organization registered successfully",
        "username": user.username,
    }


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    print(f'sending username: {user["username"]}s data to frontend')
    return {"username": user["username"], "email": user["email"]}


@router.post("/login")
async def login(user_data: UserLogin, response: Response):
    user = await users_collection.find_one({"email": user_data.email})

    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if user.get("isCompany"):
        raise HTTPException(status_code=403, detail="Please login as a corporation")

    user_id = str(user["_id"])
    token = create_token(user_id)

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=3600,
    )

    return {"message": "Login successful", "username": user["username"]}


@router.post("/get-interviews")
async def get_interviews(request: InterviewRequest):
    if request.type == "user":
        query = {"user_id": request.user_id}
    else:
        query = {"company_name": request.user_id}

    projection = {
        "_id": 1,
        "interview_id": 1,
        "user_id": 1,
        "createdAt": 1,
        "room_name": 1,
    }

    cursor = interview_collection.find(query, projection)
    interviews = await cursor.to_list(length=100)

    for item in interviews:
        item["_id"] = str(item["_id"])

    return interviews


@router.post("/get-interview-results")
async def get_interview_results(request: InterviewByIdRequest):
    query = {"interview_id": request.interview_id}

    interview = await interview_collection.find_one(query)

    if not interview:
        return {"message": "Interview not found"}

    interview["_id"] = str(interview["_id"])

    return interview


@router.post("/login-org")
async def login_org(user_data: UserLogin, response: Response):
    user = await users_collection.find_one(
        {"email": user_data.email, "isCompany": True}
    )

    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid company credentials")

    user_id = str(user["_id"])
    token = create_token(user_id)

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=3600,
    )

    return {"message": "Company login successful", "username": user["username"]}


@router.post("/get-pending-interviews")
async def get_pending_interviews(request: InterviewRequest):
    if request.type == "user":
        query = {"username": request.user_id, "isCompany": False}
    else:
        query = {"company_name": request.user_id, "isCompany": True}

    cursor = rooms_collection.find(query).sort("createdAt", -1)
    rooms = await cursor.to_list(length=100)

    formatted_pending = []
    for doc in rooms:
        formatted_pending.append(
            {
                "_id": str(doc["_id"]),
                "user_id": doc.get("username"),
                "interview_id": doc.get("room_name"),
                "createdAt": (
                    doc["createdAt"].isoformat()
                    if isinstance(doc["createdAt"], datetime)
                    else doc["createdAt"]
                ),
                "job_title": doc.get("job_title", "Interview Session"),
                "room_name": doc.get("room_name", "room Session"),
            }
        )

    return formatted_pending


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="lax",
        secure=True,
    )
    return {"message": "Logged out successfully"}


conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_USERNAME"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)


@router.post("/send-invite")
async def send_invite(data: EmailSchema):
    query = {"username": data.username}
    company = await users_collection.find_one(query)

    if not company:
        raise HTTPException(status_code=404, detail="User not found")

    company_name = company.get("company_info", {}).get("name", "Unknown Company")

    message = MessageSchema(
        subject=f"Meeting Invitation from {company_name}",
        recipients=[data.email],
        body=f"Join the meeting: You have been invited to join an interview for the position {data.job_title}, you can access it at : {data.link}",
        subtype="plain",
    )

    fm = FastMail(conf)
    await fm.send_message(message)

    return {"message": "Email sent successfully"}
