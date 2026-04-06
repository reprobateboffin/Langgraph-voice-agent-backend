from typing import List, Optional
from fastapi import FastAPI, Response, HTTPException, APIRouter, Body
from utils.auth import hash_password, verify_password, create_token
from pydantic import BaseModel, Field
from config.settings import settings
from motor.motor_asyncio import AsyncIOMotorClient
import os

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
db = client["interviews_db"]  # auto-created
users_collection = db["user"]  # auto-created
interview_collection = db["interviews"]


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

    # Set HttpOnly cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",  # or 'strict'
        max_age=3600,  # 1 hour
    )
    return {"message": "User registered"}


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


@router.post("/register-org")
async def register_org(user: OrgRegister, response: Response):
    # 1. Check if user already exists
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 2. Hash password and prepare document
    hashed_pw = hash_password(user.password)

    org_document = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_pw,
        "isCompany": True,  # Flag to distinguish from regular users
        "company_info": {
            "name": user.organization.companyName,
            "address": user.organization.companyAddress,
            "industry": user.organization.industry,
        },
    }

    # 3. Insert into MongoDB
    result = await users_collection.insert_one(org_document)
    user_id = str(result.inserted_id)

    # 4. Create token and set cookie (Automatic Login)
    token = create_token(user_id)

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=3600,  # 1 hour
    )

    return {
        "message": "Organization registered successfully",
        "username": user.username,
    }


from fastapi import (
    FastAPI,
    Response,
    HTTPException,
    APIRouter,
    Request,
    Depends,
)  # Added Depends
from bson import ObjectId  # Added for MongoDB lookups
from utils.auth import (
    hash_password,
    verify_password,
    create_token,
    SECRET,
)  # Import SECRET from auth.py
from jose import jwt, JWTError

SECRET = "supersecret"


async def get_current_user(request: Request):
    token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        # Use the SAME secret imported from your auth module
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Correctly convert string user_id back to MongoDB ObjectId
    user = await users_collection.find_one({"_id": ObjectId(user_id)})

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):  # Corrected syntax
    print(f'sending username: {user["username"]}s data to frontend')
    return {"username": user["username"], "email": user["email"]}


class UserLogin(BaseModel):
    email: str
    password: str


@router.post("/login")
async def login(user_data: UserLogin, response: Response):
    # 1. Find user by email
    user = await users_collection.find_one({"email": user_data.email})

    # 2. Check if user exists AND password is correct
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if user.get("isCompany"):
        raise HTTPException(status_code=403, detail="Please login as a corporation")

    # 3. Create a fresh token
    user_id = str(user["_id"])
    token = create_token(user_id)

    # 4. Set the HttpOnly cookie (match your /register settings)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=3600,  # 1 hour
    )

    return {"message": "Login successful", "username": user["username"]}


from typing import List, Optional
from pydantic import BaseModel, Field


# 1. Create a class for the request body
class InterviewRequest(BaseModel):
    user_id: str
    interview_id: Optional[str] = None
    type: str


@router.post("/get-interviews")
async def get_interviews(request: InterviewRequest):  # <--- Use the model here
    # Access data using request.user_id
    if request.type == "user":
        query = {"user_id": request.user_id}
    else:
        query = {"company_name": request.user_id}

    # Only return specific fields
    projection = {"_id": 1, "interview_id": 1, "user_id": 1, "createdAt": 1}

    cursor = interview_collection.find(query, projection)
    interviews = await cursor.to_list(length=100)

    for item in interviews:
        item["_id"] = str(item["_id"])

    return interviews


class InterviewByIdRequest(BaseModel):
    interview_id: str


@router.post("/get-interview-results")
async def get_interview_results(request: InterviewByIdRequest):
    query = {"interview_id": request.interview_id}

    interview = await interview_collection.find_one(query)

    if not interview:
        return {"message": "Interview not found"}

    # Convert ObjectId to string
    interview["_id"] = str(interview["_id"])

    return interview


@router.post("/login-org")
async def login_org(user_data: UserLogin, response: Response):
    # 1. Find ONLY company users
    user = await users_collection.find_one(
        {"email": user_data.email, "isCompany": True}
    )

    # 2. Validate
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid company credentials")

    # 3. Token
    user_id = str(user["_id"])
    token = create_token(user_id)

    # 4. Cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=3600,
    )

    return {"message": "Company login successful", "username": user["username"]}


@router.post("/logout")
async def logout(response: Response):
    # This overwrites the cookie with an empty value and an immediate expiration
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="lax",  # Must match your registration settings
        # secure=True,  # Add this if you are using HTTPS
    )
    return {"message": "Logged out successfully"}


from fastapi_mail import FastMail, MessageSchema, ConnectionConfig


class EmailSchema(BaseModel):
    email: str
    link: str


conf = ConnectionConfig(
    MAIL_USERNAME="Muhammadakbartr11@gmail.com",
    MAIL_PASSWORD="ktup owro yhpc olqt",
    MAIL_FROM="Muhammadakbartr11@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)


@router.post("/send-invite")
async def send_invite(data: EmailSchema):
    message = MessageSchema(
        subject="Meeting Invitation",
        recipients=[data.email],
        body=f"Join the meeting: {data.link}",
        subtype="plain",
    )

    fm = FastMail(conf)
    await fm.send_message(message)

    return {"message": "Email sent successfully"}
