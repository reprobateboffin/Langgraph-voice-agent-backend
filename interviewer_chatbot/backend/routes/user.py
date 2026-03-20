from fastapi import FastAPI, Response, HTTPException, APIRouter
from utils.auth import hash_password, verify_password, create_token
from pydantic import BaseModel
from config.settings import settings
from motor.motor_asyncio import AsyncIOMotorClient
import os

router = APIRouter(tags=["user"])
LIVEKIT_API_KEY = settings.livekit_api_key
LIVEKIT_API_SECRET = settings.livekit_api_secret
LIVEKIT_URL = settings.livekit_url


class UserRegister(BaseModel):
    username: str
    email: str
    password: str


uri = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(uri)
db = client["interviews_db"]  # auto-created
users_collection = db["user"]  # auto-created


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
