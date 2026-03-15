"""Auth router - JWT-based authentication"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
import json
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/api/auth", tags=["auth"])

SECRET_KEY = os.getenv("JWT_SECRET", "Vi-SaaS_super_secret_key_2024")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# User persistence (demo)
USERS_FILE = "data/users.json"
os.makedirs("data", exist_ok=True)

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except: return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

fake_users_db = load_users()

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in fake_users_db:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return fake_users_db[email]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@router.post("/signup", response_model=Token)
async def signup(user: UserCreate):
    if user.email in fake_users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = pwd_context.hash(user.password)
    fake_users_db[user.email] = {
        "name": user.name,
        "email": user.email,
        "hashed_password": hashed_pw,
        "created_at": datetime.utcnow().isoformat(),
        "predictions_count": 0,
    }
    save_users(fake_users_db)
    token = create_access_token({"sub": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"name": user.name, "email": user.email}
    }

@router.post("/login", response_model=Token)
async def login(creds: UserLogin):
    user = fake_users_db.get(creds.email)
    if not user or not pwd_context.verify(creds.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": creds.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"name": user["name"], "email": creds.email}
    }

@router.get("/me")
async def get_me(current_user: dict = Depends(verify_token)):
    return {k: v for k, v in current_user.items() if k != "hashed_password"}
