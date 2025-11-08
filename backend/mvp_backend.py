#!/usr/bin/env python3
"""
QFLARE MVP Backend - Clean Architecture
Industry-standard authentication with JWT, OAuth2, and key management
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
import uvicorn
from contextlib import asynccontextmanager

# Import key management system
from key_management import key_manager, registration_flow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"  # Enable demo mode by default

# Password hashing - simplified for MVP
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# OAuth2 and JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
security = HTTPBearer(auto_error=False)

# Database setup
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./qflare_mvp.db")
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")  # admin, user
    is_active = Column(Boolean, default=False)  # Requires admin approval
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    google_id = Column(String, nullable=True)  # For Google OAuth
    
    # Relationships
    keys = relationship("UserKeys", back_populates="user")
    models = relationship("UserModel", back_populates="owner")

class UserKeys(Base):
    __tablename__ = "user_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    public_key = Column(Text, nullable=False)
    private_key_encrypted = Column(Text, nullable=False)
    key_type = Column(String, default="kyber1024")  # Post-quantum key type
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="keys")

class UserModel(Base):
    __tablename__ = "user_models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    model_name = Column(String, nullable=False)
    model_description = Column(Text, nullable=True)
    model_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    owner = relationship("User", back_populates="models")

class AdminAction(Base):
    __tablename__ = "admin_actions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    admin_id = Column(String, ForeignKey("users.id"))
    action_type = Column(String, nullable=False)  # approve_user, generate_keys, etc.
    target_user_id = Column(String, ForeignKey("users.id"))
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=20)
    full_name: Optional[str] = None
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    model_config = {"from_attributes": True}
    
    id: str
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Database helpers
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Security helpers
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return TokenData(email=email)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    token_data = verify_token(token)
    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security), 
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Optional authentication - returns None if no token or in demo mode"""
    if DEMO_MODE:
        # In demo mode, return a demo user
        demo_user = db.query(User).filter(User.email == "demo@qflare.com").first()
        if not demo_user:
            # Create demo user if it doesn't exist
            demo_user = User(
                email="demo@qflare.com",
                username="demo",
                full_name="Demo User",
                hashed_password=pwd_context.hash("demo"),
                role="user",
                is_active=True,
                is_verified=True
            )
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)
        return demo_user
    
    if credentials is None:
        return None
    
    try:
        token = credentials.credentials
        token_data = verify_token(token)
        user = db.query(User).filter(User.email == token_data.email).first()
        return user
    except:
        return None

async def get_current_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Initialize default admin user
def create_default_admin(db: Session):
    admin_email = "admin@qflare.com"
    existing_admin = db.query(User).filter(User.email == admin_email).first()
    
    if not existing_admin:
        admin_user = User(
            email=admin_email,
            username="admin",
            full_name="System Administrator",
            hashed_password=get_password_hash("admin123"),  # Change this!
            role="admin",
            is_active=True,
            is_verified=True
        )
        db.add(admin_user)
        db.commit()
        logger.info("Created default admin user")
    
    # Also create a test user for easier testing
    test_email = "user@qflare.com"
    existing_user = db.query(User).filter(User.email == test_email).first()
    
    if not existing_user:
        test_user = User(
            email=test_email,
            username="testuser",
            full_name="Test User",
            hashed_password=get_password_hash("user123"),
            role="user",
            is_active=True,  # Pre-approved for testing
            is_verified=True
        )
        db.add(test_user)
        db.commit()
        logger.info("Created test user")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting QFLARE MVP Backend...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create default admin
    db = SessionLocal()
    try:
        create_default_admin(db)
    finally:
        db.close()
    
    logger.info("✅ QFLARE MVP Backend started successfully")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down QFLARE MVP Backend...")

# Create FastAPI app
app = FastAPI(
    title="QFLARE MVP API",
    description="Quantum-Safe Federated Learning - Clean Architecture MVP",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication Routes
@app.post("/api/auth/register", response_model=dict)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """User registration - requires admin approval"""
    
    # Check if user already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create new user (inactive until admin approves)
    user = User(
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        role="user",
        is_active=False,  # Requires admin approval
        is_verified=False
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    logger.info(f"New user registered: {user.email} (pending admin approval)")
    
    return {
        "message": "Registration successful! Please wait for admin approval.",
        "user_id": user.id,
        "status": "pending_approval"
    }

@app.post("/api/auth/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """User login with JWT tokens"""
    
    user = db.query(User).filter(User.email == user_credentials.email).first()
    
    if not user or not verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account pending admin approval. Use test credentials: admin@qflare.com/admin123 or user@qflare.com/user123"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": user.email})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@app.post("/api/auth/logout")
async def logout():
    """Logout (client should delete tokens)"""
    return {"message": "Successfully logged out"}

# Admin Routes
@app.get("/api/admin/pending-users")
async def get_pending_users(admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get users pending approval"""
    pending_users = db.query(User).filter(User.is_active == False, User.role == "user").all()
    return {"pending_users": [UserResponse.model_validate(user) for user in pending_users]}

@app.post("/api/admin/approve-user/{user_id}")
async def approve_user(user_id: str, admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Approve a user and generate their keys"""
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Activate user
    user.is_active = True
    user.is_verified = True
    
    # Generate post-quantum keys for the user using key management system
    key_data = key_manager.generate_client_keypair(user.id, "Kyber1024")
    
    user_keys = UserKeys(
        user_id=user.id,
        public_key=key_data["kem_public_key"],
        private_key_encrypted=key_data["kem_private_key_encrypted"],
        key_type=key_data["algorithm"]
    )
    
    db.add(user_keys)
    
    # Log admin action
    admin_action = AdminAction(
        admin_id=admin_user.id,
        action_type="approve_user",
        target_user_id=user.id,
        details=f"Approved user {user.email} and generated keys"
    )
    db.add(admin_action)
    
    db.commit()
    
    logger.info(f"Admin {admin_user.email} approved user {user.email}")
    
    return {"message": f"User {user.email} approved and keys generated"}

# User Dashboard Endpoints
@app.get("/api/user/stats")
async def get_user_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user statistics for dashboard"""
    try:
        # Count user's models (mock data for now)
        user_keys = db.query(UserKeys).filter(UserKeys.user_id == current_user.id).first()
        
        stats = {
            "totalModels": 0,  # TODO: Implement models table
            "activeModels": 0,
            "trainingRounds": 0,
            "keysGenerated": bool(user_keys)
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/user/models")
async def get_user_models(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's models"""
    try:
        # For now, return empty list - TODO: implement models table
        return {"models": []}
    except Exception as e:
        logger.error(f"Error getting user models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Admin Dashboard Endpoints  
@app.get("/admin/users")
async def get_all_users(current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get all users for admin dashboard"""
    try:
        users = db.query(User).all()
        return {
            "users": [
                {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "name": user.name,
                    "role": user.role,
                    "is_verified": user.is_verified,
                    "created_at": user.created_at.isoformat()
                }
                for user in users
            ]
        }
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/admin/stats")
async def get_admin_stats(current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get system statistics for admin dashboard"""
    try:
        total_users = db.query(User).count()
        pending_users = db.query(User).filter(User.is_verified == False).count()
        active_users = db.query(User).filter(User.is_verified == True).count()
        total_keys = db.query(UserKeys).count()
        
        stats = {
            "totalUsers": total_users,
            "pendingUsers": pending_users,
            "activeUsers": active_users,
            "totalKeys": total_keys
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/admin/users/{user_id}/approve")
async def approve_user_new(user_id: str, notes: str = "", current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Approve a user (new endpoint for clean dashboard)"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.is_verified:
            raise HTTPException(status_code=400, detail="User already verified")
        
        # Approve user
        user.is_verified = True
        
        # Generate keys
        try:
            user_keys = registration_flow.complete_registration(user.id, user.email)
            
            # Store keys in database
            db_keys = UserKeys(
                user_id=user.id,
                kyber_public_key=user_keys["kyber_public_key"],
                kyber_private_key=user_keys["kyber_private_key"],
                dilithium_public_key=user_keys["dilithium_public_key"],
                dilithium_private_key=user_keys["dilithium_private_key"]
            )
            db.add(db_keys)
            
        except Exception as key_error:
            logger.warning(f"Key generation failed: {key_error}")
            # Continue with user approval even if key generation fails
        
        # Log admin action
        admin_action = AdminAction(
            admin_id=current_user.id,
            user_id=user.id,
            action_type="user_approved",
            notes=notes
        )
        db.add(admin_action)
        
        db.commit()
        
        logger.info(f"Admin {current_user.email} approved user {user.email}")
        
        return {"message": f"User {user.email} approved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/admin/users/{user_id}/reject")
async def reject_user(user_id: str, notes: str = "", current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Reject a user registration"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Log admin action
        admin_action = AdminAction(
            admin_id=current_user.id,
            user_id=user.id,
            action_type="user_rejected",
            notes=notes
        )
        db.add(admin_action)
        
        # Delete user (or mark as rejected)
        db.delete(user)
        db.commit()
        
        logger.info(f"Admin {current_user.email} rejected user {user.email}")
        
        return {"message": f"User {user.email} rejected successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/admin/users/{user_id}/generate-keys")
async def generate_user_keys(user_id: str, current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Generate keys for a specific user"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.is_verified:
            raise HTTPException(status_code=400, detail="User must be verified first")
        
        # Check if keys already exist
        existing_keys = db.query(UserKeys).filter(UserKeys.user_id == user.id).first()
        if existing_keys:
            raise HTTPException(status_code=400, detail="Keys already exist for this user")
        
        # Generate keys
        user_keys = registration_flow.complete_registration(user.id, user.email)
        
        # Store keys in database
        db_keys = UserKeys(
            user_id=user.id,
            kyber_public_key=user_keys["kyber_public_key"],
            kyber_private_key=user_keys["kyber_private_key"],
            dilithium_public_key=user_keys["dilithium_public_key"],
            dilithium_private_key=user_keys["dilithium_private_key"]
        )
        db.add(db_keys)
        
        # Log admin action
        admin_action = AdminAction(
            admin_id=current_user.id,
            user_id=user.id,
            action_type="keys_generated",
            notes="Keys generated by admin"
        )
        db.add(admin_action)
        
        db.commit()
        
        logger.info(f"Admin {current_user.email} generated keys for user {user.email}")
        
        return {"message": f"Keys generated successfully for {user.email}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating keys: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================
# COMPREHENSIVE ADMIN ENDPOINTS
# ============================================

@app.get("/api/admin/dashboard-stats")
async def get_admin_dashboard_stats(admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get comprehensive dashboard statistics for admin"""
    try:
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        pending_users = db.query(User).filter(User.is_active == False).count()
        admin_count = db.query(User).filter(User.role == "admin").count()
        user_count = db.query(User).filter(User.role == "user").count()
        
        # Get recent registrations (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_registrations = db.query(User).filter(User.created_at >= seven_days_ago).count()
        
        # Get total keys
        total_keys = db.query(UserKeys).count()
        
        # Get recent logins today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_logins = db.query(User).filter(User.last_login >= today_start).count()
        
        return {
            "totalUsers": total_users,
            "activeUsers": active_users,
            "pendingUsers": pending_users,
            "adminCount": admin_count,
            "userCount": user_count,
            "recentRegistrations": recent_registrations,
            "totalKeys": total_keys,
            "todayLogins": today_logins
        }
    except Exception as e:
        logger.error(f"Error getting admin dashboard stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/admin/all-users")
async def get_all_users_detailed(admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get all users with detailed information"""
    try:
        users = db.query(User).order_by(User.created_at.desc()).all()
        
        user_list = []
        for user in users:
            has_keys = db.query(UserKeys).filter(UserKeys.user_id == user.id).first() is not None
            user_list.append({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "has_keys": has_keys,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login": user.last_login.isoformat() if user.last_login else None
            })
        
        return {"users": user_list}
    except Exception as e:
        logger.error(f"Error getting all users: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/api/admin/user/{user_id}")
async def delete_user(user_id: str, admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Delete a user (admin only)"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.role == "admin" and user.id == admin_user.id:
            raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
        
        # Delete user's keys first
        db.query(UserKeys).filter(UserKeys.user_id == user_id).delete()
        
        # Delete user
        db.delete(user)
        db.commit()
        
        logger.info(f"Admin {admin_user.email} deleted user {user.email}")
        return {"message": f"User {user.email} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/api/admin/user/{user_id}/role")
async def update_user_role(
    user_id: str, 
    new_role: str,
    admin_user: User = Depends(get_current_admin_user), 
    db: Session = Depends(get_db)
):
    """Update user role (admin only)"""
    try:
        if new_role not in ["admin", "user"]:
            raise HTTPException(status_code=400, detail="Invalid role")
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.id == admin_user.id:
            raise HTTPException(status_code=400, detail="Cannot change your own role")
        
        user.role = new_role
        db.commit()
        
        logger.info(f"Admin {admin_user.email} changed {user.email} role to {new_role}")
        return {"message": f"User role updated to {new_role}"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user role: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================
# COMPREHENSIVE USER ENDPOINTS
# ============================================

@app.get("/api/user/dashboard-stats")
async def get_user_dashboard_stats(current_user: Optional[User] = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    """Get comprehensive dashboard statistics for user"""
    try:
        # Use demo user if no current user
        if not current_user:
            return {
                "username": "demo",
                "email": "demo@qflare.com",
                "full_name": "Demo User",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "has_keys": True,
                "account_age_days": 30,
                "created_at": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                "last_login": datetime.utcnow().isoformat(),
                "key_algorithm": "KYBER1024",
                "key_created_at": (datetime.utcnow() - timedelta(days=30)).isoformat()
            }
        
        # Check if user has keys
        user_keys = db.query(UserKeys).filter(UserKeys.user_id == current_user.id).first()
        has_keys = user_keys is not None
        
        # Calculate account age
        account_age_days = (datetime.utcnow() - current_user.created_at).days if current_user.created_at else 0
        
        # Check last login
        last_login = current_user.last_login.isoformat() if current_user.last_login else None
        
        return {
            "username": current_user.username,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "role": current_user.role,
            "is_active": current_user.is_active,
            "is_verified": current_user.is_verified,
            "has_keys": has_keys,
            "account_age_days": account_age_days,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "last_login": last_login,
            "key_algorithm": user_keys.key_type if user_keys else None,
            "key_created_at": user_keys.created_at.isoformat() if user_keys else None
        }
    except Exception as e:
        logger.error(f"Error getting user dashboard stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/user/keys")
async def get_user_keys(current_user: Optional[User] = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    """Get user's cryptographic keys information"""
    try:
        # Return demo keys if no current user
        if not current_user:
            demo_key = {
                "id": "demo-key-1",
                "key_type": "KYBER1024",
                "is_active": True,
                "created_at": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                "public_key_preview": "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA..."
            }
            return {"keys": [demo_key], "total_keys": 1}
        
        keys = db.query(UserKeys).filter(UserKeys.user_id == current_user.id).all()
        
        key_list = []
        for key in keys:
            key_list.append({
                "id": key.id,
                "key_type": key.key_type,
                "is_active": key.is_active,
                "created_at": key.created_at.isoformat() if key.created_at else None,
                "public_key_preview": key.public_key[:50] + "..." if len(key.public_key) > 50 else key.public_key
            })
        
        return {"keys": key_list, "total_keys": len(key_list)}
    except Exception as e:
        logger.error(f"Error getting user keys: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/user/generate-keys")
async def generate_user_keys(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Generate new cryptographic keys for user"""
    try:
        # Check if user already has active keys
        existing_keys = db.query(UserKeys).filter(
            UserKeys.user_id == current_user.id,
            UserKeys.is_active == True
        ).first()
        
        if existing_keys:
            # Deactivate old keys automatically and generate new ones (key rotation)
            existing_keys.is_active = False
            logger.info(f"User {current_user.email} rotating keys - deactivating old keys")
        
        # Generate new keys
        key_data = key_manager.generate_client_keypair(current_user.id, "Kyber1024")
        
        new_keys = UserKeys(
            user_id=current_user.id,
            public_key=key_data["kem_public_key"],
            private_key_encrypted=key_data["kem_private_key_encrypted"],
            key_type=key_data["algorithm"],
            is_active=True
        )
        
        db.add(new_keys)
        db.commit()
        
        logger.info(f"User {current_user.email} generated new keys")
        return {
            "message": "Keys generated successfully",
            "key_type": key_data["algorithm"],
            "key_id": new_keys.id
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error generating user keys: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/user/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get user profile information"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None
    }

@app.put("/api/user/profile")
async def update_user_profile(
    full_name: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    try:
        if full_name is not None:
            current_user.full_name = full_name
        
        db.commit()
        db.refresh(current_user)
        
        logger.info(f"User {current_user.email} updated profile")
        return {
            "message": "Profile updated successfully",
            "user": {
                "username": current_user.username,
                "email": current_user.email,
                "full_name": current_user.full_name
            }
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ==================== FL TRAINING ENDPOINTS ====================

@app.post("/api/training/start")
async def start_fl_training(
    model_type: str = "CNN",
    dataset: str = "MNIST",
    rounds: int = 10,
    epsilon: float = 0.1,
    delta: float = 1e-6,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a new federated learning training session"""
    try:
        training_id = str(uuid.uuid4())
        
        # Create training session record
        training_session = {
            "id": training_id,
            "user_id": current_user.id,
            "model_type": model_type,
            "dataset": dataset,
            "total_rounds": rounds,
            "current_round": 0,
            "epsilon": epsilon,
            "delta": delta,
            "status": "initializing",
            "accuracy": 0.0,
            "loss": 0.0,
            "start_time": datetime.utcnow().isoformat(),
            "participants": 3,
            "byzantine_detected": 0
        }
        
        logger.info(f"User {current_user.email} started FL training session {training_id}")
        
        return {
            "message": "Training session started",
            "training_id": training_id,
            "session": training_session
        }
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{training_id}/status")
async def get_training_status(
    training_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get real-time training status with simulated progress"""
    try:
        # Simulate training progress
        import random
        round_num = random.randint(1, 10)
        
        return {
            "training_id": training_id,
            "status": "training" if round_num < 10 else "completed",
            "current_round": round_num,
            "total_rounds": 10,
            "accuracy": min(0.89 + (round_num * 0.01), 0.98),
            "loss": max(0.5 - (round_num * 0.04), 0.05),
            "epsilon_used": round_num * 0.01,
            "epsilon_budget": 0.1,
            "delta": 1e-6,
            "participants": 3,
            "byzantine_detected": random.randint(0, 1),
            "current_learning_rate": 0.001 / (1 + round_num * 0.1),
            "computation_time": round_num * 2.5,
            "communication_overhead": round_num * 1.2
        }
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/history")
async def get_training_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's training history"""
    try:
        # Simulate training history
        history = [
            {
                "id": str(uuid.uuid4()),
                "model_type": "CNN",
                "dataset": "MNIST",
                "rounds": 10,
                "final_accuracy": 0.97,
                "epsilon_used": 0.09,
                "status": "completed",
                "created_at": (datetime.utcnow() - timedelta(days=2)).isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "model_type": "ResNet18",
                "dataset": "CIFAR-10",
                "rounds": 15,
                "final_accuracy": 0.85,
                "epsilon_used": 0.12,
                "status": "completed",
                "created_at": (datetime.utcnow() - timedelta(days=7)).isoformat()
            }
        ]
        
        return {
            "total": len(history),
            "sessions": history
        }
    except Exception as e:
        logger.error(f"Error fetching training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/{training_id}/stop")
async def stop_training(
    training_id: str,
    current_user: User = Depends(get_current_user)
):
    """Stop an active training session"""
    try:
        logger.info(f"User {current_user.email} stopped training {training_id}")
        return {
            "message": "Training stopped successfully",
            "training_id": training_id,
            "status": "stopped"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PRIVACY METRICS ENDPOINTS ====================

@app.get("/api/privacy/metrics")
async def get_privacy_metrics(current_user: Optional[User] = Depends(get_current_user_optional)):
    """Get differential privacy metrics"""
    try:
        return {
            "epsilon_budget": 0.1,
            "epsilon_used": 0.073,
            "epsilon_remaining": 0.027,
            "delta": 1e-6,
            "noise_multiplier": 1.1,
            "clipping_threshold": 1.0,
            "privacy_level": "High",
            "queries_executed": 147,
            "privacy_accountant": {
                "method": "RDP",  # Rényi Differential Privacy
                "orders": [2, 4, 8, 16, 32, 64],
                "rdp_values": [0.015, 0.021, 0.028, 0.035, 0.042, 0.049]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/privacy/budget-history")
async def get_privacy_budget_history(current_user: Optional[User] = Depends(get_current_user_optional)):
    """Get privacy budget consumption over time"""
    try:
        history = [
            {"timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(), 
             "epsilon_used": 0.01 * (24 - i), 
             "queries": 10 * (24 - i)}
            for i in range(24)
        ]
        return {"history": history[::-1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== BYZANTINE DETECTION ENDPOINTS ====================

@app.get("/api/security/byzantine-status")
async def get_byzantine_status(current_user: Optional[User] = Depends(get_current_user_optional)):
    """Get Byzantine attack detection status"""
    try:
        import random
        return {
            "total_nodes": 5,
            "suspicious_nodes": random.randint(0, 2),
            "attacks_detected": random.randint(0, 3),
            "attacks_blocked": random.randint(0, 3),
            "detection_method": "Cosine Similarity + Median Aggregation",
            "threshold": 0.85,
            "last_scan": datetime.utcnow().isoformat(),
            "security_level": "Protected",
            "suspicious_activities": [
                {
                    "node_id": "node_" + str(random.randint(100, 999)),
                    "timestamp": (datetime.utcnow() - timedelta(minutes=random.randint(5, 60))).isoformat(),
                    "threat_level": random.choice(["Low", "Medium"]),
                    "reason": random.choice([
                        "Gradient deviation exceeds threshold",
                        "Anomalous update pattern detected",
                        "Cosine similarity below threshold"
                    ])
                }
                for _ in range(random.randint(0, 3))
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/security/audit-log")
async def get_security_audit_log(
    current_user: Optional[User] = Depends(get_current_user_optional),
    limit: int = 50
):
    """Get security audit log"""
    try:
        import random
        # Use demo user if no current user in demo mode
        user_id = current_user.id if current_user else "demo"
        events = [
            {
                "id": str(uuid.uuid4()),
                "timestamp": (datetime.utcnow() - timedelta(minutes=i*10)).isoformat(),
                "event_type": random.choice(["key_rotation", "login", "byzantine_detected", "training_started"]),
                "severity": random.choice(["info", "warning", "critical"]),
                "description": random.choice([
                    "User logged in successfully",
                    "Cryptographic keys rotated",
                    "Byzantine node detected and blocked",
                    "FL training session initiated"
                ]),
                "user_id": user_id
            }
            for i in range(min(limit, 20))
        ]
        return {"total": len(events), "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== DATASET ENDPOINTS ====================

@app.get("/api/datasets")
async def get_available_datasets(current_user: User = Depends(get_current_user)):
    """Get list of available datasets"""
    try:
        datasets = [
            {
                "id": "mnist",
                "name": "MNIST",
                "description": "Handwritten digit recognition",
                "samples": 60000,
                "classes": 10,
                "size": "11 MB",
                "type": "Image Classification"
            },
            {
                "id": "cifar10",
                "name": "CIFAR-10",
                "description": "Object recognition in images",
                "samples": 50000,
                "classes": 10,
                "size": "163 MB",
                "type": "Image Classification"
            },
            {
                "id": "fashion_mnist",
                "name": "Fashion MNIST",
                "description": "Fashion product images",
                "samples": 60000,
                "classes": 10,
                "size": "30 MB",
                "type": "Image Classification"
            }
        ]
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets/{dataset_id}/stats")
async def get_dataset_statistics(
    dataset_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed dataset statistics"""
    try:
        return {
            "dataset_id": dataset_id,
            "total_samples": 60000,
            "training_samples": 50000,
            "validation_samples": 10000,
            "classes": 10,
            "class_distribution": {str(i): 6000 for i in range(10)},
            "image_shape": [28, 28, 1],
            "data_type": "uint8",
            "normalized": False,
            "augmented": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MODEL MANAGEMENT ENDPOINTS ====================

@app.get("/api/models")
async def get_available_models(current_user: User = Depends(get_current_user)):
    """Get list of available model architectures"""
    try:
        models = [
            {
                "id": "cnn_simple",
                "name": "Simple CNN",
                "description": "2 Conv layers + 2 FC layers",
                "parameters": "~50K",
                "suitable_for": ["MNIST", "Fashion MNIST"]
            },
            {
                "id": "resnet18",
                "name": "ResNet-18",
                "description": "18-layer residual network",
                "parameters": "~11M",
                "suitable_for": ["CIFAR-10", "ImageNet"]
            },
            {
                "id": "mobilenet",
                "name": "MobileNet v2",
                "description": "Efficient mobile architecture",
                "parameters": "~3.5M",
                "suitable_for": ["CIFAR-10", "Custom datasets"]
            }
        ]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SYSTEM MONITORING ENDPOINTS ====================

@app.get("/api/system/metrics")
async def get_system_metrics(current_user: User = Depends(get_current_user)):
    """Get real-time system metrics"""
    try:
        import random
        return {
            "cpu_usage": random.uniform(20, 80),
            "memory_usage": random.uniform(40, 70),
            "gpu_usage": random.uniform(0, 95),
            "network_in": random.uniform(1, 10),  # MB/s
            "network_out": random.uniform(1, 10),  # MB/s
            "active_connections": random.randint(1, 5),
            "uptime": 86400 * 7,  # 7 days in seconds
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/nodes")
async def get_connected_nodes(current_user: User = Depends(get_current_user)):
    """Get list of connected FL nodes"""
    try:
        import random
        nodes = [
            {
                "node_id": f"node_{i+1}",
                "status": random.choice(["active", "idle", "training"]),
                "ip_address": f"192.168.1.{i+10}",
                "last_seen": (datetime.utcnow() - timedelta(minutes=random.randint(0, 30))).isoformat(),
                "contribution_score": random.uniform(0.7, 1.0),
                "total_rounds": random.randint(50, 200),
                "byzantine_score": random.uniform(0, 0.2)
            }
            for i in range(random.randint(3, 7))
        ]
        return {"total_nodes": len(nodes), "nodes": nodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NOTIFICATION ENDPOINTS ====================

@app.get("/api/notifications")
async def get_notifications(
    current_user: Optional[User] = Depends(get_current_user_optional),
    unread_only: bool = False
):
    """Get user notifications"""
    try:
        import random
        notifications = [
            {
                "id": str(uuid.uuid4()),
                "title": "Training Completed",
                "message": "Your FL training session completed with 97% accuracy",
                "type": "success",
                "read": False,
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Byzantine Attack Detected",
                "message": "Suspicious activity detected from node_342. Node has been quarantined.",
                "type": "warning",
                "read": False,
                "timestamp": (datetime.utcnow() - timedelta(hours=5)).isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Key Rotation Reminder",
                "message": "Your cryptographic keys are 90 days old. Consider rotating them.",
                "type": "info",
                "read": True,
                "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat()
            }
        ]
        
        if unread_only:
            notifications = [n for n in notifications if not n["read"]]
        
        return {
            "total": len(notifications),
            "unread": len([n for n in notifications if not n["read"]]),
            "notifications": notifications
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user)
):
    """Mark notification as read"""
    try:
        return {"message": "Notification marked as read", "notification_id": notification_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

@app.get("/")
async def root():
    return {
        "message": "QFLARE MVP API",
        "version": "2.0.0",
        "docs": "/docs"
    }

# ==================== FL VISUALIZATION ENDPOINTS ====================

@app.get("/api/training/{training_id}/nodes")
async def get_training_nodes(
    training_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get nodes participating in a specific training session"""
    try:
        # Generate mock nodes based on training session
        import random
        from datetime import datetime
        
        # Simulate 3-5 edge nodes
        num_nodes = random.randint(3, 5)
        nodes = []
        
        for i in range(num_nodes):
            node_id = f"node-{i+1:03d}"
            is_byzantine = (i == 3) and (num_nodes >= 4)  # Make node 4 Byzantine if exists
            
            nodes.append({
                "node_id": node_id,
                "node_name": f"Edge Node {i+1}",
                "status": "error" if is_byzantine else random.choice(["training", "active", "idle"]),
                "last_seen": datetime.now().isoformat(),
                "total_samples": random.randint(3000, 8000),
                "local_accuracy": 0.45 if is_byzantine else random.uniform(0.85, 0.95),
                "local_loss": 1.87 if is_byzantine else random.uniform(0.15, 0.35),
                "contribution_weight": 0.05 if is_byzantine else random.uniform(0.15, 0.35),
                "rounds_participated": 3 if is_byzantine else random.randint(5, 10),
                "data_distribution": random.choice(["IID", "Non-IID", "Skewed"]),
                "computing_power": 54 if is_byzantine else random.randint(70, 95),
                "network_latency": 156 if is_byzantine else random.randint(30, 80),
                "is_byzantine": is_byzantine,
            })
        
        return {"nodes": nodes, "training_id": training_id}
    except Exception as e:
        logger.error(f"Error fetching nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{training_id}/convergence")
async def get_convergence_data(
    training_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get model convergence data across all rounds"""
    try:
        import random
        
        # Always generate convergence data based on current round from status endpoint
        # Simulate a random current round for demonstration
        current_round = random.randint(5, 10)
        
        # Generate convergence history up to current round
        history = []
        num_nodes = random.randint(3, 5)
        
        for round_num in range(1, current_round + 1):
            # Simulate improving accuracy and decreasing loss over rounds
            base_accuracy = 0.5 + (round_num * 0.045)
            base_loss = 2.0 - (round_num * 0.18)
            
            round_data = {
                "round": round_num,
                "global_accuracy": min(0.98, base_accuracy + random.uniform(-0.01, 0.01)),
                "global_loss": max(0.05, base_loss + random.uniform(-0.05, 0.05)),
                "nodes": {},
            }
            
            # Add per-node metrics
            for i in range(num_nodes):
                node_id = f"node-{i+1:03d}"
                # Each node has slightly different performance
                node_variance = random.uniform(-0.03, 0.03)
                round_data["nodes"][node_id] = {
                    "accuracy": min(0.98, base_accuracy + node_variance),
                    "loss": max(0.05, base_loss + random.uniform(-0.1, 0.1)),
                }
            
            history.append(round_data)
        
        return {
            "training_id": training_id,
            "history": history,
            "total_rounds": current_round,
        }
    except Exception as e:
        logger.error(f"Error fetching convergence data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{training_id}/nodes/{node_id}")
async def get_node_details(
    training_id: str,
    node_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed metrics for a specific node"""
    try:
        import random
        
        # Generate node details with round history
        node_details = {
            "node_id": node_id,
            "node_name": f"Edge Node {node_id.split('-')[1]}",
            "status": "training",
            "last_seen": datetime.now().isoformat(),
            "total_samples": random.randint(4000, 7000),
            "local_accuracy": random.uniform(0.88, 0.94),
            "local_loss": random.uniform(0.18, 0.28),
            "contribution_weight": random.uniform(0.20, 0.30),
            "rounds_participated": 8,
            "data_distribution": "Non-IID",
            "computing_power": random.randint(75, 90),
            "network_latency": random.randint(35, 65),
            "is_byzantine": False,
            "round_history": [
                {
                    "round": i,
                    "accuracy": 0.5 + (i * 0.05) + random.uniform(-0.02, 0.02),
                    "loss": 2.0 - (i * 0.2) + random.uniform(-0.1, 0.1),
                    "samples_processed": random.randint(500, 800),
                    "training_time": 120 + random.randint(-20, 20),
                }
                for i in range(1, 11)
            ],
        }
        
        return node_details
    except Exception as e:
        logger.error(f"Error fetching node details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{training_id}/aggregation")
async def get_aggregation_info(
    training_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get federated aggregation process information"""
    try:
        import random
        
        num_nodes = random.randint(3, 5)
        
        return {
            "training_id": training_id,
            "aggregation_method": "FedAvg",
            "current_round": random.randint(5, 10),
            "nodes_contributing": num_nodes,
            "aggregation_status": "completed",
            "node_contributions": [
                {
                    "node_id": f"node-{i+1:03d}",
                    "weight": random.uniform(0.25, 0.35),
                    "model_size_mb": 12.4,
                    "gradient_norm": random.uniform(0.018, 0.025),
                }
                for i in range(num_nodes)
            ],
            "privacy_metrics": {
                "epsilon_spent": random.uniform(0.3, 0.6),
                "delta": 1e-5,
                "noise_scale": 0.01,
            },
            "security_metrics": {
                "byzantine_detected": 1 if num_nodes >= 4 else 0,
                "nodes_rejected": ["node-004"] if num_nodes >= 4 else [],
                "aggregation_method_used": "Multi-Krum",
            },
        }
    except Exception as e:
        logger.error(f"Error fetching aggregation info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MODEL EXPORT ENDPOINTS ====================

@app.get("/api/training/{training_id}/export/model")
async def export_model(
    training_id: str,
    format: str = "h5",
    current_user: User = Depends(get_current_user)
):
    """Export trained model in specified format"""
    try:
        from fastapi.responses import StreamingResponse
        import io
        
        # Mock model file content
        model_content = f"Mock {format.upper()} model file for training {training_id}\n"
        model_content += "This is a placeholder. Replace with actual model export logic.\n"
        
        # Create file-like object
        file_obj = io.BytesIO(model_content.encode())
        
        # Set appropriate content type
        content_types = {
            "h5": "application/x-hdf5",
            "onnx": "application/octet-stream",
            "tflite": "application/octet-stream",
            "pytorch": "application/octet-stream"
        }
        
        return StreamingResponse(
            file_obj,
            media_type=content_types.get(format, "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename=qflare_model_{training_id}.{format}"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{training_id}/export/report")
async def export_report(
    training_id: str,
    format: str = "pdf",
    current_user: User = Depends(get_current_user)
):
    """Export training report in specified format"""
    try:
        from fastapi.responses import StreamingResponse
        import io
        import json
        
        # Generate mock report data
        report_data = {
            "training_id": training_id,
            "model_type": "CNN",
            "dataset": "MNIST",
            "total_rounds": 10,
            "final_accuracy": 0.95,
            "final_loss": 0.15,
            "nodes_participated": 3,
            "epsilon_used": 0.1,
            "byzantine_detected": 1,
            "timestamp": datetime.now().isoformat()
        }
        
        if format == "json":
            # JSON format
            content = json.dumps(report_data, indent=2).encode()
            media_type = "application/json"
            filename = f"qflare_report_{training_id}.json"
        elif format == "csv":
            # CSV format
            content = "Metric,Value\n"
            for key, value in report_data.items():
                content += f"{key},{value}\n"
            content = content.encode()
            media_type = "text/csv"
            filename = f"qflare_report_{training_id}.csv"
        else:  # pdf - Generate proper PDF using reportlab
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            
            # Create PDF in memory
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1976d2'),
                spaceAfter=30
            )
            title = Paragraph("🔐 QFLARE Training Report", title_style)
            elements.append(title)
            
            # Timestamp
            timestamp_style = ParagraphStyle(
                'Timestamp',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.grey,
                spaceAfter=20
            )
            timestamp = Paragraph(f"Generated: {report_data['timestamp']}", timestamp_style)
            elements.append(timestamp)
            elements.append(Spacer(1, 0.3*inch))
            
            # Training Summary Header
            summary_header = Paragraph("Training Summary", styles['Heading2'])
            elements.append(summary_header)
            elements.append(Spacer(1, 0.2*inch))
            
            # Create data table
            data = [
                ['Metric', 'Value'],
                ['Training ID', report_data['training_id']],
                ['Model Type', report_data['model_type']],
                ['Dataset', report_data['dataset']],
                ['Total Rounds', str(report_data['total_rounds'])],
                ['Final Accuracy', f"{report_data['final_accuracy']:.2%}"],
                ['Final Loss', f"{report_data['final_loss']:.4f}"],
                ['Nodes Participated', str(report_data['nodes_participated'])],
                ['Privacy Budget (ε)', str(report_data['epsilon_used'])],
                ['Byzantine Attacks Detected', str(report_data['byzantine_detected'])]
            ]
            
            # Create table with styling
            table = Table(data, colWidths=[3*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')])
            ]))
            elements.append(table)
            
            # Build PDF
            doc.build(elements)
            buffer.seek(0)
            
            content = buffer.read()
            media_type = "application/pdf"
            filename = f"qflare_report_{training_id}.pdf"
        
        file_obj = io.BytesIO(content)
        
        return StreamingResponse(
            file_obj,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{training_id}/export/metrics")
async def export_metrics(
    training_id: str,
    current_user: User = Depends(get_current_user)
):
    """Export raw metrics data as JSON"""
    try:
        import random
        
        # Generate mock metrics
        metrics = {
            "training_id": training_id,
            "rounds": [
                {
                    "round": i,
                    "accuracy": 0.5 + (i * 0.045),
                    "loss": 2.0 - (i * 0.18),
                    "epsilon": i * 0.01,
                    "timestamp": datetime.now().isoformat()
                }
                for i in range(1, 11)
            ],
            "nodes": [
                {
                    "node_id": f"node-{i:03d}",
                    "accuracy": random.uniform(0.85, 0.95),
                    "loss": random.uniform(0.15, 0.35)
                }
                for i in range(1, 4)
            ]
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{training_id}/export/logs")
async def export_logs(
    training_id: str,
    current_user: User = Depends(get_current_user)
):
    """Export training logs as text file"""
    try:
        from fastapi.responses import StreamingResponse
        import io
        
        # Generate mock logs
        logs = f"QFLARE Training Logs\n"
        logs += f"Training ID: {training_id}\n"
        logs += f"Timestamp: {datetime.now().isoformat()}\n"
        logs += "=" * 50 + "\n\n"
        
        for i in range(1, 11):
            logs += f"[Round {i}] Started training\n"
            logs += f"[Round {i}] Accuracy: {0.5 + (i * 0.045):.4f}\n"
            logs += f"[Round {i}] Loss: {2.0 - (i * 0.18):.4f}\n"
            logs += f"[Round {i}] Completed\n\n"
        
        file_obj = io.BytesIO(logs.encode())
        
        return StreamingResponse(
            file_obj,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=qflare_logs_{training_id}.txt"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== END MODEL EXPORT ENDPOINTS ====================

# ==================== END FL VISUALIZATION ENDPOINTS ====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")