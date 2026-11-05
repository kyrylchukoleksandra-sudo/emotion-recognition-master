from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.db.models import User, UserRole
from backend.app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Replace with your Google OAuth Client ID
GOOGLE_CLIENT_ID = "83372674253-grhpkeinbp9pq0vd4k94bcbf5r043ai1.apps.googleusercontent.com"

class GoogleAuthRequest(BaseModel):
    credential: str  # ID token received from Google Frontend SDK

@router.post("/google")
def google_auth(payload: GoogleAuthRequest, db: Session = Depends(get_db)):
    try:
        id_info = id_token.verify_oauth2_token(
            payload.credential, requests.Request(), GOOGLE_CLIENT_ID
        )
        
        email = id_info.get("email")
        google_id = id_info.get("sub")
        name = id_info.get("name")
        picture = id_info.get("picture")

        if not email:
            raise HTTPException(status_code=400, detail="Invalid Google token payload")

        # Check if user exists or create new
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(
                email=email,
                full_name=name,
                picture_url=picture,
                google_id=google_id,
                role=UserRole.USER,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        access_token = create_access_token(data={"sub": user.email, "role": user.role.value, "id": user.id})

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.full_name,
                "picture": user.picture_url,
                "role": user.role.value,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Google Authentication failed: {str(e)}",
        )