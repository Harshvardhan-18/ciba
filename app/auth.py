"""
`get_current_user` FastAPI dependency.

NextAuth.js (Google provider, JWT session strategy, no DB adapter) signs JWTs
with NEXTAUTH_SECRET using HS256. The payload shape is:

    {
        "sub":     "<google_sub>",   # Google's stable user ID
        "email":   "user@example.com",
        "name":    "Alice",
        "picture": "https://...",    # Google profile photo URL
        "iat":     <unix timestamp>,
        "exp":     <unix timestamp>,
    }

FastAPI verifies the signature, then upserts the User row keyed on google_sub
(creates on first login, updates name/avatar_url on every subsequent request
so profile changes propagate automatically).

The upserted User ORM object is returned and injected into route handlers via
Depends(get_current_user).
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import User

_bearer = HTTPBearer()

ALGORITHM = "HS256"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Verify the NextAuth JWT from the Authorization: Bearer <token> header
    and return (or create) the User row for the authenticated identity.
    Raises HTTP 401 on any verification failure.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.NEXTAUTH_SECRET,
            algorithms=[ALGORITHM],
            options={"verify_aud": False},  # NextAuth doesn't set aud by default
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    google_sub: str | None = payload.get("sub")
    email: str | None = payload.get("email")
    if not google_sub or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims (sub, email)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Upsert: look up by google_sub (stable), update profile fields each login.
    result = await db.execute(select(User).where(User.google_sub == google_sub))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            google_sub=google_sub,
            email=email,
            name=payload.get("name"),
            avatar_url=payload.get("picture"),
        )
        db.add(user)
        await db.flush()  # get the UUID assigned without closing the session
    else:
        # Propagate profile changes (name/avatar) on each login.
        user.name = payload.get("name", user.name)
        user.avatar_url = payload.get("picture", user.avatar_url)

    return user
