import firebase_admin
from fastapi import Header, Depends, HTTPException, status, Request
from firebase_admin import auth
from firebase_admin import credentials
from firebase_admin import firestore

from models import User

if not firebase_admin._apps:
    cred = credentials.Certificate(r"betest-f4184-firebase-adminsdk-olp9p-e7cc5114bc.json")
    firebase_admin.initialize_app(cred)

    fdb = None


def setup():
    global fdb

    if fdb is not None:
        return

    fdb = firestore.client()


def get_firestore_db():
    setup()
    return fdb


def get_auth():
    return auth


async def verify_token(req: Request):
    jwt = req.headers.get("Authorization")
    if not jwt:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization token is missing")

    if jwt.startswith("Bearer "):
        jwt = jwt[7:]

    try:
        user = get_auth().verify_id_token(jwt)
        return User(uid=user["user_id"], email=user["email"])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid or expired token: {e}")
