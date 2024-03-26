from typing import Optional

import firebase_admin
from fastapi import Header, Depends, HTTPException, status, Request, Security
from fastapi.security import APIKeyHeader
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


api_key_header_auth = APIKeyHeader(name="Authorization", scheme_name="Authorization",
                                   description="Firebase Access Token")


async def verify_token(authorization: str = Security(api_key_header_auth)):
    if not authorization:
        raise HTTPException(status_code=403, detail="Authorization token is missing")

    try:
        user = get_auth().verify_id_token(authorization)
        return User(uid=user["user_id"], email=user["email"])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e}")
