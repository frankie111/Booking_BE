import firebase_admin
from fastapi import Header, Depends, HTTPException, status
from firebase_admin import auth
from firebase_admin import credentials
from firebase_admin import firestore

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


def get_current_user(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header is missing")

    token = authorization.split(" ")[1]
    try:
        # Verify the token using Firebase Admin SDK
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid authentication token")
