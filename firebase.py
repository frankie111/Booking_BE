import firebase_admin
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

