import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import auth

cred = credentials.Certificate(r"betest-f4184-firebase-adminsdk-olp9p-e7cc5114bc.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


def get_client():
    return db


def get_auth():
    return auth

# List all users
# page = auth.list_users()
# while page:
#     for user in page.users:
#         print(f"{user.uid}, {user.email}, {user.user_metadata.creation_timestamp}")
#     page = page.get_next_page()
