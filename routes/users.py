from fastapi import APIRouter
from pydantic import BaseModel

from firebase import get_auth, get_client

users = APIRouter()


def register_user(email, password):
    try:
        user = get_auth().create_user(email=email, password=password)
        print('Successfully created new user: {0}'.format(user.uid))
        return user
    except Exception as e:
        print('Error creating new user:', e)


def save_user_data(uid, email, name):
    try:
        db = get_client()
        user_ref = db.collection('users').document(uid)
        user_ref.set({
            'email': email,
            'name': name,
            'is_admin': False
        })
        print('Successfully saved user data to Firestore')
    except Exception as e:
        print('Error saving user data:', e)


def delete_user(uid):
    try:
        # Delete user from Firebase Authentication
        get_auth().delete_user(uid)
        print('Successfully deleted user from Firebase Authentication')

        # Delete user data from Firestore
        db = get_client()
        user_ref = db.collection('users').document(uid)
        user_ref.delete()
        print('Successfully deleted user data from Firestore')
    except Exception as e:
        print('Error deleting user:', e)


class AddUserModelResponse(BaseModel):
    message: str


class AddUserModelRequest(BaseModel):
    email: str
    name: str
    password: str


@users.post("/user/", response_model=AddUserModelResponse)
async def add_user(
        req: AddUserModelRequest
):
    user = register_user(req.email, req.password)
    save_user_data(user.uid, req.email, req.name)

    return AddUserModelResponse(message=f"Added user [{user.uid}] - {req.name}")

# name = "mircox"
# email = "mircea.rautoiu@gmail.com"
# password = "muie1234"
#
# user = register_user(email, password)
# save_user_data(user.uid, email, name)
# delete_user("R9BBVkJ6hBUyUGIRjeVTOrozIMW2")
