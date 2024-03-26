from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel

from firebase import get_auth, get_firestore_db, verify_token
from models import User

users = APIRouter()


def register_user(email, password):
    try:
        user = get_auth().create_user(email=email, password=password)
        print('Successfully created new user: {0}'.format(user.uid))
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error creating new user: {e}')


def save_user_data(user: User):
    try:
        db = get_firestore_db()
        user_ref = db.collection('users').document(user.uid)
        user_ref.set({
            'email': user.email,
            'name': user.name,
            'is_admin': user.is_admin
        })
        print('Successfully saved user data to Firestore')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error saving user data {e}')


class AddUserModelResponse(BaseModel):
    message: str


@users.post(
    "/user/",
    tags=["Users"],
    response_model=User,
    description="Register a new user"
)
async def add_user(
        req: Request, user: User = Depends(verify_token)
):
    save_user_data(user)
    return AddUserModelResponse(message=f"Added user {user.uid}")


class DeleteUserModelResponse(BaseModel):
    message: str


@users.delete(
    "/user/",
    tags=["Users"],
    response_model=DeleteUserModelResponse,
    description="Delete a user by id"
)
def delete_user(uid):
    try:
        # Delete user from Firebase Authentication
        get_auth().delete_user(uid)
        print('Successfully deleted user from Firebase Authentication')

        # Delete user data from Firestore
        db = get_firestore_db()
        user_ref = db.collection('users').document(uid)
        user_ref.delete()

        return DeleteUserModelResponse(message=f"User {uid} deleted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error deleting user: {e}')
