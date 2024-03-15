from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Table(BaseModel):
    id: Optional[str]


class MeetingRoom(BaseModel):
    id: Optional[str]
    capacity: Optional[int]


class Booking(BaseModel):
    uid: Optional[str]
    loc_id: Optional[str]  # location (table/conference room)
    start: Optional[datetime]
    end: Optional[datetime]
    nr_attend: Optional[int]


class User(BaseModel):
    uid: Optional[str]
    name: Optional[str]
    email: Optional[str]
    is_admin: Optional[bool]
