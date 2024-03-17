from datetime import datetime
from typing import Optional, Dict
from datetime import date, datetime

from pydantic import BaseModel


class Location(BaseModel):
    id: str
    capacity: int
    type: str


class Booking(BaseModel):
    uid: str
    loc_id: str  # location (table/conference room)
    start: datetime
    end: datetime
    nr_attend: int


class User(BaseModel):
    uid: str
    name: Optional[str] = None
    email: str
    is_admin: bool = False


# class RoomAvailabilityRequest(BaseModel):
#     date: date
#     room_name: str


class RoomAvailabilityRequest(BaseModel):
    date: date
    room_name: str

class RoomAvailabilityResponse(BaseModel):
    date: str
    room_name: str
    availability: Dict[str, str]
