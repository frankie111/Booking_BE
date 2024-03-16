import csv
from datetime import datetime

from fastapi import APIRouter, HTTPException
from google.cloud.firestore_v1 import FieldFilter
from pydantic import BaseModel

from firebase import get_firestore_db

bookings = APIRouter()


def add_locations_from_file():
    db = get_firestore_db()
    loc_ref = db.collection("locations")
    with open("locations.csv", newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            name, cap = row

            type_ = "table" if "CLUJ" in name else "room"
            loc_ref.document(name).set(
                {
                    "capacity": int(cap),
                    "type": type_
                }
            )


class AddBookingResponse(BaseModel):
    message: str
    booking: dict


class AddBookingRequest(BaseModel):
    uid: str
    loc_id: str
    start: datetime
    end: datetime
    nr_attend: int


@bookings.post(
    "/bookings/",
    tags=["Bookings"],
    response_model=AddBookingResponse,
    description="Add a new booking. Checks for overlapping bookings for the location at the given time"
)
def add_booking(
        req: AddBookingRequest
):
    try:
        db = get_firestore_db()
        loc_ref = db.collection("locations").document(req.loc_id)
        loc_dict = loc_ref.get()

        if not loc_dict.exists:
            raise HTTPException(status_code=404, detail="Location does not exist.")

        loc_dict = loc_dict.to_dict()
        _type = loc_dict["type"]
        cap = loc_dict["capacity"]

        # Check if number of attendees is valid
        if req.nr_attend not in range(1, cap + 1):
            raise HTTPException(status_code=400, detail=f"Invalid number of attendees.")

        # Check if meeting room is less than half-booked
        if _type == "room" and req.nr_attend < cap / 2:
            raise HTTPException(status_code=400, detail="Meeting room cannot be less than half-booked.")

        overlapping_bookings = get_overlapping_bookings(req.loc_id, req.start, req.end)
        if len(overlapping_bookings) > 0:
            raise HTTPException(status_code=400,
                                detail=f"Requested booking overlaps with {len(overlapping_bookings)} existing booking(s).")

        book_ref = loc_ref.collection("bookings").document()
        new_book = {
            "uid": req.uid,
            "start": req.start,
            "end": req.end,
            "nr_attend": req.nr_attend
        }
        book_ref.set(
            new_book
        )

        return AddBookingResponse(message=f"Added booking {book_ref.id}", booking=new_book)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_overlapping_bookings(loc_id: str, start: datetime, end: datetime):
    db = get_firestore_db()
    loc_ref = db.collection("locations").document(loc_id)

    # Query bookings that end after the current booking should start
    query = loc_ref.collection("bookings").where(filter=FieldFilter("end", ">", start)).stream()

    # Retrieve the active bookings
    active_bookings = [doc.to_dict() for doc in query]

    # Check for overlapping bookings
    overlapping_bookings = []
    for booking in active_bookings:
        booking_start = booking['start'] = booking['start'].astimezone()
        booking_end = booking['end'] = booking['end'].astimezone()
        if start < booking_end and end > booking_start:
            overlapping_bookings.append(booking)

    return overlapping_bookings


def get_all_active_bookings_for_location(loc_id: str):
    db = get_firestore_db()
    loc_ref = db.collection("locations").document(loc_id)
    loc_dict = loc_ref.get()

    if not loc_dict.exists:
        raise HTTPException(status_code=404, detail="Location does not exist.")

    # Query bookings that end after the current booking should start
    current_time = datetime.now()
    query = loc_ref.collection("bookings").where(filter=FieldFilter("end", ">", current_time)).stream()

    # Retrieve the active bookings with their document IDs
    active_bookings = []
    for doc in query:
        booking_data = doc.to_dict()
        booking_id = doc.id
        booking_data['id'] = booking_id
        active_bookings.append(booking_data)

    return active_bookings


class ActiveBookingsResponse(BaseModel):
    active_bookings: list[dict]


@bookings.get(
    "/bookings/{loc_id}/active",
    tags=["Bookings"],
    response_model=ActiveBookingsResponse,
    description="Get all active bookings for the specified location"
)
async def get_active_bookings_for_location(loc_id: str):
    active_bookings = get_all_active_bookings_for_location(loc_id)
    return ActiveBookingsResponse(active_bookings=active_bookings)

# 2024-03-16T09:00:00+0200
# start = convert_to_datetime("16-03-2024 11:00:00")
# end = convert_to_datetime("16-03-2024 13:00:00")
# add_booking(
#     "4FcS6HDIqPOG4hIeFrg5DO9Znwc2",
#     "Cockpit",
#     start,
#     end,
#     5
# )

# start = convert_to_datetime("16-03-2024 8:00:00")
# end = convert_to_datetime("16-03-2024 9:30:00")
# o_bookings = check_location_availability("Cockpit", start, end)
# for booking in o_bookings:
#     print(f"{booking["start"]} - {booking["end"]}")

# active_bookings = get_all_active_bookings_for_location("Cockpit")
# print(active_bookings)
