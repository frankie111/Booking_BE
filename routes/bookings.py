import csv
from datetime import datetime

from fastapi import APIRouter, HTTPException

from firebase import get_firestore_db
from utils.utils import convert_to_datetime

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


def create_booking(uid: str, loc_id: str, start: datetime, end: datetime, nr_attend: int):
    db = get_firestore_db()
    loc_ref = db.collection("locations").document(loc_id)
    loc_dict = loc_ref.get()

    if not loc_dict.exists:
        raise HTTPException(status_code=404, detail="Location does not exist.")

    loc_dict = loc_dict.to_dict()
    _type = loc_dict["type"]
    cap = loc_dict["capacity"]

    # Check if number of attendees is valid
    if nr_attend not in range(1, cap + 1):
        raise ValueError(f"Invalid number of attendees.")

    # Check if meeting room is less than half-booked
    if _type == "room" and nr_attend < cap / 2:
        raise ValueError("Meeting room cannot be less than half-booked.")

    check_location_availability(loc_id, start, end)

    book_ref = loc_ref.collection("bookings").document()
    book_ref.set(
        {
            "uid": uid,
            "start": start,
            "end": end,
            "nr_attend": nr_attend
        }
    )


def check_location_availability(loc_id: str, start: datetime, end: datetime):
    db = get_firestore_db()
    loc_ref = db.collection("locations").document(loc_id)
    loc_dict = loc_ref.get()

    if not loc_dict.exists:
        raise HTTPException(status_code=404, detail="Location does not exist.")

    loc_dict = loc_dict.to_dict()
    bookings = loc_ref.collection("bookings").stream()
    for booking in bookings:
        booking_data = booking.to_dict()
        booking_start = booking_data['start'].astimezone()
        booking_end = booking_data['end'].astimezone()

        # Check for overlap
        if start < booking_end and end > booking_start:
            raise HTTPException(status_code=400, detail="This location is already booked at the specified time")


start = convert_to_datetime("16-03-2024 11:00:00")
end = convert_to_datetime("16-03-2024 13:00:00")
# create_booking(
#     "4FcS6HDIqPOG4hIeFrg5DO9Znwc2",
#     "Cockpit",
#     start,
#     end,
#     5
# )

check_location_availability("Cockpit", start, end)
