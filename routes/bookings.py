import csv

from fastapi import APIRouter

from firebase import get_firestore_db
from models import Booking, Location

bookings = APIRouter()


def create_booking(booking: Booking):
    db = get_firestore_db()
    loc_ref = db.collection("locations").document(booking.loc_id)
    loc_ref.set(
        {
            "uid": booking.uid,
            "start": booking.start,
            "end": booking.end,
            "nr_attend": booking.nr_attend
        }
    )


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


add_locations_from_file()
