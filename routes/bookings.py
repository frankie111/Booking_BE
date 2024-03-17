import csv
import time
from datetime import datetime
from typing import Dict

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from google.cloud.firestore_v1 import FieldFilter
from pydantic import BaseModel

from firebase import get_firestore_db
from models import RoomAvailabilityRequest
from utils.utils import convert_to_datetime

bookings = APIRouter()

model = joblib.load("model_architecture/meeting_rooms_model/model_test_fewer_features.pkl")


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
            "nr_attend": req.nr_attend,
            "loc_id": req.loc_id
        }
        book_ref.set(
            new_book
        )
        book_ref.update({"id": book_ref.id})

        return AddBookingResponse(message=f"Added booking {book_ref.id}", booking=new_book)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_overlapping_bookings(loc_id: str, start: datetime, end: datetime):
    db = get_firestore_db()
    loc_ref = db.collection("locations").document(loc_id)

    # Query bookings that end after the current booking should start
    docs = loc_ref.collection("bookings").where(filter=FieldFilter("end", ">", start)).stream()

    start_time = time.time()  # Record the start time
    # Retrieve the active bookings
    active_bookings = [doc.to_dict() for doc in docs]

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time
    print(f"Query Execution time: {execution_time} seconds")

    # Check for overlapping bookings
    overlapping_bookings = []
    for booking in active_bookings:
        booking_start = booking['start'] = booking['start'].astimezone()
        booking_end = booking['end'] = booking['end'].astimezone()
        if start < booking_end and end > booking_start:
            overlapping_bookings.append(booking)

    return overlapping_bookings


def get_overlapping_bookings_for_loc(loc_id: str, bookings: list[dict], start: datetime, end: datetime):
    overlapping_bookings = []
    bookings = [booking for booking in bookings if booking.get('loc_id') == loc_id]
    for booking in bookings:
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
    current_time = datetime.now().astimezone()
    query = loc_ref.collection("bookings").where(filter=FieldFilter("end", ">", current_time)).stream()

    # Retrieve the active bookings with their document IDs
    active_bookings = []
    for doc in query:
        booking_data = doc.to_dict()
        booking_id = doc.id
        booking_data['id'] = booking_id
        booking_data['start'] = booking_data['start'].astimezone()
        booking_data['end'] = booking_data['end'].astimezone()
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


def check_for_gaps(intervals: list[tuple[datetime, datetime]]):
    intervals.sort()  # Sort the intervals based on the start time
    for i in range(1, len(intervals)):
        prev_end = intervals[i - 1][1]  # End time of the previous interval
        current_start = intervals[i][0]  # Start time of the current interval
        if current_start > prev_end:
            return True
    # If no gaps are found, you can return True or perform any other desired action
    return False


class LocationsStatusesResponse(BaseModel):
    statuses: dict[str, str]


@bookings.get(
    "/bookings/",
    tags=["Bookings"],
    response_model=LocationsStatusesResponse,
    description="Get a dict of statuses for all Locations"
)
async def get_status_for_locations(start: datetime, end: datetime):
    start_time = time.time()  # Record the start time
    db = get_firestore_db()
    loc_ref = db.collection("locations")
    locations = loc_ref.stream()

    # Get all active bookings
    query = db.collection_group("bookings").where(filter=FieldFilter("end", ">", start)).stream()
    active_bookings = [booking.to_dict() for booking in query]

    loc_statuses = {}

    for loc in locations:
        loc_id = loc.id
        ov_bookings = get_overlapping_bookings_for_loc(loc_id, active_bookings, start, end)
        covered_intervals = []
        for booking in ov_bookings:
            booking_start = booking['start'].astimezone()
            booking_end = booking['end'].astimezone()
            covered_intervals.append((booking_start, booking_end))

        if len(covered_intervals) == 0:
            status = "FREE"
        else:
            if check_for_gaps(covered_intervals):
                status = "PARTIALLY BOOKED"
            else:
                min_start = min(covered_intervals, key=lambda x: x[0])[0]
                max_end = max(covered_intervals, key=lambda x: x[1])[1]

                # Check if loc is fully booked in the specified time interval
                if min_start <= start and max_end >= end:
                    status = "FULLY BOOKED"
                else:
                    status = "PARTIALLY BOOKED"

        loc_statuses[loc_id] = status

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return LocationsStatusesResponse(statuses=loc_statuses)


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

# start = convert_to_datetime("16-03-2024 5:00:00")
# end = convert_to_datetime("16-03-2024 18:00:00")

# o_bookings = check_location_availability("Cockpit", start, end)
# for booking in o_bookings:
#     print(f"{booking["start"]} - {booking["end"]}")

# active_bookings = get_all_active_bookings_for_location("Cockpit")
# print(active_bookings)

# start_time = time.time()  # Record the start time

# db = get_firestore_db()
# loc_ref = (db.collection("locations").
#            document("Cockpit").
#            collection("bookings").
#            document("w0zI208eCYO6hTELabHF"))
# doc = loc_ref.get()
# end_time = time.time()  # Record the end time
# execution_time = end_time - start_time
# print(f"Query Execution time: {execution_time} seconds")

# get_status_for_locations(start, end)

room_mapping = {
    'Pit-Lane': 0,
    'Dry-lane': 1,
    'Joker Lap': 2,
    'Quick 8': 3,
    'Pole Position': 4,
    'Cockpit': 5,
}
# def prepare_input_data(request_date, room_name):
#     # Extract features from date
#     date_parsed = pd.to_datetime(request_date)
#     day_of_week = date_parsed.dayofweek
#     month = date_parsed.month
#     week_of_year = date_parsed.isocalendar()[1]
#
#     # Map room_name to its code
#     if room_name in room_mapping:
#         room_code = room_mapping[room_name]
#     else:
#         raise ValueError("Room name not recognized.")
#
#     # Prepare input data in the same order as during training
#     input_data = [day_of_week, month, week_of_year, room_code]
#     return input_data
#
# @bookings.post("/predict-availability/", response_model=Dict[str, str])
# def predict_availability(request: RoomAvailabilityRequest):
#     try:
#         # Assuming you've implemented prepare_input_data function
#         input_data = prepare_input_data(request.date, request.room_name)
#         prediction = model.predict([input_data])[0]
#         availability = "Available" if prediction == 1 else "Not Available"
#         return {"availability": availability}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# def prepare_input_data(request_date, room_name, interval):
#     date_parsed = pd.to_datetime(request_date)
#     day_of_week = date_parsed.dayofweek
#     month = date_parsed.month
#     week_of_year = date_parsed.isocalendar()[1]
#
#     if room_name in room_mapping:
#         room_code = room_mapping[room_name]
#     else:
#         raise ValueError("Room name not recognized.")
#
#     # Assuming interval is directly usable by the model (ensure this matches your model's training)
#     input_data = [day_of_week, month, week_of_year, room_code, interval]
#     return input_data

def prepare_input_data(request_date, room_name):
    date_parsed = pd.to_datetime(request_date)
    day_of_week = date_parsed.dayofweek
    month = date_parsed.month
    week_of_year = date_parsed.isocalendar()[1]

    if room_name in room_mapping:
        room_code = room_mapping[room_name]
    else:
        raise ValueError("Room name not recognized.")

    # Prepare input data excluding interval since we are now predicting for all intervals
    input_data = [day_of_week, month, week_of_year, room_code]
    return input_data



# @bookings.post("/predict-availability/", response_model=Dict[str, str])
# def predict_availability(request: RoomAvailabilityRequest):
#     try:
#         # Update to include interval in the preparation
#         input_data = prepare_input_data(request.date, request.room_name, request.interval)
#         # Prediction now returns predictions for each interval
#         predictions = model.predict([input_data])[0]
#         availability_statuses = ['Not Available' if pred == 0 else 'Available' for pred in predictions]
#
#         # Map interval names to their predictions for response
#         interval_names = ['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']
#         availability = dict(zip(interval_names, availability_statuses))
#
#         return {"date": request.date, "room_name": request.room_name, "availability": availability}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @bookings.post("/predict-availability/", response_model=Dict[str, Dict[str, str]])
# def predict_availability(request: RoomAvailabilityRequest):
#     try:
#         # Generate features from the request data without needing the interval
#         input_data = prepare_input_data(request.date, request.room_name)
#
#         # Predict using the model which now predicts for all intervals
#         predictions = model.predict([input_data])
#
#         # Make sure predictions are in a suitable format for iteration
#         if predictions.ndim == 1:
#             predictions = [predictions]  # Wrap in another list if single dimension
#
#         interval_names = ['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']
#         availability_statuses = []
#
#         # Assuming the model correctly predicts multiple outputs for the input
#         for interval_index, pred in enumerate(predictions[0]):
#             status = "Available" if pred == 1 else "Not Available"
#             availability_statuses.append(status)
#
#         availability = dict(zip(interval_names, availability_statuses))
#
#         return {"date": request.date.strftime("%Y-%m-%d"), "room_name": request.room_name, "availability": availability}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# @bookings.post("/predict-availability/", response_model=RoomAvailabilityRequest)
# def predict_availability(request: RoomAvailabilityRequest):
#     try:
#         # Generate features from the request data without needing the interval
#         input_data = prepare_input_data(request.date, request.room_name)
#
#         # Predict using the model which now predicts for all intervals
#         predictions = model.predict([input_data])
#
#         # Assuming predictions are in the correct format (an array of predictions per interval)
#         if predictions.ndim == 1:
#             predictions = [predictions]  # Ensure it's a list of lists for consistency
#
#         interval_names = ['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']
#         availability_statuses = ["Available" if pred == 1 else "Not Available" for pred in predictions[0]]
#
#         availability = dict(zip(interval_names, availability_statuses))
#
#         return {
#             "date": request.date.strftime("%Y-%m-%d"),
#             "room_name": request.room_name,
#             "availability": availability
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@bookings.post("/predict-availability/", response_model=RoomAvailabilityRequest)
def predict_availability(request: RoomAvailabilityRequest):
    try:
        # Generate features from the request data without needing the interval
        input_data = prepare_input_data(request.date, request.room_name)

        # Predict using the model which now predicts for all intervals
        predictions = model.predict([input_data])

        # Assuming predictions are in the correct format (an array of predictions per interval)
        if predictions.ndim == 1:
            predictions = [predictions]  # Ensure it's a list of lists for consistency

        interval_names = ['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']
        availability_statuses = ["Available" if pred == 1 else "Not Available" for pred in predictions[0]]

        availability = dict(zip(interval_names, availability_statuses))

        return {
            "date": request.date.strftime("%Y-%m-%d"),
            "room_name": request.room_name,
            "availability": availability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

