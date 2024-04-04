import asyncio
import json
from datetime import datetime

import requests

from routes.bookings import delete_all_bookings, get_status_for_locations, check_for_gaps
from routes.users import get_user_data_by_id
from utils.time_utils import convert_to_datetime

token = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImJhNjI1OTZmNTJmNTJlZDQ0MDQ5Mzk2YmU3ZGYzNGQyYzY0ZjQ1M2UiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vYmV0ZXN0LWY0MTg0IiwiYXVkIjoiYmV0ZXN0LWY0MTg0IiwiYXV0aF90aW1lIjoxNzExNDU0ODgxLCJ1c2VyX2lkIjoiU2VxTE4wVDJQS05oNHJPSUl4YlA3MGxFYzRFMyIsInN1YiI6IlNlcUxOMFQyUEtOaDRyT0lJeGJQNzBsRWM0RTMiLCJpYXQiOjE3MTE0NTQ4ODEsImV4cCI6MTcxMTQ1ODQ4MSwiZW1haWwiOiJkZW1vLnVzZXJAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7ImVtYWlsIjpbImRlbW8udXNlckBnbWFpbC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJwYXNzd29yZCJ9fQ.m3NpSdESBHDDr6biQWZuIzAEWqBWSQ8VTj2MmpZiwUu0lekK7ylIXunDZe1Oq8OXcPE5F7xf2IazbUu4LwSKOIxuexDdBTCODYDqU-a4Cql3BGFMghGSiELd4xtDXe-dlda7xQPScRprucOlHuYzDFiRabM5yiwFUv8gnlBLa3-2JIpGjQm3cSeEA4he911Tu8PwpDA9GzZJitr-mnX9180RoNzrKHgk-wYJigiqAe2M-f4aEY-Ug7khW6_Jxh6pWKs3M-LtB3-yYpeIvD4AkjePwVgcXyErOIx8_QnHr6WH3PjHvtKzf6mSK9QjbmkLwzKW56qEJ-mfNV05gJE99g"


def test_validate_endpoint():
    headers = {
        "authorization": token
    }

    response = requests.post(
        "https://localhost:8000/ping",
        headers=headers,
        verify=False
    )

    return response.text


# print(get_user_data_by_id("4FcS6HDIqPOG4hIeFrg5DO9Znwc2"))
delete_all_bookings()

# start = convert_to_datetime("01-04-2024 09:00:00")
# end = convert_to_datetime("01-04-2024 19:00:00")
#
# response = asyncio.run(get_status_for_locations(start, end))
# statuses = response.statuses
# print(statuses["CLUJ_5_beta_1.2"])
#
