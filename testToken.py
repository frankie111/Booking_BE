import asyncio
import json
from datetime import datetime

import requests

import firebase
from firebase import verify_token
from routes.bookings import delete_all_bookings, get_status_for_locations, check_for_gaps
from routes.users import get_user_data_by_id
from utils.time_utils import convert_to_datetime

token = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjgwNzhkMGViNzdhMjdlNGUxMGMzMTFmZTcxZDgwM2I5MmY3NjYwZGYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vYmV0ZXN0LWY0MTg0IiwiYXVkIjoiYmV0ZXN0LWY0MTg0IiwiYXV0aF90aW1lIjoxNzEyMzAzMjA5LCJ1c2VyX2lkIjoiNk5NMEJuc05Ud2NoWW9uN1hDTVAwZVRhZ0oyMiIsInN1YiI6IjZOTTBCbnNOVHdjaFlvbjdYQ01QMGVUYWdKMjIiLCJpYXQiOjE3MTIzMDMyMDksImV4cCI6MTcxMjMwNjgwOSwiZW1haWwiOiJkZW1vLnVzZXJAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7ImVtYWlsIjpbImRlbW8udXNlckBnbWFpbC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJwYXNzd29yZCJ9fQ.IF5tQSX_eDfmkKvZrTWmjorUaUNaAF4IKfB4EhYjI9sRHXyZxR7HTq8TlmKRZRkJ3ODXdr5-I4_NvtijZjSp6gQXjJT2pke3uMrnIOshHInl11ITfsAoSnDCmJ-abBSZO1thU_WX3RKHsebfkdPuRgbe4uBC4CLYHolskrvaEYzIUIh6FthA3nB3CI6jKZ1_d4hElkyLLENj2FpssioCRGn4-rpm9vBHA4BwN84uKk2Ayg2kpuqbHucXByXy2UriHfwMKlsybt9JOmokWgiPTaCOuMzAMOpszNxe7bNE0y85eF09921hfqrvybid-zk4wJA8Z5TiDdUkK6ZjLOKgMQ"


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

# firebase.setup()
# asyncio.run(verify_token(token))
