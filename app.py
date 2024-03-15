from typing import Optional, List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

origins = [
    'http://localhost:3000',
    'http://127.0.0.1:3000'
]

# Set up CORS middleware to allow cross-origin requests from the defined origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows access from specified origins
    allow_credentials=True,  # Allows cookies to be included in cross-origin HTTP requests
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Specify which method are allowed
    allow_headers=["X-Requested-With", "Content-Type", "Accept", "Origin", "Authorization"],  # Specific headers allowed
)
