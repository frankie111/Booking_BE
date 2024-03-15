from typing import Optional, List

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import ssl

from routes.users import users

app = FastAPI(
    title="Odin API",
    description="API for Booking",
    version="0.1.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)
app.include_router(users)

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(
    "C:/Windows/System32/cert.pem",
    keyfile="C:/Windows/System32/key.pem"
)

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
