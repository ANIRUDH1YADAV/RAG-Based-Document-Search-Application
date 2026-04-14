"""
MongoDB client initialization.
"""

import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "adaptive_rag")
SERVER_SELECTION_TIMEOUT_MS = int(
	os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "3000")
)

client = AsyncIOMotorClient(
	MONGO_URL,
	serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
)
db = client[DB_NAME]
