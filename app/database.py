from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
import urllib.parse

# Load environment variables
load_dotenv()

# MongoDB connection settings
# MONGODB_URL = os.getenv("MONGODB_URL")
password = urllib.parse.quote_plus("sp@mongo2025")
MONGODB_URL = f"mongodb+srv://shanil:{password}@cluster0.jjqpodf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Async client for async operations
client = AsyncIOMotorClient(MONGODB_URL)
db = client[DATABASE_NAME]

async def get_database():
    return db

async def initialize_database():
    try:
        # Test the connection
        await client.admin.command('ping')
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    await initialize_database()
    print("Connected to MongoDB!")
    yield
    # Shutdown code
    client.close()
    print("Disconnected from MongoDB!") 