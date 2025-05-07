from fastapi import FastAPI
from .routers import candidate
from .database import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(candidate.router)
