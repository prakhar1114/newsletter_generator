from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from summariser.vectordb_client import init_vector_db
from summariser.urls.admin import admin_router # We will create this next

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    """
    # 1. Initialize Qdrant Collection
    print("Checking Vector DB collections...")
    init_vector_db()

    # 2. You could also initialize SQL DBs or LLM clients here

    yield
    # Shutdown logic goes here (e.g., closing DB connections)

def create_app() -> FastAPI:
    app = FastAPI(
        title="Summariser AI API",
        version="0.1.0",
        lifespan=lifespan
    )

    # Standard Middleware (for frontend/different origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include different URL groups (Routers)
    app.include_router(admin_router, prefix="/admin")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "database": "connected"}

    return app

app = create_app()
