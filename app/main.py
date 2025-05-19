from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

from app.api.routes import router
from app.config import UPLOAD_DIR

# Create FastAPI app
app = FastAPI(
    title="Document Research & Theme Identification Chatbot",
    description="An interactive chatbot that can perform research across a large set of documents, identify common themes, and provide detailed, cited responses to user queries.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve uploaded files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return html_content


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8090))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)