"""Main FastAPI application entry point"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, predict, gemini

app = FastAPI(
    title="AI Visa Prediction API",
    description="Machine Learning powered visa approval prediction and processing time estimation",
    version="1.0.0"
)

# CORS - allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development across multiple ports
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(gemini.router)

@app.get("/")
async def root():
    return {
        "message": "AI Visa Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "api": "Vi-SaaS Prediction Platform"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
