from fastapi import FastAPI,UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from modeling.gpr import main
import io

# Initialize FastAPI app
app = FastAPI()


origins = [
    "http://127.0.0.1:5500",
"http://localhost:8000"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Origins allowed to access the API
    allow_credentials=True,  # Allow cookies to be sent
    allow_methods=["*"],  # HTTP methods allowed (GET, POST, etc.)
    allow_headers=["*"],  # Headers allowed
)
@app.post("/upload_train_data/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()  # This will read the file's content as bytes
    rmse,r2 = main(io.BytesIO(contents))
    
    return JSONResponse(content={
        "rmse": rmse,
        "r2": r2 })


