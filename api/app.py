from fastapi import FastAPI, HTTPException, File, UploadFile

from pydantic import BaseModel
# from modeling.gpr import load_model, predict

# Initialize FastAPI app
app = FastAPI()

@app.post("/upload_train_data/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
