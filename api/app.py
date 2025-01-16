from fastapi import FastAPI, HTTPException, File, UploadFile

from pydantic import BaseModel
from modeling.gpr import main

import io
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

@app.post("/upload_train_data/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()  # This will read the file's content as bytes
    main(io.BytesIO(contents))
    return {"filename": file.filename}
