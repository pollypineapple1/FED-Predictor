import sys
import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from interface.main import predict

# # Add the parent directory (FED-Predictor) to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()

# first decorated function: welcome
@app.get("/")
def root():
    return {"message": "Welcome to FED-predictor API"}

# second decorated function: predict
@app.post("/predict")
async def predict_class(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No upload file sent")
    
    try:
        # Directly pass the file to the predict function
        result = predict(file.file)  # file.file is a file-like object
        
        return {"prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))