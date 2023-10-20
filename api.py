
from fastapi import FastAPI, UploadFile, File, HTTPException
from .data_processing import DataProcessing


@app.post("/upload/")
async def upload_api(file: UploadFile = File(...)):
    with open(f"/temp/path/{file.filename}", "wb") as buffer:
        buffer.write(file.file.read())
    data_processing = DataProcessing()
    return data_processing.send(f"/temp/path/{file.filename}")

@app.post("/learn/")
async def learn_api():
    # ... (similar structure to upload_api)

@app.post("/predict/")
async def predict_api(input: UploadFile = File(...), output: str = None):
    # ... (similar structure to upload_api)
    
@app.post("/add-skill/")
async def add_skill_api():
    # ... (similar structure to upload_api)

@app.get("/list-skills/")
def list_skills_api():
    # ... (Endpoint logic)
    
@app.get("/list-runtimes/")
def list_agents_api():
    # ... (Endpoint logic)

@app.get("/metrics/")
def get_metrics_api():
    # ... (Endpoint logic)

@app.get("/info/")
def info_api():
    # ... (Endpoint logic)

@app.get("/logs/")
def logs_api():
    # ... (Endpoint logic)
        
