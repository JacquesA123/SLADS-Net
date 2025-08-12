# Acquire an EDS spectrum from the PyPhenom environment via an API, even when the current script is running outside that environment
from fastapi import FastAPI
from pydantic import BaseModel
from getSpotSpectrum import getSpotSpectrum
import numpy as np

app = FastAPI()

# Define request body with two floats: x and y
class CoordinatesRequest(BaseModel):
    x: float
    y: float

@app.post("/getspectrum")
async def get_spectrum(req: CoordinatesRequest):
    # Call your acquisition function with the coordinates
    spectrum = getSpotSpectrum(req.x, req.y)
    
    # Convert NumPy array to list for JSON serialization
    return {"spectrum": spectrum.tolist()}