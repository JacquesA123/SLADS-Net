# Test for Phenom API
from fastapi import FastAPI
from ParticleImageSegmentationLibrary import create_timestamped_folder
from ParticleImageSegmentationLibrary import AcquireSEMImage_at_current_location
from getSpotSpectrum import getSpotSpectrum
import os

base_path = r"C:\PhenomData\FastAPI Tests\Test1"
filename = 'image'
image_side_length_in_pixels = 256

app = FastAPI()


@app.get("/")

# Acquire SEM Image
async def acquire_image():
    # Create a timestamped folder for this run
    run_folder = create_timestamped_folder(base_path, prefix="Run")
    
    # Acquire the image in that folder
    full_image_path = os.path.join(run_folder, f"{filename}.png")  # or whatever extension your function uses
    AcquireSEMImage_at_current_location(run_folder, filename, image_side_length_in_pixels)
    
    return {
        "message": "Image acquired successfully",
        "folder": run_folder,
        "file": full_image_path
    }

# Acquire EDS Spot Spectrum
async def acquire_EDS_spot_measurement():
    getSpotSpectrum(
        i,j,
        size,
        SavePath,
        MainPath, 
        image_path,
        dpp,
        settings,
        address,
        phenom,
        ppi,
        sample_name,
        dwell_time)