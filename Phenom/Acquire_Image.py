# Test for Phenom API
from fastapi import FastAPI
from ParticleImageSegmentationLibrary import create_timestamped_folder
from ParticleImageSegmentationLibrary import AcquireSEMImage_at_current_location
from getSpotSpectrum import getSpotSpectrum
import os
import PyPhenom as ppi
import license
import numpy as np

base_path = r"C:\PhenomData\FastAPI Tests\Test1"
filename = 'image'
image_side_length_in_pixels = 256

app = FastAPI()

# Acquire SEM Image
@app.get("/acquire_image")
async def acquire_image(base_path: str, image_side_length_in_pixels: int, filename: str):
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
@app.get("/acquire_EDS_spot_spectrum")
async def acquire_EDS_spot_spectrum(i: float, j: float, width: int, height: int, SavePath: str, MainPath: str,  sample_name: str, dwell_time: int):
    phenom = ppi.Phenom(license.PhenomAddress, license.PhenomUsername, license.PhenomPassword)
    dpp = phenom.Spectrometer
    settings = ppi.LoadEidSettings()
    dpp.ApplySettings(settings.spot)
    phenom.SetSemSpotSize(4.1)
    address = license.PhenomAddress
    size = (width, height)
    spectrum = getSpotSpectrum(
        i,j,
        size,
        SavePath,
        MainPath,
        dpp,
        address,
        phenom,
        sample_name,
        dwell_time)
    print('acquired spectrunm')
    print(type(spectrum))
    print(f'SPectrum acquired with getspotspectrum has dimensions {np.shape(spectrum)}')
    spectrum_list = spectrum.tolist()   # Python list
    print(len(spectrum_list))
    return {"spectrum": spectrum_list}