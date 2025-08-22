
import requests
from datetime import datetime
import os
import numpy as np

def perform_single_EDS_measurement(coordinate):
    # COnfiguration
    
    width = 256
    height = 256
    SavePath = r"C:\PhenomData\FastAPI Tests\AcquiringSpotSpectrum"
    MainPath = "/"
    sample_name = 'FastAPI_test_sample'
    dwell_time = 7

    # Make timestamped folder
    # Get the current time as a string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    prefix = 'Run'
    # Create the folder name (e.g., "Run_2025-08-07_14-38-00")
    folder_name = f"{prefix}_{timestamp}"

    # Full path
    full_path = os.path.join(SavePath, folder_name)

    # Create the folder
    os.makedirs(full_path, exist_ok=True)

    i, j = coordinate
    i, j = i / 256 , j / 256
    print(f'i = {i}')
    print(f'j = {j}')
    url = "http://127.0.0.1:8000/acquire_EDS_spot_spectrum"
    params = {
        "i": i,
        "j": j,
        "width": width,
        "height": height,
        "SavePath": full_path,
        "MainPath": MainPath,
        "sample_name": sample_name,
        "dwell_time": dwell_time
    }

    response = requests.get(url, params=params)
    jsoned_response = response.json()
    print(f'Response type is {type(response)}')
    print(f'Once converted to json, it is of type {type(jsoned_response)}')
    print(f'The length of the dictionary is {len(jsoned_response)}')
    print(type(jsoned_response))        # Should be <class 'dict'>
    print(len(jsoned_response))         # Should be 1
    print(jsoned_response.keys())       # Shows the dictionary keys
    print(list(jsoned_response.keys())) # Convert to list if you want indexing
    for key, value in jsoned_response.items():
        print(type(value))
        print(f'Length of the spectrum list is {len(value)}')
        # print(value)
    spectrum = np.array(response.json()["spectrum"])
    print(f'Spectrum acquired through the url function has shape {np.shape(spectrum)}')

    # print(response.status_code)     # HTTP code
    # print(response.headers)         # Headers
    # print(response.text)            # Raw text
    # print(response.json())          # Parsed JSON (dict/list)

    return spectrum

def perform_EDS_measurements(MeasuredIdxs):
    # COnfiguration
    
    width = 256
    height = 256
    SavePath = r"C:\PhenomData\FastAPI Tests\AcquiringSpotSpectrum"
    MainPath = "/"
    sample_name = 'FastAPI_test_sample'
    dwell_time = 1

    # Make timestamped folder
    # Get the current time as a string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    prefix = 'Run'
    # Create the folder name (e.g., "Run_2025-08-07_14-38-00")
    folder_name = f"{prefix}_{timestamp}"

    # Full path
    full_path = os.path.join(SavePath, folder_name)

    # Create the folder
    os.makedirs(full_path, exist_ok=True)
    print(np.shape(MeasuredIdxs))

    # Create list to store all the acquired spectra
    all_spectra = []

    for coordinate in MeasuredIdxs:
        print(type(coordinate))
        print(np.shape(coordinate))
        print(coordinate)
        i, j = coordinate
        i, j = i / 256 , j / 256
        print(f'i = {i}')
        print(f'j = {j}')
        url = "http://127.0.0.1:8000/acquire_EDS_spot_spectrum"
        params = {
            "i": i,
            "j": j,
            "width": width,
            "height": height,
            "SavePath": full_path,
            "MainPath": MainPath,
            "sample_name": sample_name,
            "dwell_time": dwell_time
        }

        response = requests.get(url, params=params)
        jsoned_response = response.json()
        print(f'Response type is {type(response)}')
        print(f'Once converted to json, it is of type {type(jsoned_response)}')
        print(f'The length of the dictionary is {len(jsoned_response)}')
        print(type(jsoned_response))        # Should be <class 'dict'>
        print(len(jsoned_response))         # Should be 1
        print(jsoned_response.keys())       # Shows the dictionary keys
        print(list(jsoned_response.keys())) # Convert to list if you want indexing
        for key, value in jsoned_response.items():
            print(type(value))
            print(f'Length of the spectrum list is {len(value)}')
            # print(value)
        spectrum = np.array(response.json()["spectrum"])
        print(f'Spectrum acquired through the url function has shape {np.shape(spectrum)}')
        all_spectra.append(spectrum)
        print(np.shape(all_spectra))

        # print(response.status_code)     # HTTP code
        # print(response.headers)         # Headers
        # print(response.text)            # Raw text
        # print(response.json())          # Parsed JSON (dict/list)

    return all_spectra