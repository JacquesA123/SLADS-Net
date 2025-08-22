# -*- coding: utf-8 -*-

import time
import PyPhenom as ppi
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py # This is for the HDF5 file format
import hyperspy.api as hs
import numpy as np
    
'''def setSpot(phenom, imageSize, position):
    # VV was in original code, change is recorded below
    #mode = ppi.SemViewingMode(ppi.ScanMode.Spot, ppi.ScanParams(64, 64, 1, ppi.DetectorMode.All, False, 0, ppi.Position((position[0] + 0.5) / imageSize[0] - 0.5, (position[1] + 0.5) / imageSize[1] - 0.5)))

    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(imageSize[0],imageSize[1]) 
    acqScanParams.detector = ppi.DetectorMode.All 
    acqScanParams.nFrames = 16 
    acqScanParams.hdr = False 
    acqScanParams.center = ppi.Position((position[0] + 0.5) / imageSize[0] - 0.5,
                                        (position[1] + 0.5) / imageSize[1] - 0.5)
    acqScanParams.scale = 1.0
    """mode = ppi.SemViewingMode(
        ppi.ScanMode.Spot,
        ppi.ScanParams(64, 64, 1, ppi.DetectorMode.All, False, 0, 
                       ppi.Position((position[0] + 0.5) / imageSize[0] - 0.5, 
                                    (position[1] + 0.5) / imageSize[1] - 0.5)))"""
    mode = ppi.SemViewingMode(ppi.ScanMode.Spot, acqScanParams)
    
    # VV from original code
    phenom.SetSemViewingMode(mode)
'''
def setSpot_test(phenom, imageSize, position):
    """
    Summary:
    

    Parameters:
    - 

    Returns:
    - 
    """
    # VV was in original code, change is recorded below
    #mode = ppi.SemViewingMode(ppi.ScanMode.Spot, ppi.ScanParams(64, 64, 1, ppi.DetectorMode.All, False, 0, ppi.Position((position[0] + 0.5) / imageSize[0] - 0.5, (position[1] + 0.5) / imageSize[1] - 0.5)))

    acqScanParams = ppi.ScanParams()
    # acqScanParams.size = ppi.Size(imageSize[0],imageSize[1]) 
    acqScanParams.size = ppi.Size(imageSize[0],imageSize[1])
    acqScanParams.detector = ppi.DetectorMode.All 
    acqScanParams.nFrames = 16 
    acqScanParams.hdr = False 
    acqScanParams.center = ppi.Position(position[0], position[1]) # This works for square images only. See manual to implement rectangular images

    acqScanParams.scale = 0.001
    """mode = ppi.SemViewingMode(
        ppi.ScanMode.Spot,
        ppi.ScanParams(64, 64, 1, ppi.DetectorMode.All, False, 0, 
                       ppi.Position((position[0] + 0.5) / imageSize[0] - 0.5, 
                                    (position[1] + 0.5) / imageSize[1] - 0.5)))"""
    mode = ppi.SemViewingMode(ppi.ScanMode.Spot, acqScanParams)
    
    # VV from original code
    phenom.SetSemViewingMode(mode)

def writeSpectrum(spectrum, filename, address, sample_name, dwell_time):
    # VV was in original code, change is recorded below
    #msa = ppi.Spectrum.MsaData()
    #msa.spectrum = spectrum
    #ppi.Spectrum.WriteMsaFile(msa, filename)

    # Commented out the hyperspy code
    dict0 = {'offset': spectrum.offset, 'scale': spectrum.dispersion, 'size': len(spectrum.data), 'units': 'eV'}
    s = hs.signals.Signal1D(np.array(spectrum.data), axes=[dict0])
    s.set_signal_type("EDS_SEM")

    s.save(f'{filename}.msa', encoding = 'utf8')

    spectrum = np.array(spectrum.data)
    print(f'From inside writespectrum, we create a spectrum of shape {np.shape(spectrum)}')
    np.save(filename, spectrum)
    with h5py.File(f"{filename}.hdf5", "w") as f:
        # Create the dataset
        dset = f.create_dataset(f"{filename}_dataset", data=spectrum.data)

        # Add metadata (attributes) to the dataset
        dset.attrs["Sample Name"] = f'{sample_name}'
        dset.attrs['Instrument Address'] = f'{address}'
        dset.attrs['Dwell/Acquisition Time'] = f'{dwell_time}'
        dset.attrs['Date and Time of Acquisition'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return spectrum

def place_spot_marker(image_path, position):

    # Load your SEM image
    img = mpimg.imread(image_path)

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')  # Show grayscale image

    # Define list of normalized (x, y) coordinates
    x_coord, y_coord = position[0], position [1]
    print(f'x_coord = {x_coord}')
    print(f'y_coord = {y_coord}')

    # Get image dimensions in pixels
    height, width = img.shape[0], img.shape[1]
    print(f'height = {height}')
    print(f'width = {width}')

    # Plot an 'X' at each coordinate
    x_px = x_coord * width
    print(f'x_px = {x_px}')
    y_px = y_coord * height 
    print(f'xypx = {y_px}')
    ax.plot(x_px, y_px, marker='x', color='red', markersize=10, markeredgewidth=2)

    # Turn off axis labels
    ax.axis('off')

    # Save the new image
    plt.savefig(f'annotated_{os.path.basename(image_path)}_{position[0]}_{position[1]}.tiff', dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()

def initialize_marker_plot(image_path):
    """
    Initializes the plot with the SEM image and returns the figure and axis objects.
    """
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis('off')  # Turn off axis labels
    return fig, ax, img.shape  # Return image shape for coordinate scaling

def add_marker(ax, img_shape, position):
    """
    Adds a marker to an existing axis using normalized coordinates.
    """
    height, width = img_shape[0], img_shape[1]
    x_coord, y_coord = position
    x_px = x_coord * width
    y_px = y_coord * height
    ax.plot(x_px, y_px, marker='.', color='red', markersize=5, markeredgewidth=2)

    # Add text label with normalized coordinates
    label = f'({x_coord:.2f}, {y_coord:.2f})'
    ax.text(x_px + 0, y_px - 5, label, color='red', fontsize=4, weight='bold')  # Adjust offset and styling

def save_annotated_image(fig, image_path):
    """
    Saves the annotated image with all markers added.
    """
    out_path = f'annotated_{os.path.basename(image_path)}'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure when done
    return out_path

def getSpotSpectrum(
        i,j,
        size,
        SavePath,
        MainPath, 
        dpp,
        address,
        phenom,
        sample_name,
        dwell_time):   
    
    # Change coordinate system
    i_new, j_new = i - 0.5, j - 0.5
    setSpot_test(phenom, size, (i_new, j_new))

    # Perform EDS measurement
    dpp.ClearSpectrum()
    dpp.Start()
    time.sleep(dwell_time)
    dpp.Stop()

    # Save spectrum data
    os.chdir(SavePath)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    spectrum = writeSpectrum(dpp.GetSpectrum(), f'Spectrum at ({i:.2f}, {j:.2f}) taken at {timestamp}', address = address, sample_name = sample_name, dwell_time = dwell_time)
    os.chdir(MainPath)

    return spectrum # returns a spectrum in numpy array format

    # print(f'Spectrum Acquired at ({i}, {j})')