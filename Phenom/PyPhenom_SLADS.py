# Script for acquiring an image using the SEM

import sys, math, os, datetime, time, random
import PyPhenom as ppi
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import license #license information held in a seperate directory for transferability, can be hard coded


# Connect to the Phenom in imaging mode
phenom = ppi.Phenom(license.PhenomUsername, license.PhenomUsername, license.PhenomPassword)
#phenom = ppi.Phenom() # Use this instead of importing the license.py if the license is installed in the ppi license manager

# Switch to NavCam mode, move to a specified location, and capture an image
def AcquireNavCamImage(x_pos, y_pos, img):
    phenom.MoveToNavCam()
    # Move the phenom to an absolute location; (0,0) is the origin
    phenom.MoveTo(x_pos, y_pos, algorithm = ppi.NavigationAlgorithm.BacklashOnly) 
    # Move the phenom by a certain amount relative to its current position
    #phenom.MoveBy(1e-6, 1e-6) 
    acqCamParams = ppi.CamParams()
    acqCamParams.size = ppi.Size(912, 912)
    acqCamParams.nFrames = 1
    acqNavCam = phenom.NavCamAcquireImage(acqCamParams)
    ppi.Save(acqNavCam, f'{img}.tiff')

def AcquireNavCamImage_at_current_location(SavePath, img, size):
    phenom.MoveToNavCam()
    acqCamParams = ppi.CamParams()
    acqCamParams.size = ppi.Size(size, size)
    acqCamParams.nFrames = 1
    acqNavCam = phenom.NavCamAcquireImage(acqCamParams)
    os.chdir(SavePath)
    filename = f"{img}.tiff"
    filepath = os.path.join(SavePath, filename)

    # Save the image
    ppi.Save(acqNavCam, filepath)

    # Return the full image path
    return filepath


# Grid overlay is used for drift analysis
def overlay_grid_on_image(image_path, num_lines=10):
    """
    Overlays a cyan grid on a grayscale SEM image and saves the result.

    Parameters:
    - image_path (str): Path to the input image file (e.g., a .tiff SEM image).
    - num_lines (int): Number of grid intervals across the image. Default is 10.
    """

    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)

    height, width = img_array.shape

    # Dynamically calculate spacing
    x_spacing = width // num_lines
    y_spacing = height // num_lines

    # Set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_array, cmap='gray')

    # Overlay vertical grid lines
    for x in range(0, width, x_spacing):
        ax.axvline(x, color='cyan', linewidth=0.5)

    # Overlay horizontal grid lines
    for y in range(0, height, y_spacing):
        ax.axhline(y, color='cyan', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # Save the figure with "_withGrid.png" added to the original filename
    base, _ = os.path.splitext(image_path)
    output_path = f"{base}_withGrid.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('plots complete')

# Acquire SEM Image using spot pattern
def AcquireSEM_spot(x, y, filename):
    """
    Summary:
    Uses the SEM spot patterning method to reconstruct the image, as opposed to the standard raster scanning pattern. Saves the reconstruction to a file in the folder "SpotReconstructions"

    Parameters:
    - (x, y): Coordinate of the point to be scanned
    - filename (str): Name that will be used for the reconstructed image file

    Returns:
    - doesn't return anything, but saves the image to the directory it creates
    """
    # Make a timestamped directory for saving data
    mydir = os.path.join(r"C:\PhenomData\SpotReconstructions", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    os.chdir(mydir)

    # Prep for SEM image acquisition
    phenom.MoveToSem()
    print(phenom.GetOperationalMode())
    phenom.MoveTo(x, y, algorithm = ppi.NavigationAlgorithm.BacklashOnly)
    phenom.SemAutoFocus() # Autofocuses the SEM (which is the same as finding the optimal working distance)
    phenom.SemAutoContrastBrightness()

    '''# Acquire SEM image
    start_time = time.time()
    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(1024,1024) # Change to 256 x 256 for SLADS
    acqScanParams.detector = ppi.DetectorMode.All
    acqScanParams.nFrames = 16 # number of frames to average for signal to noise improvement.
    acqScanParams.hdr= False # a Boolean to use High Dynamic Range mode
    acqScanParams.scale = 1.0
    acq = phenom.SemAcquireImage(acqScanParams)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed taking image: {elapsed_time:.2f} seconds")
    acq.metadata.displayWidth = 0.5
    acq.metadata.dataBarLabel = "Label"
    acqWithDatabar = ppi.AddDatabar(acq)
    ppi.Save(acq, f'{filename}.tiff')
    ppi.Save(acqWithDatabar, f'{filename}withDatabar.tiff')'''

    '''# Acquire Spot reconstruction
    point = ppi.Patterning.PointScanPattern()
    point.position = ppi.Position(-0.2, -0.05)
    point.size = ppi.SizeD(0.15, 0.05)
    #phenom.SetSemScanDefinition(point.Render())
    vm = phenom.GetSemViewingMode()
    print(vm)
    vm.scanMode = ppi.ScanMode.Pattern
    phenom.SetSemViewingMode(vm)
    time.sleep(5)
    vm.scanMode = ppi.ScanMode.Imaging
    phenom.SetSemViewingMode(vm)'''

    '''# Rectangle reconstruction
    rectangle = ppi.Patterning.RectangleScanPattern()
    rectangle.center = ppi.Position(-0.2, -0.05)
    rectangle.size = ppi.SizeD(0.15, 0.05)
    rectangle.pitchX = 1e-3
    rectangle.pitchY = 1e-3
    rectangle.rotation = math.radians(20)
    rectangle.dwellTime = 1e-6
    rectangle.lineScanStyle = ppi.Patterning.LineScanStyle.Serpentine'''

    # Point pattern
    point = ppi.Patterning.PointScanPattern()
    point.position = ppi.Position(-0.2, -0.05)
    #phenom.SetSemScanDefinition()
    vm = phenom.GetSemViewingMode()
    print(vm)
    vm.scanMode = ppi.ScanMode.Pattern
    print(vm.scanMode)
    # phenom.SetSemViewingMode(vm)
    # time.sleep(5)
    vm.scanMode = ppi.ScanMode.Imaging
    print(vm.scanMode)
    # phenom.SetSemViewingMode(vm)
    # Acquire SEM image
    start_time = time.time()
    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(1024,1024) # Change to 256 x 256 for SLADS
    acqScanParams.detector = ppi.DetectorMode.All
    acqScanParams.nFrames = 16 # number of frames to average for signal to noise improvement.
    acqScanParams.hdr= False # a Boolean to use High Dynamic Range mode
    acqScanParams.scale = 1.0
    acq = phenom.SemAcquireImage(acqScanParams)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed taking image: {elapsed_time:.2f} seconds")
    acq.metadata.displayWidth = 0.5
    acq.metadata.dataBarLabel = "Label"
    acqWithDatabar = ppi.AddDatabar(acq)
    ppi.Save(acq, f'{filename}.tiff')
    ppi.Save(acqWithDatabar, f'{filename}withDatabar.tiff')

# SEM Drift Identification
def Analyze_SEM_Drift(x0, y0, x1, y1, filename):
    """
    Summary:
    Points the SEM and takes an image at an initial location (x0, y0), moves to a different "intermediate" location (x1, y1) and takes an image there, and then moves back to the initial location and takes another image. Overlays grids onto both the initial and final images so the user can manually analyze the drift that occurred.

    Parameters:
    - (x0, y0) (x1, y1) (float): Coordinates of the initial and intermediate images, respectively
    - filename (str): Name that will be used for both the initial and final image files.

    Returns:
    - doesn't return anything, but saves the image to the directory it creates
    """
    #Make a timestamped directory for saving data
    mydir = os.path.join(r"C:\PhenomData\DriftAnalysis", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    os.chdir(mydir)

    # Move far away
    phenom.MoveTo(0.009, 0.009)
    # Take SEM image at initial location
    # phenom.MoveTo(-9e-3, -9e-3)
    phenom.MoveTo(x0 + 1e-7, y0 + 1e-7) # Move very close to the desired acquisition position before moving to the desired position, in order to reduce drift
    AcquireSEMImage(x0, y0, filename + '_initial')

    # Move to intermediate location and take SEM image
    # AcquireSEMImage(x1, y1, filename + '_intermediate')
    phenom.MoveTo(0.009, 0.009)

    # Move back to initial location and take SEM image
    # phenom.MoveTo(-9e-3, -9e-3)
    phenom.MoveTo(x0 + 1e-7, y0 + 1e-7) # Move very close to the desired acquisition position before moving to the desired position, in order to reduce drift
    AcquireSEMImage(x0, y0, filename + '_final')

    # Overlay grids on initial and final images (not databar ones)
    overlay_grid_on_image(f"{filename}_initial.tiff")
    overlay_grid_on_image(f"{filename}_final.tiff")

def AcquireSEMImage_at_current_location(path, filename, image_side_length_in_pixels):
    phenom.MoveToSem()
    # phenom.SemAutoFocus() # Autofocuses the SEM (which is the same as finding the optimal working distance)
    # phenom.SemAutoContrastBrightness()

    # Change to directory where the image will be saved
    os.chdir(path)
    # Acquire SEM image
    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(image_side_length_in_pixels, image_side_length_in_pixels) # Change to 256 x 256 for SLADS
    acqScanParams.detector = ppi.DetectorMode.All
    acqScanParams.nFrames = 16 # number of frames to average for signal to noise improvement.
    acqScanParams.hdr= False # a Boolean to use High Dynamic Range mode
    acqScanParams.scale = 1.0
    acq = phenom.SemAcquireImage(acqScanParams)
    acq.metadata.displayWidth = 0.5
    acq.metadata.dataBarLabel = "Label"
    acqWithDatabar = ppi.AddDatabar(acq)
    # The two lines below were commented out to use a slightly different file saving version that allows us to return the file path
    # ppi.Save(acq, f'{filename}.tiff')
    # ppi.Save(acqWithDatabar, f'{filename}withDatabar.tiff')

     # Save both versions
    raw_path = os.path.join(path, f"{filename}.tiff")
    databar_path = os.path.join(path, f"{filename}withDatabar.tiff")

    ppi.Save(acq, raw_path)
    ppi.Save(acqWithDatabar, databar_path)

    # Return full file path of the raw image (or databar image if you prefer)
    return raw_path

# Switch to SEM mode, move to a specified location, and capture an image
def AcquireSEMImage(x_pos, y_pos, filename, convert_image_to_numpy = True):
    phenom.MoveToSem()
    print(phenom.GetOperationalMode())
    phenom.MoveTo(x_pos, y_pos, algorithm = ppi.NavigationAlgorithm.BacklashOnly)
    time.sleep(5)
    phenom.SemAutoFocus() # Autofocuses the SEM (which is the same as finding the optimal working distance)
    phenom.SemAutoContrastBrightness()
    # time.sleep(7)

    '''# Viewing Detector setting (modes can be found in PPI manual)
    viewingMode = phenom.GetSemViewingMode()
    viewingMode.scanParams.detector = ppi.DetectorMode.All # Using the SED detector requires extra steps; see PPI manual
    phenom.SetSemViewingMode(viewingMode)

    # Pressure setting
    pressure = phenom.SemGetVacuumChargeReductionState().pressureEstimate
    phenom.SemSetTargetVacuumChargeReduction(ppi.VacuumChargeReduction.Off) # see manual for vacuum options
    '''
    start_time = time.time()
    # Acquire SEM image
    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(1024,1024) # Change to 256 x 256 for SLADS
    acqScanParams.detector = ppi.DetectorMode.All
    acqScanParams.nFrames = 16 # number of frames to average for signal to noise improvement.
    acqScanParams.hdr= False # a Boolean to use High Dynamic Range mode
    acqScanParams.scale = 1.0
    acq = phenom.SemAcquireImage(acqScanParams)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed taking image: {elapsed_time:.2f} seconds")
    acq.metadata.displayWidth = 0.5
    acq.metadata.dataBarLabel = "Label"
    acqWithDatabar = ppi.AddDatabar(acq)
    ppi.Save(acq, f'{filename}.tiff')
    ppi.Save(acqWithDatabar, f'{filename}withDatabar.tiff')
    
    # Convert image to numpy array for image processing
    if convert_image_to_numpy:
        numpyArray = np.array(acq.image)

# Convert an SEM image to numpy for pixel analysis
# def Convert_SEM_to_numpy(image):


def StartUpSEM(kV, beam_intensity, hfw):
    phenom.Activate()
    phenom.MoveToSem()
    phenom.SetSemHighTension(kV) # Set the SEM high tension (i.e. -5000 corresponds to 5 kV)
    phenom.SetSemSpotSize(beam_intensity) # Set beam spot intensity (Low = 2.1, Image = 3.3, Point = 4.3, Map = 5.1)
    phenom.SetHFW(hfw) # Set the horizontal field width

def SEM_random_walk(nimages, trial_name):
    #Make a timestamped directory for saving data

    mydir = os.path.join(r"C:\PhenomData\AutomatedImagingTrials", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    os.chdir(mydir)
    for i in range(nimages):
        x_pos, y_pos = random.uniform(-0.003, 0.003), random.uniform(-0.003, 0.003) # can go to any position in the range (-0.009, 0.009)
        AcquireSEMImage(x_pos, y_pos, trial_name + str(i))

# Convert spectra to numpy array
def convert_spectrum(file_path):
    # Open the file and read lines
    with open(file_path, 'r') as file:
        # Read all lines in the file
        lines = file.readlines()

        # Extract values from lines 32 to 2079 (1-indexed: lines[31:2079])
        spectrum_values = []
        for i in range(31, 2079):  # Start from line 32 (index 31)
            line = lines[i].strip()  # Remove any extra whitespace or newline characters
            try:
                # Convert line to a float and append to the list
                value = float(line)
                spectrum_values.append(value)
            except ValueError:
                print(f"Warning: Unable to convert line {i+1} to float")

        # Convert the list of values to a numpy array
        spectrum_array = np.array(spectrum_values)

        # Ensure that we have exactly 2048 values
        if spectrum_array.size != 2048:
            raise ValueError(f"Expected 2048 values, but got {spectrum_array.size}.")

        # Return the 1x2048 numpy array
        return spectrum_array


# Create and run an EDS project, which includes one image and multiple spots
def Start_EDS_Acquisition(max_time, max_counts):
    # Confirm that I have an EDS license
    #print("Has EDX license: " + str(ppi.HasLicense(ppi.Features.Spectroscopy)))
    #print("Has mapping license: " + str(ppi.HasLicense(ppi.Features.EdsMap)))

    # Set up the analyzer to take EDS measurements
    analyzer = ppi.Application.ElementIdentification.EdsJobAnalyzer(phenom) # can have multiple spotmeasurements using one analyzer
    settings = phenom.LoadPulseProcessorSettings()
    analyzer.preset = settings.spot
    return analyzer

# Perform an EDS spot analysis
def AcquireEDS(positions, spot_counter, max_time, max_counts):
    """
    Should accept a format of form [[x_1, y_1], [x_2, y_2], etc.] for mask / coords
    aka list of coordinates, each of form [x,y]
    So that you can accept either 1 point or many
    Should return a numpy array of form len(coords) by len(each spectrum)

    In SLADS case, each update would be of form [[x,y]]
    """
    analyzer = Start_EDS_Acquisition(max_time, max_counts)
    # Initialize the numpy array that will be returned
    spectrums_array = np.zeros((len(positions), 2048))
    for index, position in enumerate(positions):
        x_pos, y_pos = position[0], position[1]
        if spot_counter % 12 != 0:
            if spot_counter % 12 != 11:
                # Add a spot (note the unorthodox coordinate system in the PPI manual)
                spotData = analyzer.AddSpot(ppi.Position(x_pos, y_pos), maxTime=max_time, maxCounts=max_counts) # stops capturing spot data when the first of max_time or max_counts conditions gets met
                analyzer.Wait() # This waits until all requested jobs (AddSpot) have been carried out
                ppi.Spectroscopy.WriteMsaFile(spotData.spotSpectrum, f"{spot_counter}.SpotSpectrum.msa") # Create spectrum file
                image.AddAnalysis(spotData) # Adds the spot data to the project image
            else:
                # Save the current project data
                ppi.Spectroscopy.SaveEdsProject(project, f'{spot_counter}.Spectrum.elid')
                ppi.Spectroscopy.SaveEdsProject(project, f'{spot_counter}.Spectrum.phen')

                # Add a spot (note the unorthodox coordinate system in the PPI manual)
                spotData = analyzer.AddSpot(ppi.Position(x_pos, y_pos), maxTime=max_time, maxCounts=max_counts) # stops capturing spot data when the first of max_time or max_counts conditions gets met
                analyzer.Wait() # This waits until all requested jobs (AddSpot) have been carried out
                ppi.Spectroscopy.WriteMsaFile(spotData.spotSpectrum, f"{spot_counter}.SpotSpectrum.msa") # Create spectrum file
                image.AddAnalysis(spotData) # Adds the spot data to the project image
        else:
            # Take an SEM image, create a project, and add the image to the project data
            semImage = phenom.SemAcquireImage(256, 256)
            image = ppi.Spectroscopy.EdsImage(semImage, f'{spot_counter}.Image1')
            project = ppi.Spectroscopy.EdsProject()
            project.AddImage(image)
            analyzer.AcquireDriftCorrectionReference() # Prevents drift by making the acquired image a reference image

            # Add a spot (note the unorthodox coordinate system in the PPI manual)
            spotData = analyzer.AddSpot(ppi.Position(x_pos, y_pos), maxTime=max_time, maxCounts=max_counts) # stops capturing spot data when the first of max_time or max_counts conditions gets met
            analyzer.Wait() # This waits until all requested jobs (AddSpot) have been carried out
            ppi.Spectroscopy.WriteMsaFile(spotData.spotSpectrum, f"{spot_counter}.SpotSpectrum.msa") # Create spectrum file
            image.AddAnalysis(spotData) # Adds the spot data to the project image

        # Extra elemental data capabilities that can be incorporated if needed
        #quantification = ppi.Spectroscopy.Quantify(spotData.spotSpectrum) # acquire the chemical composition of the spectrum
        # quantification = ppi.Spectroscopy.Quantify(spotData.spotSpectrum, elements) # if looking for specific elements
        # ID = ppi.Spectroscopy.Identify(spectrum, excludedElements) # if you want to exclude elements
        #energyRange = (0.0,10.0) 
        #spectrum_image = ppi.Spectroscopy.DrawSpectrum(spotData.spotSpectrum, quantification, energyRange = (0,0)) # energyRange is optional and leaving it blank will result in autosizing
        #ppi.Save(spectrum_image, f"{IterNum}.Spectrum.bmp")

        # Convert spectrum of a pixel to a numpy array
        spectrums_array[index] = convert_spectrum(f"{index}.SpotSpectrum.msa")
        spot_counter += 1

    return spectrums_array

'''
# Prepare to apply SLADS-EDS at a location
# Take an SEM image, create a project, and add the image to the project data
semImage = phenom.SemAcquireImage(256, 256)
image = ppi.Spectroscopy.EdsImage(semImage, f'{project_name}.Image1')

# Set up the analyzer to take EDS measurements
analyzer = ppi.Application.ElementIdentification.EdsJobAnalyzer(phenom) # can have multiple spot measurements using one analyzer
settings = phenom.LoadPulseProcessorSettings()
analyzer.preset = settings.spot
analyzer.AcquireDriftCorrectionReference() # Prevents drift at a certain image location
analyzer.EdsAnalyzerFailure.connect(lambda msg: print('EdsAnalyzerFailure("{}")'.format(msg))'''


# Acquire a SEM image
'''
StartUpSEM(-15000, 3.3, 5e-5) # arguments are: high tension, beam spot intensity, horizontal field width
filename = 'NickelSpheres'
x_pos = -0.001
y_pos = 0.0045
AcquireSEMImage(x_pos, y_pos, filename)

# Perform a random walk that captures n images
nimages = 4
trial_name = 'Location1_'
SEM_random_walk(nimages, trial_name)'''

'''
# Install a ppi license
ppi.InstallLicense('MVE015801777', 'MVE015801777', '7498CBHRZAJ2')
print("Has EDS license: " + str(ppi.HasLicense(ppi.Features.Spectroscopy)))
print("Has mapping license: " + str(ppi.HasLicense(ppi.Features.EdsMap)))
analyzer = ppi.Application.ElementIdentification.EdsJobAnalyzer(phenom) # creates the analyzer object
spotData = analyzer.AddSpot(ppi.Position(0, 0), maxTime=10, maxCounts=10000)
analyzer.Wait()
print(ppi.Spectroscopy.Quantify(spotData.spotSpectrum)) # see chemical composition of acquired spectrum
ppi.Spectroscopy.WriteMsaFile(spotData.spotSpectrum, f'{filename}.msa') 
for license in ppi.GetLicenses():
    print(license)
'''
'''
#Run an EDS project
print("Has EDS license: " + str(ppi.HasLicense(ppi.Features.Spectroscopy)))
print("Has mapping license: " + str(ppi.HasLicense(ppi.Features.EdsMap)))
ppi.CheckLicense(ppi.Features.Spectroscopy)
for license in ppi.GetLicenses():
    print(license)
#project_name = 'george'
#run_EDS_project(project_name)

# Convert a spectrum to a numpy array
filepath = "C:\PhenomData\AutomatedImagingTrials\PbSn_60\PbSn_60_Image3\PbSn_60_Image3_Sn\Export\Image 1_spot_1\spectrum.emsa"
spectrum_array = convert_spectrum(filepath)
print(spectrum_array)
np.savetxt('spectrum_array.txt', spectrum_array)'''
'''for license in ppi.GetLicenses():
    print(license)
    print("Has EDS license: " + str(ppi.HasLicense(ppi.Features.Spectroscopy)))
    print("Has mapping license: " + str(ppi.HasLicense(ppi.Features.EdsMap)))
    ppi.CheckLicense(ppi.Features.Spectroscopy)'''