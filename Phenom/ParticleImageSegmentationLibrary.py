# Particle Image Segmentation Library

import numpy as np
import cv2
from matplotlib import pyplot as plt
from getSpotSpectrum import getSpotSpectrum
from getSpotSpectrum import add_marker
from getSpotSpectrum import save_annotated_image
from getSpotSpectrum import initialize_marker_plot
from getSpotSpectrum import place_spot_marker
import time
import shutil
import json
import numpy as np
from PyPhenom_SLADS import AcquireSEMImage_at_current_location
from PyPhenom_SLADS import AcquireNavCamImage_at_current_location
import PyPhenom as ppi
import license
import os
from datetime import datetime

# Create an indexed folder
def create_indexed_folder(SavePath, folder_name, index):
    parent_folder = SavePath
    child_folder = folder_name + f'_{index}'
    full_path = os.path.join(parent_folder, child_folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path

# Create a folder with the current datetime
def create_timestamped_folder(base_path, prefix="Run"):
    # Get the current time as a string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create the folder name (e.g., "Run_2025-08-07_14-38-00")
    folder_name = f"{prefix}_{timestamp}"
    
    # Full path
    full_path = os.path.join(base_path, folder_name)
    
    # Create the folder
    os.makedirs(full_path, exist_ok=True)
    
    return full_path

# Create a folder
def create_folder(SavePath, folder_name):
    parent_folder = SavePath
    child_folder = folder_name
    full_path = os.path.join(parent_folder, child_folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path

# Convert numpy data to python format for saving into json metadata file
def convert_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):  # numpy scalar types
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    else:
        return obj
    
# re and save SEM image
def acquire_and_save_image(x, y, image_index, SavePath, image_side_length_in_pixels):
    image_filename = f'SEM_image_{image_index}'
    image_path = AcquireSEMImage_at_current_location(SavePath, os.path.basename(image_filename), image_side_length_in_pixels)
    return image_path

# Get current position
def get_current_position(phenom):
    pos = phenom.GetStageModeAndPosition().position
    return pos.x, pos.y

# See function title
def convert_micrometer_radius_to_pixel_area(micrometer_value, image, image_horizontal_field_width):
    # print("image shape:", np.shape(image))
    return np.pi * (micrometer_value)**2 * (1e-12) * (np.shape(image)[0])**2 * (image_horizontal_field_width)**-2

# Convert micrometers to pixels
def convert_micrometers_to_pixels(micrometer_value, image, image_horizontal_field_width):
    return micrometer_value * 1e6 * (np.shape(image)[0] / image_horizontal_field_width)

# Convert pixels to micrometers
def convert_pixels_to_micrometers(pixel_value, image, image_horizontal_field_width):
    return pixel_value * (image_horizontal_field_width / np.shape(image)[0]) * 1e6

# Create binary mask for image
def create_binary_mask(image_path, threshold_intensity, max_binary_value, images_path):

    image = cv2.imread(image_path, 0) # Load image in grayscale (0 corresponds to grayscale)

    # Create threshold for binary mask
    ret, threshold = cv2.threshold(image, threshold_intensity, max_binary_value, cv2.THRESH_BINARY)
    filename = 'Binary_Mask.png'
    os.chdir(images_path)
    cv2.imwrite(filename, threshold)
    
    return threshold

# Define particle class
class Particle():
 
    def __init__(self, name, center, sampling_points, radius, contour):
        self.name = name
        self.center = center
        self.sampling_points = sampling_points
        self.radius = radius
        self.contour = contour

# Filter contours
def filter_and_draw_contours(images_path, image, threshold, minimum_area, image_path):
    # Create copy of image where the contours will be drawn onto
    image_copy = image.copy()

    # Detect the contours
    contours, _ = cv2.findContours(image=threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # Filter out non-circular contours and small contours
    filtered_contours = []
    for contour in contours:
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue  # Avoid division by zero and exclude contours that are too small
        
        # Exclude contours that are close to the contour of the image (since they may be partially cut off)
        x, y, w, h = cv2.boundingRect(contour)
        image_height, image_width = image.shape[:2]

        epsilon = 1  # Allow small tolerance
        if x <= epsilon or y <= epsilon or (x + w) >= image_width - epsilon or (y + h) >= image_height - epsilon:
            continue  # Exclude this contour

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        circularity_threshold = 0.75 # Set this from 0 to 1, with 1 being perfectly circular. The value 0.75 is a good choice.
        
        if circularity > circularity_threshold and area > minimum_area:
            filtered_contours.append(contour)
            cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), 2)
    
    # Save filtered contours
    filename = 'filtered_contours.png'
    os.chdir(images_path)
    cv2.imwrite(filename, image_copy)

    return filtered_contours, image_copy




# Find particle center
def find_particle_center(contour, image_copy, i):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        # Use the centroid formula
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center = (cx, cy)

    return center

# Label particle center in a way which avoids overlap between the drawn radius and the labeling text
def label_particle_center(center, average_contour_point, image_copy, i):
    # Label the center in a way that avoids overlap with the drawn radius
    if center[1] < average_contour_point[1]:
        cv2.putText(img = image_copy, text = f"Particle {i}", org = (center[0] - 20, center[1] - 20),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 0), thickness = 2)
    else:
        cv2.putText(img = image_copy, text = f"Particle {i}", org = (center[0] - 20, center[1] + 20),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 0), thickness = 2)
# Find particle radius
def find_and_draw_particle_radius(image_copy, contour, center):

    # Get all distances from center to contour points
    contour_points = contour[:, 0, :] # reshapes the contour
    distances = np.linalg.norm(contour_points - center, axis=1)

    # Compute the mean distance from center to contour
    mean_val = np.mean(distances)

    # Find the index of the point whose distance is closest to the mean distance from center to contour
    average_contour_point_index = np.argmin(np.abs(distances - mean_val))

    # Get the point on the contour that is closest to the center (call this the acerage contour point)
    average_contour_point = contour_points[average_contour_point_index]

    # Get the radius of the particle
    radius = np.linalg.norm(average_contour_point - center)

    # Draw the line between the center and the point closest to the mean distance
    cv2.line(image_copy, average_contour_point, center, color = (0, 255, 0), thickness = 2)

    return radius, average_contour_point

# Pick out points to sample using EDS
def find_and_draw_radial_points_for_sampling(image_copy, center, average_contour_point, number_of_points):
    # Pick out six points for EDS spot measurements that are equally spaced from center to edge point (inclusive)
    sampling_points = []
    for j in range(0, number_of_points):
        point = center + (j / (number_of_points - 1)) * (average_contour_point - center)
        sampling_point = tuple(np.array(point).astype(int))
        sampling_points.append(sampling_point)
        cv2.circle(image_copy, sampling_point, radius = 3, color=(255, 0, 0), thickness=-1)
    
    return sampling_points

# Perform EDS analysis of all particles on an image
def Perform_EDS_on_particles_in_SEM_image(phenom, SavePath, address, position, threshold_intensity, max_binary_value, minimum_particle_radius, number_of_sampling_points, sample_name, images_taken, SEM_image_length, total_number_of_particles, dwell_time, image_side_length_in_pixels):
    # Set up the phenom horizontal field width
    phenom.SetHFW(SEM_image_length)
    
    # Set up the spectrometer
    dpp = phenom.Spectrometer
    settings = ppi.LoadEidSettings()
    dpp.ApplySettings(settings.spot)

    # Create folder where everything related to the current SEM image will be stored
    # SEM_image_folder_name = f'SEM_image_at_({position[0]}, {position[1]})'
    SEM_image_folder_name = 'SEM_image'
    SEM_image_path = create_indexed_folder(SavePath, SEM_image_folder_name, images_taken)
    print(f"Created folder with index {images_taken}")

    # Create folder for image-related data and add image to it
    images_path = create_folder(SEM_image_path, 'Images')
    image_path = acquire_and_save_image(position[0], position[1], images_taken, images_path, image_side_length_in_pixels)

    image = cv2.imread(image_path, 0) # Load the image in pixel format

    # Initialize the plotting
    # fig, ax, image_shape = initialize_marker_plot(image_path)

    # Create binary mask and add it to 'Images' folder
    threshold = create_binary_mask(image_path, threshold_intensity, max_binary_value, images_path)


    # Create folder for particle-related data
    particles_path = create_folder(SEM_image_path, 'Particles')

    # Filter the particles
    minimum_area = convert_micrometer_radius_to_pixel_area(minimum_particle_radius, threshold, SEM_image_length)
    print(f"minimum area = {minimum_area}")
    particles = []
    filtered_contours, image_copy = filter_and_draw_contours(images_path, image, threshold, minimum_area, image_path)
    print(f'number of contours detected = {len(filtered_contours)}')

    # Convert the filtered contours into particle data
    for i, contour in enumerate(filtered_contours):
        center = find_particle_center(contour, image_copy, i)
        radius, average_contour_point = find_and_draw_particle_radius(image_copy, contour, center)
        label_particle_center(center, average_contour_point, image_copy, i)
        sampling_points = find_and_draw_radial_points_for_sampling(image_copy, center, average_contour_point, number_of_sampling_points)

        # Instantiate the particle object 
        particle = Particle(f'Particle_{i}', center, sampling_points, radius, contour)
        particles.append(particle)

        # Create folder for particle
        particle_folder = create_indexed_folder(particles_path, 'Particle', i)

        # Save particle EDS data
        spectra_folder_path= create_indexed_folder(particle_folder, 'Spectra_Particle', i)
        perform_EDS_on_single_particle(image, spectra_folder_path, SavePath, dpp, settings, address, phenom, particle, sample_name, dwell_time)

        # Update the particle count globally
        total_number_of_particles += 1


        # Convert the sampling points to EDS coordinates
        sampling_points_in_EDS_coords = []
        for sampling_point in particle.sampling_points:
            sampling_point_EDS_coords = convert_pixels_to_EDS_coordinates(sampling_point, image)
            sampling_points_in_EDS_coords.append(sampling_point_EDS_coords)
        
        # Save particle metadata
        metadata = {
            u"Radius (\u03bcm)": convert_pixels_to_micrometers(radius, image, SEM_image_length),
            "center": convert_pixels_to_EDS_coordinates(center, image),
            "sampling_points": sampling_points_in_EDS_coords
        }
        metadata_clean = convert_to_python(metadata)
        json_path = os.path.join(particle_folder, "metadata.json")
        with open(json_path, "w") as f:
            json.dump(metadata_clean, f, indent=4)


    # Save image with annotated contours, radii, and centers
    filename = 'annotated_particle_features.png'
    os.chdir(images_path)
    cv2.imwrite(filename, image_copy)
    


    # Save metadata for SEM image
    SEM_image_metadata = {
        "Image Coordinates": (position[0], position[1]),
        "Working Distance": phenom.GetSemWD()
    }
    SEM_image_metadata_clean = convert_to_python(SEM_image_metadata)
    json_path = os.path.join(SEM_image_path, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(SEM_image_metadata_clean, f, indent=4)

    # Get out of spot analysis mode
    acqScanParams = ppi.ScanParams()
    acqScanParams.size = ppi.Size(1024,1024) 
    acqScanParams.detector = ppi.DetectorMode.All 
    acqScanParams.nFrames = 16 
    acqScanParams.hdr = False 
    acqScanParams.center = ppi.Position(0,0)
    acqScanParams.scale = 1.0
    non_spot_mode = ppi.SemViewingMode(ppi.ScanMode.Imaging, acqScanParams)
    phenom.SetSemViewingMode(non_spot_mode)

    # Update the global particle count
    print(f'Total number of particles is {total_number_of_particles}')
    return total_number_of_particles

# Perform EDS Measurements on a single particle
def perform_EDS_on_single_particle(image, SavePath, MainPath, dpp, settings, address, phenom, particle, sample_name, dwell_time):
    # Loop through the previously determined number of points along the particle radius
    for sampling_point in particle.sampling_points:
        # Convert the point to EDS coordinates
        sampling_point_EDS_coords = convert_pixels_to_EDS_coordinates(sampling_point, image)
        print(sampling_point_EDS_coords)

        # Take EDS spot measurement
        size = (np.shape(image)[0], np.shape(image)[1])
        getSpotSpectrum(sampling_point_EDS_coords[0], sampling_point_EDS_coords[1], size,SavePath,MainPath, dpp,address,phenom, sample_name = sample_name, dwell_time=dwell_time)

        # Annotate the SEM image with the EDS spot measurement locations
        # os.chdir(SavePath)
        # add_marker(ax, image_shape, sampling_point_EDS_coords)


# Convert from OpenCV pixel coordinates to EDS Coordinates
def convert_pixels_to_EDS_coordinates(point, image):
    new_x_coordinate = point[0] / np.shape(image)[0]
    new_y_coordinate = point[1] / np.shape(image)[1]
    return (new_x_coordinate, new_y_coordinate)

# Draw the sampling region onto the navcam image
def draw_sampling_region_onto_NavCam_image(image_path, sample_area_center, sample_area_length, NavCam_HFW, NavCam_image_size):
    # Takes Navcam image as input and loads it in opencv
    image = cv2.imread(image_path, 0) # Load image in grayscale (0 corresponds to grayscale)

    # Draws a rectangle onto the Navcam image
    print(f'NavCam image size is {NavCam_image_size}')
    sample_area_length_in_pixels = (NavCam_image_size / NavCam_HFW) * sample_area_length
    top_left_point_x_pos = int((NavCam_image_size - sample_area_length_in_pixels) / 2)
    top_left_point_y_pos = int((NavCam_image_size - sample_area_length_in_pixels) / 2)
    bottom_right_point_x_pos =  int((NavCam_image_size + sample_area_length_in_pixels) / 2)
    bottom_right_point_y_pos = int((NavCam_image_size + sample_area_length_in_pixels) / 2)
    
    # # Keep the bottom left position in pixels for the SEM image location tracking
    # bottom_left_point_x_pos = int((NavCam_image_size - sample_area_length_in_pixels) / 2)
    # bottom_left_point_y_pos = int((NavCam_image_size + sample_area_length_in_pixels) / 2)

    pt1 = (top_left_point_x_pos, top_left_point_y_pos)
    pt2 = (bottom_right_point_x_pos, bottom_right_point_y_pos)
    print('hello')
    print(pt1)
    print(pt2)
    cv2.rectangle(image, pt1, pt2, color = (255, 255, 255))
    cv2.imwrite(image_path, image)

def add_marker_to_navcam_image(NavCam_image_path, position, sample_area_center, NavCam_image_size, NavCam_HFW, bottom_left_absolute_coords):
    image = cv2.imread(NavCam_image_path, 0)
    # x_pixel = int((NavCam_image_size / 2) + (position[0] - sample_area_center[0]) * (NavCam_image_size / NavCam_HFW))
    # y_pixel = int((NavCam_image_size / 2) + (position[1] - sample_area_center[1]) * (NavCam_image_size / NavCam_HFW))
    x_pixel = bottom_left_x_pixel + pixel_difference(position, bottom_left_absolute_coords, NavCam_image_size, NavCam_HFW)[0]
    y_pixel = bottom_left_y_pixel + pixel_difference(position, bottom_left_absolute_coords, NavCam_image_size, NavCam_HFW)[1]
    print(f'x pixel is {x_pixel}')
    print(f'y pixel is {y_pixel}')
    cv2.circle(image, (x_pixel, y_pixel), 10, (255, 255, 255))
    cv2.imwrite(NavCam_image_path, image)

def pixel_difference(point_1, point_2, image_pixel_length, image_meter_length):
    delta_x = int((point_1[0] - point_2[0]) * (image_pixel_length / image_meter_length))
    delta_y = int((point_1[1] - point_2[1]) * (image_pixel_length / image_meter_length))
    return (delta_x, delta_y)
    
# Perform chain imaging
def perform_chain_imaging(phenom, SavePath, threshold_intensity, max_binary_value, minimum_particle_radius, sample_area_center, sample_area_length, shift_between_images_length, sample_name, address, number_of_sampling_points_per_particle, SEM_image_length, total_number_of_particles, dwell_time, desired_number_of_particles, image_side_length_in_pixels):

    # Determine initial position for image chain
    initial_image_position = (sample_area_center[0] - sample_area_length, sample_area_center[0] - sample_area_length)
    phenom.MoveTo(initial_image_position[0], initial_image_position[1], algorithm = ppi.NavigationAlgorithm.BacklashOnly)

    # Print dimensions of sample area
    print(f'The sampling area has dimensions {sample_area_length} x {sample_area_length} and is centered at {sample_area_center}.')
    print(f"The upper boundary occurs at {sample_area_center[1] + 0.5 * sample_area_length}")

    # Acquire navcam image of  a larger region which contains the sampling region
    phenom.MoveToNavCam()
    if 2 * sample_area_length > 0.002017:
        NavCam_HFW = 2 * sample_area_length
    else:
        NavCam_HFW = 0.002017
    print(f'NavCam HFW is {NavCam_HFW}')
    phenom.SetHFW(NavCam_HFW)
    NavCam_image_size = 912
    NavCam_image_path = AcquireNavCamImage_at_current_location(SavePath, "larger_sampling_region", NavCam_image_size)
    draw_sampling_region_onto_NavCam_image(NavCam_image_path, sample_area_center, sample_area_length, NavCam_HFW, NavCam_image_size)

    # Keep track of bottom left of sampling area for added SEM Image Location markers
    bottom_left_of_sampling_area = (sample_area_center[0] - 0.5 * sample_area_length, sample_area_center[1] - 0.5 * sample_area_length)
    bottom_left_of_sampling_area_pixels = (int((bottom_left_of_sampling_area[0])))

    # Start image chain
    phenom.SetHFW(SEM_image_length)
    images_taken = 0
    leftmost_image_of_current_row_position = initial_image_position

    while total_number_of_particles < desired_number_of_particles:
        
        x_coordinate, y_coordinate = get_current_position(phenom)
        position = (x_coordinate, y_coordinate)

        # Check if position is past the right boundary of the sample region
        if x_coordinate > (sample_area_center[0] + sample_area_length):
            # Move to the start of the next row
            phenom.MoveTo(leftmost_image_of_current_row_position[0], leftmost_image_of_current_row_position[1], algorithm = ppi.NavigationAlgorithm.BacklashOnly)
            phenom.MoveBy(0, shift_between_images_length, algorithm = ppi.NavigationAlgorithm.BacklashOnly)
            print("Moving up a row")

            # Update position and row reference
            x_coordinate, y_coordinate = get_current_position(phenom)
            position = (x_coordinate, y_coordinate)
            leftmost_image_of_current_row_position = (x_coordinate, y_coordinate)

            # Check if position is out of desired vertical range
            if y_coordinate > (sample_area_center[1] + sample_area_length):
                print("Stopping image acquisition because vertical limit of the sample area has been reached")
                break

            total_number_of_particles = Perform_EDS_on_particles_in_SEM_image(phenom, SavePath, address, position, threshold_intensity, max_binary_value, minimum_particle_radius, number_of_sampling_points_per_particle, sample_name, images_taken, SEM_image_length, total_number_of_particles, dwell_time, image_side_length_in_pixels)

            # Add marker to navcam image
            add_marker_to_navcam_image(NavCam_image_path, position, sample_area_center, NavCam_image_size, NavCam_HFW)

            images_taken += 1

            # Move to the right
            print("Moving to the right")
            time.sleep(1)
            phenom.MoveBy(shift_between_images_length, 0, algorithm = ppi.NavigationAlgorithm.BacklashOnly)
            

        else:

            # Perform EDS Analysis
            total_number_of_particles = Perform_EDS_on_particles_in_SEM_image(phenom, SavePath, address, position, threshold_intensity, max_binary_value, minimum_particle_radius, number_of_sampling_points_per_particle, sample_name, images_taken, SEM_image_length, total_number_of_particles, dwell_time, image_side_length_in_pixels)

            # Add marker to navcam image
            add_marker_to_navcam_image(NavCam_image_path, position, sample_area_center, NavCam_image_size, NavCam_HFW)

            images_taken += 1

            # Move the phenom to the right
            time.sleep(1)
            phenom.MoveBy(shift_between_images_length, 0, algorithm = ppi.NavigationAlgorithm.BacklashOnly)
            print(position)
            print('Moving to the right')

    # Save project metadata
    project_metadata = {
        "Sample Name": sample_name,
        "Total Number of Particles": total_number_of_particles,
        "Acquisition/Dwell Time": dwell_time,
        "SEM Image Horizontal Field Width (meters)": SEM_image_length,
        "SEM Image Side Length (pixels)": image_side_length_in_pixels 
    }
    project_metadata_clean = convert_to_python(project_metadata)
    json_path = os.path.join(SavePath, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(project_metadata_clean, f, indent=4)
