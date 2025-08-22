# Run Nickel Spheres EDS Analysis

import ParticleImageSegmentationLibrary as pisl
import PyPhenom as ppi
import license

## Prepare the Phenom SEM ##
address = ppi.FindPhenom(1).ip
phenom = ppi.Phenom(address,  license.PhenomUsername,  license.PhenomPassword)

## Project Configuration ##

# Project Folder
save_path = r"C:\PhenomData\NickelSpheres\Segmentation Project\Testing\Image Chain Tests\Test5"
project_path = pisl.create_timestamped_folder(save_path)


# Sampling Area Settings
sample_name = 'Nickel Spheres 8/20/25'
sample_area_center = (1.2e-3, 2e-3) # Coordinates range from -9e-3 to 9e-3 (meters)
sample_area_length = 1.5e-3 # Define the length of the square-shaped samplign area

# Particle info
desired_number_of_particles = 500 # Autonomous Scanning will stop after this many particles have been analyzed
minimum_particle_radius = 4.7 # Please enter this value in micrometers

# SEM Settings
SEM_image_length = 6.24e-5 # Can be from around 2.4e-6 to 6.98e-4 meters
shift_between_images_length = SEM_image_length * 2 # Multiply by a constant for extra cushion to prevent detection of duplicate features
image_side_length_in_pixels = 1024

# EDS Settings
dwell_time = 2 # Amount of EDS Acquisition time per particle
number_of_sampling_points_per_particle = 5 # Set the number of points to be acquired radially

# Image Processing Settings
threshold_intensity = 125 # Values below this will be filtered out as background
max_binary_value = 255 # Values above the threshold intensity will be set to this


## Run the project ##

# Global variables
total_number_of_particles = 0 # This global variable will be updated throughout the imaging process and eventually stored as project metadata

# Perform Chain Imaging
pisl.perform_chain_imaging(phenom, project_path, threshold_intensity, max_binary_value, minimum_particle_radius, sample_area_center, sample_area_length, shift_between_images_length, sample_name, address, number_of_sampling_points_per_particle, SEM_image_length, total_number_of_particles, dwell_time, desired_number_of_particles, image_side_length_in_pixels)


