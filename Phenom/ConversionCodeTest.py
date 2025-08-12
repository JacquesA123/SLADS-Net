# Check if the conversion code works
from ParticleImageSegmentationLibrary import convert_micrometer_area_to_pixels
import cv2
import numpy as np

micrometer_value = 62.4 ** 2
image_path = r"C:\PhenomData\NickelSpheres\Segmentation Project\Testing\Image Chain Tests\Test5\Run_2025-08-08_12-52-37\SEM_image_at_(0.002849999999999974, 0.002847500000000017)_1\Images\SEM_image_1_at_(0.002849999999999974, 0.002847500000000017).tiff"
image = cv2.imread(image_path, 0)
print(np.shape(image))
image_horizontal_field_width = 6.24e-5 # in meters
converted_value = convert_micrometer_area_to_pixels(micrometer_value, image, image_horizontal_field_width)
print(converted_value)