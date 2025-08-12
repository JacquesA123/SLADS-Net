# HDF5 file testing
import h5py
import numpy as np
from datetime import datetime

print(datetime.now())
arr = np.load("C:\PhenomData\EDS_Spot_Marking_tests\Test20\Spectrum at (-0.50, -0.50).npy")
with h5py.File("test_file.hdf5", "w") as f:
        dset = f.create_dataset("test_dataset", data=arr)

        # Add metadata (attributes) to the dataset
        dset.attrs["Sample_name"] = 'Lead-Tin'
        
         # get the minimum value
        print(min(dset)) 
        
        # get the maximum value
        print(max(dset))
        
        # get the values ranging from index 0 to 15
        print(dset[:15])
        # print(dset.attrs["Sample_name"])
with h5py.File("test_file.hdf5", "r") as f:
        print(dset.attrs["Sample_name"])
# f.attrs['large'] = np.arange(1_000_000, dtype=np.uint32)

# Read the HDF5 file
# reading_file = h5py.File('test_file.hdf5', 'r')

# Add attributes to the file


