# Print EDS Dict
import numpy as np


Path = r'C:\Users\labuser\Downloads\PyPhenom (2.1)\PyPhenom\Jacques_Argonne_Internship\PyPhenom1.7Codes\Useful_SASSI_stuff\EDS_dictionary\Dictionary\Dictionary\atom_sort.npy'
numpy_array = np.load(Path)
np.set_printoptions(threshold=np.inf)
print(numpy_array.shape)
print(numpy_array)