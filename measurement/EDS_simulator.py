
import numpy as np
from model import CNN_Classifier
import torch
from torch import softmax, Tensor, load
import matplotlib.pyplot as plt

class EDS_Simulator:
    
    def __init__(self, spectrums):

        self.spectrums = spectrums
        self.image = None
        self.classifier = CNN_Classifier(2)
        self.classifier.load_state_dict(
            load(r"C:\Users\labuser\Downloads\PyPhenom (2.1)\PyPhenom\Jacques_Argonne_Internship\Phenom Repositories\SLADS-Net_V3\SLADS-Net\measurement\pretrained_models\CNN.pt", weights_only=True)
        )

    
    def generate_classifications(self):
        # N = self.spectrums.shape[0]

        # # Run CNN on all spectra at once
        # logits = self.classifier(Tensor(self.spectrums).float().reshape(N, 1, 2048))
        # probs = softmax(logits, dim=1).detach().numpy()

        # # Get predicted class for each spectrum
        # classes = np.argmax(probs, axis=1)
        # print(type(classes))
        # print(classes)

        # Trying NOlan's code
        # Commented out the function below in order to adapt the function for input batches that are not equal in size to the entire image
        probabilities = softmax(
                            self.classifier(
                                Tensor(self.spectrums).float().reshape(-1, 1, 2048))
                        ).detach().numpy()
        print(f'The shape of probabilities is {np.shape(probabilities)}')
        self.image = np.argmax(probabilities, axis=2)
        print(f'The shape of self image is {np.shape(self.image)}')
    
    
    
    def get_measured_values(self, mask):
        pixel_coords = np.array(mask)
        row_indices, col_indices = pixel_coords.T
        return self.image[row_indices, col_indices]
    
    def plot_classifications(self):
        plt.imshow(self.image)
        plt.colorbar()
    
class EDS_Manager:
    
    def __init__(self, image_shape, spectrum_length = 2048):
        
        self.image = np.zeros(image_shape) - 1  # -1 -> undefined
        self.spectrums = np.zeros(image_shape)
        self.classifier = CNN_Classifier(2)
        self.classifier.load_state_dict(
            load(r"C:\Users\labuser\Downloads\PyPhenom (2.1)\PyPhenom\Jacques_Argonne_Internship\Phenom Repositories\SLADS-Net_V3\SLADS-Net\measurement\pretrained_models\CNN.pt", weights_only=True)
        )
    
    def update(self, position, spectrum):
        spectrum = torch.tensor(spectrum).float().reshape(1, 1, 2048)

        # Forward pass
        logits = self.classifier(spectrum)

        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        print(f'Probabilities are {probs}')
        # Detach before NumPy conversion
        intermediate = probs.detach().numpy()

        # Argmax in NumPy
        classification = np.argmax(intermediate)
        # print(f'Classification is {classification}')

        # Store classification result
        # print(f'position is {position}')
        # print(type(position))
        # print(position.shape)
        # print(position[0])
        # print(position[1])
        self.image[position[0]][position[1]] = classification
        return classification
    
    def get_position(self, position):
        res = self.image[position]
        return res if (res != -1) else None
    
    def get_measured_values(self, mask):
        print(f'The shape of the image is {np.shape(self.image)}')
        print(f'The shape of the mask is {np.shape(mask)}')
        return self.image[mask]
    
    def override_data(self, image, spectrum):
        pass