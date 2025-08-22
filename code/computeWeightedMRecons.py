#! /usr/bin/env python3
import numpy as np

def computeWeightedMRecons(NeighborValues,NeighborWeights,TrainingInfo):
    """
    D/C in FeatReconMethod refers to Discrete/Continuous.
    DWM -> Discrete weighted mode
    CWM -> Continuous weighted mean (I'm not sure why sum -> mean, the below # comment was there before this summary docstring.)
    """
    
    # Weighted Mode Computation
    if TrainingInfo.FeatReconMethod == 'DWM':
        ClassLabels = np.unique(NeighborValues)
        print(np.shape(ClassLabels))
        print(ClassLabels[0])
        ClassWeightSums = np.zeros((np.shape(NeighborWeights)[0], np.shape(ClassLabels)[0]))
        for i in range(0, np.shape(ClassLabels)[0]):
            TempFeats = np.zeros((np.shape(NeighborWeights)[0], np.shape(NeighborWeights)[1]))
            np.copyto(TempFeats, NeighborWeights)
            # print("TempFeats shape:", TempFeats.shape)
            # print("NeighborValues shape:", NeighborValues.shape)
            # print("Mask shape:", (NeighborValues != ClassLabels[i]).shape)
            TempFeats[NeighborValues != ClassLabels[i]] = 0
            ClassWeightSums[:, i] = np.sum(TempFeats, axis=1)
        IdxOfMaxClass = np.argmax(ClassWeightSums, axis=1)
        ReconValues = ClassLabels[IdxOfMaxClass]

    # Weighted Mean Computation
    elif TrainingInfo.FeatReconMethod == 'CWM':
        #print((NeighborValues).shape)
        #print((NeighborWeights).shape)
        #print((NeighborValues * NeighborWeights[..., np.newaxis]).shape)
        # ReconValues = np.sum(NeighborValues * NeighborWeights[..., np.newaxis], axis=(1,2))
        ReconValues = np.sum(NeighborValues * NeighborWeights, axis=1) 
        #print(ReconValues.shape)

    return ReconValues
