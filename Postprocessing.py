# Neural Network for Solar Irradiance Modelling (NN-SIM).
# Postprocessing step to reverse normalization of Solar Spectral Irradiance (SSI)
# and Total Solar Irradiance (TSI) and save as matlab file
#
# input:    normalized (zero mean unit variance) SSI and TSI.(size[time x [SSI TSI]])
#
# output:   SSI and TSI.
#
#
# Code for the paper: Mauceri et al., "Neural Network for Solar Irradiance Modelling: NN-SIM, Solar Physics, 2019"
# available at: ???
# Steffen Mauceri, Sept. 2019
# Version 7.0

import numpy as np
import scipy.io

# specify name of NN-SIM predictions
name = 'ensemble_predictions_NN-SIM_noNoise'

#import prediction
prediction = np.load('data/'+ name +'.npy')

#convert back to irradiance
data = np.load('data/normalization_processed.npz')
target_raw_std = data['target_raw_std']
target_raw_mean = data['target_raw_mean']
irrad_NN = (prediction[1:,]*target_raw_std)+target_raw_mean

# save as matlab file
scipy.io.savemat('data/'+ name +'.mat', dict(irrad_NN=irrad_NN))