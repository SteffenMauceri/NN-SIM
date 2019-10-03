# Neural Network for Solar Irradiance Modelling (NN-SIM).
# Preprocessing step to normalize solar porxies and Solar Spectral Irradiance (SSI)
# and Total Solar Irradiance (TSI)
#
# input:    solar proxy (size[time x proxy])
#           SSI and TSI. (size[time x [SSI TSI]])
#
# output:   normalized (zero mean unit variance) solar proxy, SSI and TSI.
#
#
# Code for the paper: Mauceri et al., "Neural Network for Solar Irradiance Modelling: NN-SIM, Solar Physics, 2019"
# available at: ???
# Steffen Mauceri, Sept. 2019
# Version 7.0

import numpy as np


# calculate a new normalization or use a previously saved one.
calc_new_IO = True


if calc_new_IO:
    features_raw_training=np.genfromtxt('data/feature_training.csv', delimiter=',')
    features_raw_prec_training=np.genfromtxt('data/feature_training_prec.csv', delimiter=',')

features_raw_prediction=np.genfromtxt('data/feature_prediction.csv', delimiter=',')
features_raw_prec_prediction=np.genfromtxt('data/feature_prediction_prec.csv', delimiter=',')

if calc_new_IO:
    features_raw_mean = np.mean(features_raw_prediction, axis=0)        #calculate mean of Features
    features_raw_std = np.std(features_raw_prediction, axis=0) +0.001   #calculate standard-distribution of Features
else:
    data = np.load('data/normalization_processed.npz')
    features_raw_mean = data['features_raw_mean']
    features_raw_std = data['features_raw_std']

# normalize features for better training
if calc_new_IO:
    features_training = (features_raw_training - features_raw_mean)/features_raw_std    #normalize Features
    features_prec_training = features_raw_prec_training / features_raw_std  # normalized standard error for noise

features_prediction = (features_raw_prediction - features_raw_mean)/features_raw_std
features_prec_prediction = features_raw_prec_prediction/features_raw_std


#normalize targets for better training
if calc_new_IO:
    target_raw_training = np.genfromtxt('data/target_training.csv', delimiter=',')
    target_raw_prec_training = np.genfromtxt('data/target_training_prec.csv', delimiter=',')

    target_raw_mean = np.mean(target_raw_training, axis=0)
    target_raw_std = np.std(target_raw_training, axis=0)
    target_training = (target_raw_training - target_raw_mean)/target_raw_std
    target_prec_training = target_raw_prec_training/target_raw_std

    #save everything
    np.savez('data/training_processed',
         target_training=target_training, target_prec_training=target_prec_training, features_training=features_training,
         features_prec_training=features_prec_training)
    np.savez('data/normalization_processed',
             features_raw_mean=features_raw_mean, features_raw_std=features_raw_std,
             target_raw_std=target_raw_std, target_raw_mean=target_raw_mean)

np.savez('data/prediction_processed',
         features_prediction=features_prediction, features_prec_prediction=features_prec_prediction)
