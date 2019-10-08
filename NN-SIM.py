# Neural Network for Solar Irradiance Modelling (NN-SIM).
# Maine Program to train and use an ensemble of Neural Networks to relate solar porxies to Solar Spectral Irradiance (SSI)
# and Total Solar Irradiance (TSI)
#
# input:    normalized (zero mean unit variance) solar proxy (size[time x proxy])
#           normalized (zero mean unit variance) SSI and TSI. (size[time x [SSI TSI]])
#           run 'Preprocessing.py' to generate input
#
# output:   normalized SSI and TSI at times where solar proxies were provided (1979 to present).
#
#
# Code for the paper: Mauceri et al., "Neural Network for Solar Irradiance Modelling: NN-SIM, Solar Physics, 2019"
# available at: ???
# Steffen Mauceri, Sept. 2019
# Version 7.0

from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import tensorflow as tf
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def importance_sampling(features_training,features_prec, targets_training, target_prec):
    # find max and min values in solar proxies, SSI and TSI and copies these samples multiple times.
    # this helps training the Neural Network

    max_id = targets_training.argmax(axis=0)
    min_id = targets_training.argmin(axis=0)

    targets_training = np.concatenate((targets_training,targets_training[min_id,:],targets_training[max_id,:], targets_training[min_id,:],targets_training[max_id,:]), axis=0)
    features_training = np.concatenate((features_training, features_training[min_id, :], features_training[max_id, :], features_training[min_id, :], features_training[max_id, :]), axis=0)

    features_prec = np.concatenate((features_prec,features_prec[min_id,:],features_prec[max_id,:], features_prec[min_id,:],features_prec[max_id,:]), axis=0)
    target_prec = np.concatenate((target_prec, target_prec[min_id, :], target_prec[max_id, :], target_prec[min_id, :], target_prec[max_id, :]), axis=0)

    return features_training, features_prec, targets_training, target_prec

def Noise(value, precision):
    #Adds noise to solar proxies, SSI and TSI proportional to their uncertainty
    X = np.multiply(precision, np.random.randn(len(value[:,0]),len(value[0,:])))
    value = value + X
    return value

def NN_SIM(ID):
    logs_path = 'tmp/logs/' + str(time.time()) + str(ID)

    if os.path.isfile('data/ensemble_predictions_' + name_Save + '.npy'):
        pass
    else:
        prediction = np.zeros((1, 290))
        np.save('data/ensemble_predictions_' + name_Save + '.npy', prediction)

    # scramble order of training samples
    np.random.seed(ID)
    rand = np.int16(np.random.rand(No_samples) * (len(features_training_all[:-1, 1])))
    features_training = features_training_all[rand, :]
    features_prec_training = features_prec_training_all[rand, :]
    targets_training = targets_training_all[rand, :]
    targets_prec_training = targets_prec_training_all[rand, :]

    if Noise_IO:
        # add noise to our solar proxies for complete time (1979 to present)
        features_prediction = Noise(features_prediction_all , features_prec_prediction)
        # add noise to our training set
        features_training = Noise(features_training, features_prec_training)
        targets_training = Noise(targets_training, targets_prec_training)
    else:
        features_prediction = features_prediction_all

    # Define Neural Network architecture of ensemble
    np.random.seed(ID)
    n_hidden_per_output = np.int32(np.random.rand()*4+3)  # number of neurons in first hidden layer
    connected = np.int32(np.random.rand()*10+5)
    reg1 = np.random.rand()*1/10000+2/10000
    reg2 = np.random.rand()*1/10000+2/10000
    #n_hidden_per_output = n_hidden_per_output.astype(np.int32)
    #connected = connected.astype(np.int32)

    n_inputs = 6  # number of different solar proxies
    n_outputs = 290  # number of outputs (number of SSI wavelength + 1)

    # tf Graph input placeholders
    X = tf.placeholder('float', [None, n_inputs])
    Y = tf.placeholder('float', [None, n_outputs])

    # transform fully connected layers to only partially connected
    cutting_matrix = np.zeros(
        [n_hidden_per_output * n_outputs * (connected + n_hidden_per_output), n_outputs])  # everything is unconnected
    k = 0
    for i in range(0, n_hidden_per_output * n_outputs, n_hidden_per_output):
        cutting_matrix[i:i + n_hidden_per_output * connected, k] = np.ones([n_hidden_per_output * connected])
        k += 1
    cutting_matrix[n_hidden_per_output:n_hidden_per_output + n_hidden_per_output * connected, 0] = np.zeros(
        [n_hidden_per_output * connected])  # remove uv connection from TSI
    cutting_matrix[150:150 + connected * n_hidden_per_output, 0] = np.ones(
        [n_hidden_per_output * connected])  # add vis connection
    cutter = tf.constant(cutting_matrix[:n_hidden_per_output * n_outputs, :], 'float')

    # Store weight & biases
    weights = {
        'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_per_output * n_outputs])),
        'out': tf.Variable(tf.random_normal([n_hidden_per_output * n_outputs, n_outputs])),
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_per_output * n_outputs])),
        'out': tf.Variable(tf.random_normal([n_outputs]))
    }

    # Define NN Architecture
    def MLP(x):
        layer_1 = tf.nn.elu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        out_layer = tf.add(tf.matmul(layer_1, tf.multiply(cutter, weights['out'])), biases['out'])
        return out_layer
    logits = MLP(X)

    # calc L2 regularization
    regularization = (tf.nn.l2_loss(weights['h1'])) * reg1 + tf.nn.l2_loss(
        tf.multiply(cutter, weights['out'])) * reg2
    # calc Mean Square Error
    MSE = tf.losses.mean_squared_error(logits, Y)
    # define cost function as Mean Square Error and Regularization
    loss_op = MSE + regularization  # add up both losses

    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.05, beta1=0.9, beta2=0.999, epsilon=1) #0.05; 0.9; 0.999; 1
    train_op = optimizer.minimize(loss_op)

    saver = tf.train.Saver()
    # Initializing variables
    init = tf.global_variables_initializer()

    minima = 10000  # make it infinity
    last_MSE = minima

    # ...........................start training...........................

    with tf.Session() as sess: #start a tf session
        sess.run(init)
        writer = tf.summary.FileWriter(logs_path, sess.graph)
        writer.close()

        if Load_IO: # load a NN that is already trained
            while True:
                try:
                    saver.restore(sess, 'data/ensemble' + name_Load + str(ID)) #load our trained model weights and biases
                    break
                except:
                    print('error in ' + str(ID))
                    return

        else:
            for epoch in range(training_epochs):  # training_epochs = number of total epoch runs
                n_batches = int(len(features[:, 1]) / batch_size)  # number of batches
                for i in range(n_batches):
                    sess.run([train_op, loss_op],
                             feed_dict={X: features_training[i * batch_size:i * batch_size + batch_size, :],
                                        Y: targets_training[i * batch_size:i * batch_size + batch_size, :]})

                if epoch % display_step == 0:
                    # test on testing dataset
                    MSE_test = sess.run(MSE, feed_dict={X: features_test, Y: targets_test})
                    MSE_training = sess.run(MSE, feed_dict={X: features_training, Y: targets_training})

                    print(MSE_training)
                    print(MSE_test)
                    print('epoch ' + str(epoch))

                    delta_MSE = last_MSE - MSE_training
                    if delta_MSE < 0.000001 and delta_MSE > 0:
                        print('No more learning')
                        break
                    last_MSE = MSE_training

                    if minima * 1.01 < MSE_test:
                        break

                    if minima > MSE_test:
                        minima = MSE_test

            saver.save(sess, 'data/ensemble' + name_Save + str(ID))


        prediction = sess.run([logits], feed_dict={X: features_prediction})
        prediction_np = np.asarray(prediction[0], dtype=np.float64)
        MSE_test = sess.run(MSE, feed_dict={X: features_test, Y: targets_test})
        MSE_training = sess.run(MSE, feed_dict={X: features_training, Y: targets_training})

        while True:
            try: # write results to file
                old_results = np.load('data/ensemble_predictions_' + name_Save +'.npy')
                prediction = np.concatenate((old_results, prediction_np), axis=0)
                np.save('data/ensemble_predictions_' + name_Save + '.npy', prediction)
                print(str(ID) + ' saved')
                print(MSE_training)
                print(MSE_test)
                break
            except: # if another parallel process is currently writing in that file,
                    # we wait for a few seconds and try again
                print(str(ID) + ' sleeps for 10 sec')
                time.sleep(np.random.rand() * 10 + 10)
#..........................................................................

## Make Changes ...................................................................

n_ensembles = 50        # initialize number of ensemble members                                         #50
training_epochs = 15000 # how often we iterate over our samples. Set to 0 if NN-SIM is already trained  #15000
batch_size = 100        # how many samples we average over before updating NN-SIM weights, biases.      #100
display_step = 40      # parameter for print/update
No_samples = 2000       # Number of samples we randomly choose from the complete training-set           #2000
Noise_IO = False        # Apply Noise to solar proxies, SSI and TSI
Load_IO = False        # Load a trained NN
name_Load='NN-SIM_noNoise'  # Name of NN we like to load #published version : 7.0_alwaysNoise and 7.0_neverNoise
name_Save = 'NN-SIM_noNoise'

## STOP Make Changes ...............................................................

#..........................................................................
# load full length solar proxy time-series (1979 to present)
data = np.load('data/prediction_processed.npz')
features_prediction_all = data['features_prediction']
features_prec_prediction = data['features_prec_prediction']

# load training dataset (solar proxy - SSI)
data = np.load('data/training_processed.npz')
targets = data['target_training'] # SIMc SSI
targets_prec_training = data['target_prec_training']# SIMc uncertainties
features = data['features_training'] # Solar Proxies
features_prec_training = data['features_prec_training'] # Solar Proxies uncertainties

# choose a validation set that we monitor during training
randTest = np.int16(np.random.rand(50) * (len(features[:-1, 1])))
#randTest = np.int16(np.arange(600,2,1200))
all = np.arange(len(features[:-1, 1]))
mask = np.ones(len(features[:-1, 1]), dtype=bool)
mask[randTest] = False
notTest = all[mask]

# split samples into training and validation
features_test = features[randTest,:]
targets_test=targets[randTest,:]

# and training-set
features_training_all=features[notTest,:]
features_prec_training_all = features_prec_training[notTest,:]

targets_training_all=targets[notTest,:]
targets_prec_training_all = targets_prec_training[notTest,:]

# perform importance sampling of the training samples. We multiply extreme training samples to enhance training
[features_training_all, features_prec_training_all, targets_training_all, targets_prec_training_all] = importance_sampling(features_training_all,features_prec_training_all, targets_training_all, targets_prec_training_all)

ID = np.arange(0,n_ensembles)    # initialize number of ensemble members #50

#paralize our for loop
pool=ThreadPool(processes=1)  # processes=1
pool.map(NN_SIM, ID)

pool.close()
pool.join()
