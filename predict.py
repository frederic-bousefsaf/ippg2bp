import numpy as np
import tensorflow as tf
import scipy.io

from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json


# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()

# Run over a specific GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# GLOBAL VARIABLES
VERBOSE = 2
BATCH_SIZE = 16


# LOAD MODEL
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('weights.h5')
model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error')
model.summary()


# LOAD TEST DATA
data = scipy.io.loadmat('data_test.mat')

xtest = np.zeros((data['CWT_ppg_test'].shape[1], data['CWT_ppg_test'][0,0]['cfs'][0,0].shape[0], data['CWT_ppg_test'][0,0]['cfs'][0,0].shape[1],2))
ytest = np.zeros((data['CWT_bp_test'].shape[1], data['CWT_bp_test'][0,0]['cfs'][0,0].shape[0], data['CWT_bp_test'][0,0]['cfs'][0,0].shape[1],2))

for i in range(data['CWT_ppg_test'].shape[1]):
    xtest[i,:,:,0] = np.real(data['CWT_ppg_test'][0,i]['cfs'][0,0])
    xtest[i,:,:,1] = np.imag(data['CWT_ppg_test'][0,i]['cfs'][0,0])
    ytest[i,:,:,0] = np.real(data['CWT_bp_test'][0,i]['cfs'][0,0])
    ytest[i,:,:,1] = np.imag(data['CWT_bp_test'][0,i]['cfs'][0,0])


results = np.zeros((ytest.shape[0]), dtype=np.object)

for i in range(ytest.shape[0]):
    pred = model.predict(np.expand_dims(xtest[i],0))
    results[i] = {}
    results[i]['prediction'] = pred[0]

scipy.io.savemat('results.mat',{'results':results})


