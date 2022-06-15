import numpy as np
import tensorflow as tf
import scipy.io

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam

from segmentation_models.models.unet import Unet        
# from segmentation_models.models.unetpp import Unetpp
# from segmentation_models.models.pspnet import PSPNet
# from segmentation_models.models.fpn import FPN
# from segmentation_models.models.linknet import Linknet 


# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()

# Run over a specific GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# GLOBAL VARIABLES
BACKBONE = 'resnext101' # 'vgg16', 'vgg19', 'resnet101', 'seresnet101', 'resnext101', 'seresnext101', 'inceptionresnetv2', 'inceptionv3', 'densenet201'
FREEZE_ENCODER = False
VERBOSE = 2
EPOCHS = 500
BATCH_SIZE = 16


# LOAD TRAINING DATA
data = scipy.io.loadmat('data_training.mat')

xtrain = np.zeros((data['CWT_ppg_training'].shape[1], data['CWT_ppg_training'][0,0]['cfs'][0,0].shape[0], data['CWT_ppg_training'][0,0]['cfs'][0,0].shape[1],2))
ytrain = np.zeros((data['CWT_bp_training'].shape[1], data['CWT_bp_training'][0,0]['cfs'][0,0].shape[0], data['CWT_bp_training'][0,0]['cfs'][0,0].shape[1],2))

for i in range(data['CWT_ppg_training'].shape[1]):
    xtrain[i,:,:,0] = np.real(data['CWT_ppg_training'][0,i]['cfs'][0,0])
    xtrain[i,:,:,1] = np.imag(data['CWT_ppg_training'][0,i]['cfs'][0,0])
    ytrain[i,:,:,0] = np.real(data['CWT_bp_training'][0,i]['cfs'][0,0])
    ytrain[i,:,:,1] = np.imag(data['CWT_bp_training'][0,i]['cfs'][0,0])


# LOAD VALIDATION DATA
data = scipy.io.loadmat('data_validation.mat')

xvalid = np.zeros((data['CWT_ppg_validation'].shape[1], data['CWT_ppg_validation'][0,0]['cfs'][0,0].shape[0], data['CWT_ppg_validation'][0,0]['cfs'][0,0].shape[1],2))
yvalid = np.zeros((data['CWT_bp_validation'].shape[1], data['CWT_bp_validation'][0,0]['cfs'][0,0].shape[0], data['CWT_bp_validation'][0,0]['cfs'][0,0].shape[1],2))

for i in range(data['CWT_ppg_validation'].shape[1]):
    xvalid[i,:,:,0] = np.real(data['CWT_ppg_validation'][0,i]['cfs'][0,0])
    xvalid[i,:,:,1] = np.imag(data['CWT_ppg_validation'][0,i]['cfs'][0,0])
    yvalid[i,:,:,0] = np.real(data['CWT_bp_validation'][0,i]['cfs'][0,0])
    yvalid[i,:,:,1] = np.imag(data['CWT_bp_validation'][0,i]['cfs'][0,0])


# DEFINE AND TRAIN MODEL
model = Unet(BACKBONE, classes=xtrain.shape[3], encoder_weights='imagenet', encoder_freeze=FREEZE_ENCODER, activation=None)

# Channel adaptation
inp = Input(shape=(None, None, xtrain.shape[-1]))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = model(l1)
model = Model(inp, out, name=model.name)

model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True, mode='auto')  
history_checkpoint = CSVLogger('history.csv', append=False)

model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error', metrics=['mean_absolute_error'])
model.summary()

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

history = model.fit(xtrain, ytrain, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(xvalid, yvalid), callbacks=[model_checkpoint, history_checkpoint], verbose=VERBOSE)
