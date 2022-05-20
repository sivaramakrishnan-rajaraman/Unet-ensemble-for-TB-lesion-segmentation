#import libraries
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras import backend as K
K.clear_session()

#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #if having multiple GPUs in the system 

#%%
import tensorflow
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import h5py
import time
import skimage.transform
import cv2
import glob
import imageio
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from scipy import ndimage
from skimage import measure, color, io, img_as_ubyte
from skimage.segmentation import clear_border
import random
import csv
from tqdm import tqdm
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from sklearn.metrics import *
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import skimage.io as io
import skimage.transform as trans
from mpl_toolkits import axes_grid1
from skimage import img_as_uint
import tensorflow as tf
from tensorflow import keras
import math
from math import pi
from math import cos
from math import floor
from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import metrics as metrics
from tensorflow.keras.metrics import *
from tensorflow.keras.applications import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.preprocessing import *
from tensorflow.keras.preprocessing.image import *
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
sm.set_framework('tf.keras')
tensorflow.keras.backend.set_image_data_format('channels_last')

#%%
#helper functions
# for interactive plots

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

#%%
# https://matplotlib.org/stable/tutorials/colors/colormaps.html

def plot_sample(X, y, preds, binary_preds, ix=None):
    
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10), dpi=300)
    ax[0].set_facecolor('black')
    ax[0].grid(False)
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0].set_title('Input')

    ax[1].set_facecolor('black')
    ax[1].grid(False)
    ax[1].imshow(y[ix].squeeze(), cmap='gray')
    ax[1].set_title('GT Mask')

    ax[2].set_facecolor('black')
    ax[2].grid(False)
    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1, cmap='gray')
    ax[2].set_title('Predicted Mask')
    
    ax[3].set_facecolor('black')
    ax[3].grid(False)
    ax[3].imshow(X[ix, ..., 0], cmap='gray')
    ax[3].contour(preds[ix].squeeze(), colors='blue', levels=[0.5])
    ax[3].contour(y[ix].squeeze(), colors='red', levels=[0.5])
    ax[3].set_title('Predicted/ground-truth overlap');
    plt.savefig('combined_image.png', dpi=400, bbox_inches='tight')
    
#%%
# colored masks

MASK_COLORS = [
    "red", "green", "blue",
    "yellow", "magenta", "cyan"
]

def reshape_arr(arr):
    """[summary]
    
    Args:
        arr (numpy.ndarray): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])
        
def get_cmap(arr):
    """[summary]
    
    Args:
        arr (numpy.ndarray): [description]
    
    Returns:
        string: [description]
    """
    if arr.ndim == 3:
        return "gray"
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return "jet"
        elif arr.shape[3] == 1:
            return "gray"
        
def mask_to_rgba(mask, color="green"):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): [description]
        color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".
    
    Returns:
        numpy.ndarray: [description]
    """    
    assert(color in MASK_COLORS)
    assert(mask.ndim==3 or mask.ndim==2)

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)

def zero_pad_mask(mask, desired_size):
    """[summary]
    
    Args:
        mask (numpy.ndarray): [description]
        desired_size ([type]): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask

def plot_imgs(
        org_imgs,
        mask_imgs,
        pred_imgs=None,
        nm_img_to_plot=10,
        figsize=10,
        alpha=0.5,
        color="green"): #green before
    
    assert(color in MASK_COLORS)

    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(
        nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize), squeeze=False, dpi = 400
    )
    axes[0, 0].set_title("original", fontsize=30)
    axes[0, 1].set_title("ground truth", fontsize=30)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=30)
        axes[0, 3].set_title("overlay", fontsize=30)
    else:
        axes[0, 2].set_title("overlay", fontsize=30)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(
                mask_to_rgba(
                    zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
                cmap=get_cmap(pred_imgs),
                alpha=alpha,
            )
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(
                mask_to_rgba(
                    zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
                cmap=get_cmap(mask_imgs),
                alpha=alpha,
            )
            axes[m, 2].set_axis_off()
        im_id += 1

    plt.savefig('overlay.png', format='png', dpi=400)
    plt.show()

#%%
# loss function
'''
We use the boundary uncertainty evaluation and combine them
with the Focal Tversky loss function
# from https://github.com/mlyg/boundary-uncertainty

'''
def identify_axis(shape):
     # Three dimensional
     if len(shape) == 5 : return [1,2,3]
     # Two dimensional
     elif len(shape) == 4 : return [1,2]
     # Exception - Unknown
     else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
     
def border_uncertainty_sigmoid(seg, alpha = 0.9, beta = 0.1): #varies for the dataset
     """
     Parameters
     ----------
     alpha : float, optional
         controls certainty of ground truth inner borders, by default 0.9.
         Higher values more appropriate when over-segmentation is a concern
     beta : float, optional
         controls certainty of ground truth outer borders, by default 0.1
         Higher values more appropriate when under-segmentation is a concern
     """

     res = np.zeros_like(seg)
     check_seg = seg.astype(np.bool)
     
     seg = np.squeeze(seg)

     if check_seg.any():
         kernel = np.ones((3,3),np.uint8)
         im_erode = cv2.erode(seg,kernel,iterations = 1)
         im_dilate = cv2.dilate(seg,kernel,iterations = 1)
         
         # compute inner border and adjust certainty with alpha parameter
         inner = seg - im_erode
         inner = alpha * inner
         # compute outer border and adjust certainty with beta parameter
         outer = im_dilate - seg
         outer = beta * outer
         # combine adjusted borders together with unadjusted image
     
         res = inner + outer + im_erode
         
         res = np.expand_dims(res,axis=-1)

         return res
     else:
         return res

# Enables batch processing of boundary uncertainty 
def border_uncertainty_sigmoid_batch(y_true):
     y_true_numpy = y_true.numpy()
     return np.array([border_uncertainty_sigmoid(y) for y in y_true_numpy]).astype(np.float32)


#%%
#focal tversky loss with boundary uncertainty

def focal_tversky_loss_sigmoid(y_true, y_pred, delta=0.7, gamma=0.75, 
                                boundary=True, smooth=0.000001):
    axis = identify_axis(y_true.get_shape())
    if boundary:
        y_true = tf.py_function(func=border_uncertainty_sigmoid_batch, 
                                inp=[y_true], Tout=tf.float32)
    
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon) 
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Average class scores
    focal_tversky_loss = K.mean(K.pow((1-tversky_class), gamma))
     
    return focal_tversky_loss

#%%
# model evaluation metrics

def dice_coefficient(y_true, y_pred):
    # flatten the image arrays for true and pred
    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred[:,:,:,0])

    epsilon=1.0 # to prevent dividing by zero
    return (2*K.sum(y_true*y_pred)+epsilon)/(K.sum(y_true)+K.sum(y_pred)+epsilon)

def dice_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

def recall(y_true, y_pred):
    # flatten the image arrays for true and pred
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    return (K.sum(y_true * y_pred)/ (K.sum(y_true) + K.epsilon()))  

def precision(y_true, y_pred):
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    return (K.sum(y_true * y_pred) / (K.sum(y_pred) + K.epsilon()))  

def iou(y_true, y_pred):  #this can be used as a loss if you make it negative
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    union = y_true + ((1 - y_true) * y_pred)
    return (K.sum(y_true * y_pred) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def iou_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

#%%
# prepare data: training data

im_height, im_width = 256, 256
ids_train = next(os.walk("data/train/image"))[2] # list of names all images in the given path
print("No. of images = ", len(ids_train))
k = enumerate(ids_train)

X_tr = np.zeros((len(ids_train), im_height, im_width, 3), dtype=np.float32)
Y_tr = np.zeros((len(ids_train), im_height, im_width, 1), dtype=np.float32)
print(X_tr.shape)
print(Y_tr.shape)

for n, id_ in tqdm(enumerate(ids_train), total=len(ids_train)):
    # Load images
    img = load_img("data/train/image/"+id_, 
                    color_mode = "rgb")
    x_img = img_to_array(img)
    x_img = resize(x_img, (256,256,3), 
                    mode = 'constant', preserve_range = True)
    # Load masks
    mask = img_to_array(load_img("data/train/label/"+id_, 
                                  color_mode = "grayscale"))
    mask = resize(mask, (256,256,1), 
                  mode = 'constant', preserve_range = True)
    # Save images
    X_tr[n] = x_img/255.0
    Y_tr[n] = mask/255.0 

#%%
#validation data

ids_val = next(os.walk("data/val/image"))[2] # list of names all images in the given path
print("No. of images = ", len(ids_val))
k = enumerate(ids_val)

X_val = np.zeros((len(ids_val), im_height, im_width, 3), dtype=np.float32)
Y_val = np.zeros((len(ids_val), im_height, im_width, 1), dtype=np.float32)
print(X_val.shape)
print(Y_val.shape)

for n, id_ in tqdm(enumerate(ids_val), total=len(ids_val)):
    # Load images
    img = load_img("data/val/image/"+id_, 
                    color_mode = "rgb")
    x_img = img_to_array(img)
    x_img = resize(x_img, (256,256,3), 
                    mode = 'constant', preserve_range = True)
    # Load masks
    mask = img_to_array(load_img("data/val/label/"+id_, 
                                  color_mode = "grayscale"))
    mask = resize(mask, (256,256,1), 
                  mode = 'constant', preserve_range = True)
    # Save images
    X_val[n] = x_img/255.0
    Y_val[n] = mask/255.0 

#%%
#test data

ids_test = next(os.walk("data/test/image"))[2] # list of names all images in the given path
print("No. of images = ", len(ids_test))

X_ts = np.zeros((len(ids_test), im_height, im_width, 3), dtype=np.float32)
Y_ts = np.zeros((len(ids_test), im_height, im_width, 1), dtype=np.float32)
print(X_ts.shape)
print(Y_ts.shape)

for n, id_ in tqdm(enumerate(ids_test), total=len(ids_test)):
    # Load images
    img = load_img("data/test/image/"+id_, 
                    color_mode = "rgb")
    x_img = img_to_array(img)
    x_img = resize(x_img, (256,256,3), 
                    mode = 'constant', preserve_range = True)
    # Load masks
    mask = img_to_array(load_img("data/test/label/"+id_, 
                                  color_mode = "grayscale"))
    mask = resize(mask, (256,256,1), 
                  mode = 'constant', preserve_range = True)
    # Save images
    X_ts[n] = x_img/255.0
    Y_ts[n] = mask/255.0 

#%%
#print shapes of the data

print('The shape of the train images are', X_tr.shape)
print('The shape of the validation images are', X_val.shape)
print('The shape of the test images are',X_ts.shape)
print('The shape of the train masks are',Y_tr.shape)
print('The shape of the validation masks are',Y_val.shape)
print('The shape of the test masks are',Y_ts.shape)

#%%
# Visualize any randome image along with the mask for the train data

ix = random.randint(0, len(X_tr))
has_mask = Y_tr[ix].max() > 0 # salt indicator

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 15))
ax1.imshow(X_tr[ix, ..., 0], cmap = 'gray', interpolation = 'bilinear')
if has_mask: 
    ax1.contour(Y_tr[ix].squeeze(), colors = 'k', linewidths = 5, levels = [0.5])
ax1.set_title('CXR')
ax2.imshow(Y_tr[ix].squeeze(), cmap = 'gray', interpolation = 'bilinear')
ax2.set_title('TB_Mask')

#%%
# repeat for the test data

ix = random.randint(0, len(X_ts))
has_mask = Y_ts[ix].max() > 0 # salt indicator

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 15))
ax1.imshow(X_ts[ix, ..., 0], cmap = 'gray', interpolation = 'bilinear')
if has_mask: 
    ax1.contour(Y_ts[ix].squeeze(), colors = 'k', linewidths = 5, levels = [0.5])
ax1.set_title('CXR')
ax2.imshow(Y_ts[ix].squeeze(), cmap = 'gray', interpolation = 'bilinear')
ax2.set_title('TB_Mask')

#%%
# convert the images and masks to numpy arrays
np.save('data/X_tr.npy',X_tr)
np.save('data/X_val.npy',X_val)
np.save('data/X_ts.npy',X_ts)
np.save('data/Y_tr.npy',Y_tr)
np.save('data/Y_val.npy',Y_val)
np.save('data/Y_ts.npy',Y_ts)

#%%
#load the data from the numpy arrays
X_tr = np.load('data/X_tr.npy')
Y_tr = np.load('data/Y_tr.npy')
X_val = np.load('data/X_val.npy')
Y_val = np.load('data/Y_val.npy')
X_ts = np.load('data/X_ts.npy') 
Y_ts = np.load('data/Y_ts.npy') 

#%%
#instantiate the models
img_width, img_height = 256,256
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input)

#%%
n_classes=1 
activation='sigmoid' 
batch_size = 8
n_epochs = 256
image_size = 256

BACKBONE1 = 'inceptionv3'
BACKBONE2 = 'efficientnetb0' 
BACKBONE3 = 'resnet34'
BACKBONE4 = 'densenet121'
BACKBONE5 = 'seresnext50'

# define model
model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', 
                  classes=n_classes, activation=activation)
model1.summary()

model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', 
                  classes=n_classes, activation=activation)
model2.summary()

model3 = sm.Unet(BACKBONE3, encoder_weights='imagenet', 
                  classes=n_classes, activation=activation)
model3.summary()

model4 = sm.Unet(BACKBONE4, encoder_weights='imagenet', 
                  classes=n_classes, activation=activation)
model4.summary()

model5 = sm.Unet(BACKBONE5, encoder_weights='imagenet', 
                  classes=n_classes, activation=activation)
model5.summary()

#%%
# compile and train, repeat for other models

opt = keras.optimizers.Adam(lr=0.001)
loss_func=focal_tversky_loss_sigmoid,
model1.compile(optimizer=opt, 
              loss=loss_func, 
              metrics=['binary_accuracy', 
                       dice_coefficient, 
                       precision, 
                       recall, 
                       iou,
                       iou_loss])
print(model1.summary())
callbacks = [EarlyStopping(monitor='val_loss', 
                           patience=10, 
                           verbose=1, 
                           min_delta=1e-4,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss', 
                               factor=0.5, 
                               patience=5, 
                               verbose=1,
                               min_delta=1e-4, 
                               mode='min'),
             ModelCheckpoint(monitor='val_loss', 
                             filepath='weights/model1.hdf5', 
                             save_best_only=True, 
                             save_weights_only=True, 
                             mode='min', 
                             verbose = 1)]
results = model1.fit(X_tr, Y_tr, 
                    batch_size=batch_size, 
                    epochs=n_epochs, 
                    callbacks=callbacks,
                    validation_data=(X_val, Y_val))

#%%
#plot accuracy and loss curves
#loss
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), 
         np.min(results.history["val_loss"]), 
         marker="x", color="r", 
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()

#accuracy
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["acc"], label="acc")
plt.plot(results.history["val_acc"], label="val_acc")
plt.plot( np.argmax(results.history["val_acc"]), np.max(results.history["val_acc"]), 
         marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

#%%
#Inference: repeat for other models

model1.load_weights("weights/model1.hdf5")
model1.summary()
# Evaluate on validation set 
score_val = model.evaluate(X_val, Y_val, verbose=1)
print(model.metrics_names)
print('Metrics:', score_val)
# Evaluate on the test set 
score_test = model.evaluate(X_ts, Y_ts, verbose=1)
print(model.metrics_names)
print('Metrics:', score_test)

#%%
#predict on test data

preds_test = model.predict(X_ts, verbose=1)
Y_ts_hat = model.predict(X_ts, verbose=1)
print(preds_test.shape)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

#%%
#reshape the predictions and binarize
Y_ts_hat_int = Y_ts_hat.reshape(Y_ts_hat.shape[0]*Y_ts_hat.shape[1]*Y_ts_hat.shape[2], 1)
print(Y_ts_hat_int.shape)
Y_ts_int = Y_ts.reshape(Y_ts.shape[0]*Y_ts.shape[1]*Y_ts.shape[2], 1)
print(Y_ts_int.shape)
Y_ts_hat_int = np.where(Y_ts_hat_int>0.5, 1, 0)
Y_ts_int  = np.where(Y_ts_int>0.5, 1, 0)

#%%
#ROC curve
ground_truth_labels = Y_ts_int.ravel() # vectorization
score_value = Y_ts_hat_int.ravel() 
fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
roc_auc = auc(fpr,tpr)
print ("\nArea under the ROC curve : " +str(roc_auc))
roc_curve =plt.figure(figsize=(10,10), dpi=400)
plt.plot(fpr,tpr,'-',color="b", 
         label='Area Under the Curve (AUC = %0.4f)' % roc_auc)
plt.title('ROC curve',{'fontsize':20})
plt.xlabel("FPR (False Positive Rate)",{'fontsize':20})
plt.ylabel("TPR (True Positive Rate)",{'fontsize':20})
plt.legend(loc="lower right")
plt.savefig("./models/auc.png")

#%%
#PR curve, mAP is the area under the PR curve

precision, recall, thresholds = precision_recall_curve(Y_ts_int,Y_ts_hat_int)
precision = np.fliplr([precision])[0] 
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision,recall)
print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure(figsize=(15,10), dpi=40)
plt.rcParams['axes.facecolor'] = 'white'
plt.plot(recall,precision,'-',color="r",
          label='Area Under the Curve (AUC = %0.4f) ' % AUC_prec_rec)
plt.title('Precision - Recall curve',{'fontsize':20})
plt.xlabel("Recall",{'fontsize':20})
plt.ylabel("Precision",{'fontsize':20})
plt.legend(loc="lower right")
plt.savefig("./models/pr.png")

#%%
#other metrics: confusion matrix, precision, recall, IoU, Dice

threshold_confusion = 0.5
print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((Y_ts_hat_int.shape[0]))
for i in range(Y_ts_hat_int.shape[0]):
    if Y_ts_hat_int[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion_baseline = confusion_matrix(Y_ts_int, y_pred)
print (confusion_baseline)
accuracy_baseline = 0
if float(np.sum(confusion_baseline))!=0:
    accuracy_baseline = float(confusion_baseline[0,0]+confusion_baseline[1,1])/float(np.sum(confusion_baseline))
print ("Global Accuracy: " +str(accuracy_baseline))
specificity_baseline = 0
if float(confusion_baseline[0,0]+confusion_baseline[0,1])!=0:
    specificity_baseline = float(confusion_baseline[0,0])/float(confusion_baseline[0,0]+confusion_baseline[0,1])
print ("Specificity: " +str(specificity_baseline))
sensitivity_baseline = 0
if float(confusion_baseline[1,1]+confusion_baseline[1,0])!=0:
    sensitivity_baseline = float(confusion_baseline[1,1])/float(confusion_baseline[1,1]+confusion_baseline[1,0])
print ("Sensitivity: " +str(sensitivity_baseline))
precision_baseline = 0
if float(confusion_baseline[1,1]+confusion_baseline[0,1])!=0:
    precision_baseline = float(confusion_baseline[1,1])/float(confusion_baseline[1,1]+confusion_baseline[0,1])
print ("Precision: " +str(precision_baseline))

#Jaccard similarity index
jaccard_index_baseline = jaccard_score(Y_ts_int, y_pred)
print ("\nJaccard similarity score: " +str(jaccard_index_baseline))

#F1 score
F1_score_baseline = f1_score(Y_ts_int, y_pred, 
                    labels=None, 
                    average='binary', sample_weight=None)
print ("\nF1 score: " +str(F1_score_baseline))

#%%
# check quality of predictions using sample CXRs from the test set

plot_sample(X_ts, Y_ts, preds_test, preds_test_t, ix=27)

#%%
#Predict masks using the trained models, repeat for other models

source = glob.glob("data/test/image/*.png")
source.sort()
image_size = 256

model1.load_weights('weights/model1.hdf5')
model1.summary()

for f in source:
    img = Image.open(f)    
    img_name = f.split(os.sep)[-1]    
    #preprocess the image
    img = img.resize((256,256))
    x = image.img_to_array(img)
    x = x.astype('float32') / 255 #float32 before        
    x1 = np.expand_dims(x, axis=0) 
        
    #predict on the image
    pred1 = model1.predict(x1)    
    test_img1 = np.reshape(pred1, (256,256,1)) 
    #write to a image file
    imageio.imwrite('data/test/predict/{}.png'.format(img_name[:-4]), 
                    test_img1)

#%%
# Ensemble 
#doing bitwise-OR, AND, and MAX using the predictions of the top-K models
# K ranges from 3 to 5. Here, we demosntrate using the top-3 predictions. 
# Repat for other ensembles.

filenames1 = glob.glob("data/predictions/top_1/*.png")
filenames1.sort()
filenames2 = glob.glob("data/predictions/top_2/*.png")
filenames2.sort()
filenames3 = glob.glob("data/predictions/top_3/*.png")
filenames3.sort()
for f1,f2,f3 in zip(filenames1,filenames2,filenames3):
    img_name = os.path.basename(f1)
    img_name = img_name[:-4] + ".png"
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)
    img3 = cv2.imread(f3)  
    
    img_m1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_m2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_m3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)    
    
    #bitwise AND of the top-3 models    
    img_AND = cv2.bitwise_and(img_m1, img_m2, img_m3) 
    
    #bitwise OR of the top-3 models    
    img_OR = cv2.bitwise_or(img_m1, img_m2, img_m3)
    
    #bitwise MAX of the predictions of the top-3 models 
    out1 = np.maximum(img_m1, img_m2)
    img_MAX = np.maximum(img_m3, out1)   
    
    # write predictions to a image file, repeat for all ensemble methods   
    cv2.imwrite('data/predictions/AND/{}.png'.format(img_name[:-4]), 
                   img_AND)
    cv2.imwrite('data/predictions/OR/{}.png'.format(img_name[:-4]), 
                   img_OR)
    cv2.imwrite('data/predictions/MAX/{}.png'.format(img_name[:-4]), 
                   img_MAX)

#%%
# constructing a stacking ensemble
# A second-level fully-convolutional meta-learner is used to learn 
# the features extracted from the penumtimate layers of the top-K models
# K = [3,4,5]. Here, we show for top-3 models. Repeat for other models.

#%%
# perform stacking
n_models = 3 

def load_all_models(n_models):
    all_models = list()    
    model1.load_weights('weights/top_1.hdf5') # path to your model
    model_loss1a=Model(inputs=model1.input,
                        outputs=model1.get_layer('decoder_stage4b_relu').output) #name of the penultimate layer
    x1 = model_loss1a.output
    model1a = Model(inputs=model1.input, outputs=x1, name='model1')
    all_models.append(model1a)
    model2.load_weights('weights/top_2.hdf5')
    model_loss2a=Model(inputs=model2.input,
                        outputs=model2.get_layer('decoder_stage4b_relu').output) 
    x2 = model_loss2a.output
    model2a = Model(inputs=model2.input, outputs=x2, name='model2')
    all_models.append(model2a)
    model3.load_weights('weights/top_3.hdf5')
    model_loss3a=Model(inputs=model3.input,
                        outputs=model3.get_layer('decoder_stage4b_relu').output) 
    x3 = model_loss3a.output
    model3a = Model(inputs=model3.input, outputs=x3, name='model3')
    all_models.append(model3a)
    return all_models

# load models
n_members = 3
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

#%%
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers [1:]:
        # make not trainable
            layer.trainable = False    
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    ensemble_outputs = [model(model_input) for model in members]    
    merge = Concatenate()(ensemble_outputs)
    # meta-learner, fully-convolutional 
    x4 = Conv2D(128, (3,3), activation='relu', 
                name = 'NewConv1', padding='same')(merge)
    x5 = Conv2D(64, (3,3), activation='relu', 
                name = 'NewConv2', padding='same')(x4)
    x6 = Conv2D(32, (3,3), activation='relu', 
                name = 'NewConv3', padding='same')(x5)    
    x7 = Conv2D(1, (1,1), activation='sigmoid', 
                name = 'NewConvfinal')(x6)
    model= Model(inputs=model_input, #model_input,
                  outputs=x7)
    model.summary()
    return model
    
#%%
# Creating Ensemble and training the model
print("Creating Ensemble")
ensemble = define_stacked_model(members)
print("Ensemble architecture: ")
print(ensemble.summary())

#%%
#print layer names and their number
{i: v for i, v in enumerate(ensemble.layers)}

# print trainable layers
for l in ensemble.layers:
    print(l.name, l.trainable)

#%%
#set trainable and non-trainable layers
# make everything until the stacked-meta-learner as non-trainable
for layer in ensemble.layers[:6]: # varies depending on your model architecture
    layer.trainable = False
for layer in ensemble.layers[6:]:
    layer.trainable = True

# print trainable layers
for l in ensemble.layers:
    print(l.name, l.trainable)    

#%%
# compile and train the model
opt = keras.optimizers.Adam(lr=0.001)
loss_func=focal_tversky_loss_sigmoid
ensemble.compile(optimizer=opt, 
              loss=loss_func, 
              metrics=['binary_accuracy', 
                       dice_coefficient, 
                       precision, 
                       recall, 
                       iou,
                       iou_loss])

callbacks = [EarlyStopping(monitor='val_loss', 
                           patience=10, 
                           verbose=1, 
                           min_delta=1e-4,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss', 
                               factor=0.5, 
                               patience=5, 
                               verbose=1,
                               min_delta=1e-4, 
                               mode='min'),
             ModelCheckpoint(monitor='val_loss', 
                             filepath='weights/ensemble.hdf5', 
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min', 
                             verbose = 1)]

t=time.time() 
print('-'*30)
print('Start Training the model...')
print('-'*30)

results_ensemble = ensemble.fit(X_tr, Y_tr, 
                    batch_size=batch_size, 
                    epochs=256, #n_epochs, 
                    callbacks=callbacks,
                    verbose=1,
                    validation_data=(X_val, Y_val))
print('Training time: %s' % (time.time()-t))
#%%
# evaluate performance like before

'''
END OF CODE

'''