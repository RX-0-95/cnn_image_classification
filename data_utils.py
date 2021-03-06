
import matplotlib.pyplot as plt 
from skimage import io 
import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from six.moves import cPickle as pickle
import os
from matplotlib.pyplot import imread
import platform

def plt_rgb(img:np.ndarray):
    img_np = img.copy().astype('uint8')
    plt.imshow(img_np)

def plt_rgb_histgram(img:np.ndarray): 
    img_np = img.copy().astype('uint8')
    #img = io.imread(img_np)




def load_pickle(f):
  version = platform.python_version_tuple()
  if version[0] == '2':
      return  pickle.load(f)
  elif version[0] == '3':
      return  pickle.load(f, encoding='latin1')
  raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(fn):
    with open(fn, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000,subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for training  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'CIFAR10_Data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    ## Data post_process 
   
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
    return X_train, y_train, X_val, y_val, X_test, y_test

def count_nested_list_elemets(element):
    count = 0 
    if isinstance(element,list):
        for each_element in element:
            count += count_nested_list_elemets(each_element)
    else:
        count += 1 
    return count 