from os import device_encoding
from re import S, X
from typing import Dict

import tensorflow as tf 
from tensorflow import nn
from tensorflow import keras
from tensorflow.keras import initializers, layers
from tensorflow.keras import regularizers
from tensorflow.python import tf2
from tensorflow.python.framework.func_graph import flatten 
from tensorflow.keras.regularizers import l2
from functools import partial
import numpy as np
from tensorflow.python.keras.backend import variable
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.ops.functional_ops import _set_read_only_resource_inputs_attr 

@tf.custom_gradient
def to_fixpoint(value,wl=8,fl=4):
    
    #Quantize the variabl
    #value: tf.tensor 
    #wl: world length 
    #fl: fractional lenght 
    #return: qunatized tf.tensor 
    
    #max value 
    min_val = tf.cast(-2**(wl-fl-1),tf.float32)
    precision = tf.cast(2**(-fl),tf.float32)
    max_val = tf.cast(-min_val-precision,tf.float32)
    #print('max val:{},min_val:{}'.format(max_val,min_val))
    value_q = tf.convert_to_tensor(value,dtype=tf.float32)
    value_q = tf.math.round(value/precision)*precision
    value_q = tf.clip_by_value(value_q,min_val,max_val)
    #value_q = value 
    #print(value_q.dtype)
    def grad(upstream):
        return upstream,0.0,0.0
    return value_q,grad


l2_reg=1e-3
conv2d_regularizer = tf.keras.regularizers.l2(l2_reg)

def initializer(scale = 0.002):
    return tf.initializers.VarianceScaling(scale)

conv3x3 = partial(keras.layers.Conv2D,kernel_size=3,
            use_bias=False,padding='same',kernel_initializer='he_normal',
            kernel_regularizer = conv2d_regularizer)



def activation_func(activation):
    return dict([
        ['relu',keras.layers.ReLU()],
        ['leaky_relu',keras.layers.LeakyReLU(alpha=0.01)],
        ['selu',keras.layers.ELU()],
        ['none',keras.layers.Layer()]
    ])[activation]

class ResidualBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, activation = 'relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels 
        self.activation = activation
        self.blocks = keras.layers.Layer()  
        self.activate = activation_func(activation)
        self.shortcut = keras.layers.Layer() 

    def call(self,x):
        residual = x 
        #print('residual shape: {}'.format(residual.shape))
        if self.should_apply_shortcut:
            #print('Apply short cut')
            residual = self.shortcut(x)
            #print('residual after shortcut: {}'.format(residual.shape))
        x = self.blocks(x)
        #print('blocks-fn: {}'.format(self.blocks))
        #print('x after blocks: {}'.format(x.shape))

        x += residual
        x = self.activate(x)
        return x

    # apply shortcut when channel size changes
    @property 
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels 


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, 
                expansion = 1, downsampling =1,
                conv=conv3x3, *args,**kwargs):
        super().__init__(in_channels, out_channels,*args, **kwargs)
        self.expansion,self.downsampling,self.conv = expansion,downsampling,conv 
        self.shortcut = keras.Sequential([
            keras.layers.Conv2D(self.out_channels,3,
                                strides=self.downsampling,use_bias=False,padding='same'),
            keras.layers.BatchNormalization() 
        ]) if self.should_apply_shortcut else None
    @property 
    def expand_channels(self):
        return self.out_channels*self.expansion
    
    @property 
    def should_apply_shortcut(self):
       return self.in_channels != self.out_channels 
    

class ResNetBasicBlock(ResNetResidualBlock):
    '''
    2 layers of 3x3 conv2d/batchnorm/conv and residual 
    '''
    expansion = 1 
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = keras.Sequential([
            self.conv(self.out_channels,padding='same',use_bias=False,strides=self.downsampling),
            
            keras.layers.BatchNormalization(),
            activation_func('relu'),
     
            self.conv(self.expand_channels,padding='same',use_bias=False),
            keras.layers.BatchNormalization(),
            #activation_func('relu'),
        ])

class ResNetLayer(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, block = ResNetBasicBlock,n=1,*args,**kwargs):
        super().__init__(*args, **kwargs)
        #print('Res layer block-fn:{}'.format(block))
        # Downsampling directily by convolution layers that have stride of 2 
        downsampling = (2,2) if in_channels != out_channels else 1 
        self.blocks = keras.Sequential()
        self.blocks.add(block(in_channels,out_channels,*args,**kwargs,downsampling = downsampling))
        for _ in range(n-1):
            #print("add layer")
            self.blocks.add(block(out_channels*block.expansion,out_channels,downsampling=1,*args,**kwargs))
    
    def call(self, inputs, **kwargs):
        #print('call res layer')
        inputs = self.blocks(inputs)
        return inputs 


class ResNetEncoder(keras.layers.Layer):
    def __init__(self,in_channels=3, block_sizes=[16,32,64],deepths=[2,2,2],
        activation='relu',block=ResNetBasicBlock,*args,**kwargs):
        super().__init__() 

        self.block_sizes = block_sizes
        self.gate = keras.Sequential([
            keras.layers.Conv2D(self.block_sizes[0],kernel_size=[3,3],strides=1,
                            padding='same',use_bias=True,kernel_initializer=initializer()),
            keras.layers.BatchNormalization(),
            activation_func(activation),
            #keras.layers.MaxPool2D(pool_size=(2,2),strides=None, padding='VALID')
        ])
        self.in_out_block_sizes = list(zip(block_sizes,block_sizes[1:]))
        self.blocks = list([
            ResNetLayer(block_sizes[0],block_sizes[0],n=deepths[0],block=block,*args,**kwargs),
            *[ResNetLayer(in_channels*block.expansion,
            out_channels,n=n,
            block=block,*args,**kwargs)
            for (in_channels,out_channels),n in zip(self.in_out_block_sizes,deepths[1:])]
        ])
          
    def call(self,x):
        #print(x.shape)
        x =self.gate(x)
        #print(x.shape)
        for block in self.blocks:
            x = block(x)
        #print("Encoder out: {}".format(x.shape))
        #x = self.res_block(x)
        return x 


class ResNetDecoder(keras.layers.Layer):
    def __init__(self,n_classes):
        super().__init__()
        self.flatten = keras.layers.Flatten() 
        self.decoder = keras.layers.Dense(n_classes,kernel_initializer=initializer())
        self.softmax = tf.keras.layers.Softmax()  
    def call(self,x):
        x = self.flatten(x)
        #print("Decoder flatten: {}".format(x.shape))
        x = self.decoder(x)
        #print("Decoder out: {}".format(x.shape))
        x = self.softmax(x)
        return x 


class ResNet20(keras.Model):
    def __init__(self,in_channels, classes, *args,**kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels)
        self.decoder = ResNetDecoder(classes)
        #self.decoder = keras.layers.Dense(classes)    
    def call(self,x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x 
        
#BK = ResNetBasicBlock(in_channels=3,out_channels=10)
input_shape = (1,32,32,2)
""""
model = ResNet20(3,10)
model.build(input_shape)
model.summary()
print(len(model.trainable_variables))
BK = ResNetResidualBlock(in_channels=1,out_channels=4)
input_shape = (1,32,32,2)
for var in model.trainable_variables:
    print(var.name)
print(activation_func('relu'))

print("fdsf")
"""
if __name__ == '__main__':
    """
    g1 = tf.random.Generator.from_seed(1,alg='philox')
    res_basic_block = ResNetBasicBlock(3,10)
    dum_input = g1.normal(shape = input_shape)
    dum_out = res_basic_block(dum_input)
    print(dum_out.shape)
    """
    in_channels = 16 
    out_channels = 32 
    dummy_input = tf.zeros([14,32,32,in_channels])
    resnet_layer = ResNetLayer(in_channels,out_channels,n=1)
    a = resnet_layer(dummy_input)
