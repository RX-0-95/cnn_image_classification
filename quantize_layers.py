
from cnn_model import Quan_Lennet5, to_fixpoint
from tensorflow import keras
import tensorflow as tf 
from tensorflow.keras.regularizers import l2
import copy 
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



class quantizeConv2D(keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, padding, 
                kernel_initializer, 
                kernel_regularizer, to_fix_fn, **kwargs):
        super().__init__(filters, kernel_size, 
                         padding=padding, 
                        #activation=activation, 
                        kernel_initializer=kernel_initializer, 
                        kernel_regularizer=kernel_regularizer, 
                        **kwargs)
        self.to_fix_fn = to_fix_fn
    def call(self, x):
        x = super().call(x)
        x = self.to_fix_fn(x,self)
        return x 
        
class quantizeBatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, axis, momentum, epsilon, center, scale, 
                beta_initializer, gamma_initializer, moving_mean_initializer, 
                moving_variance_initializer, beta_regularizer, 
                gamma_regularizer, beta_constraint, gamma_constraint, 
                renorm, renorm_clipping, renorm_momentum, fused, trainable, 
                virtual_batch_size, adjustment, name,to_fix_fn, **kwargs):
        super().__init__(axis=axis, momentum=momentum, 
                epsilon=epsilon, center=center, scale=scale, 
                beta_initializer=beta_initializer, 
                gamma_initializer=gamma_initializer, 
                moving_mean_initializer=moving_mean_initializer, 
                moving_variance_initializer=moving_variance_initializer, 
                beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, 
                beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, 
                renorm=renorm, renorm_clipping=renorm_clipping, 
                renorm_momentum=renorm_momentum, fused=fused, 
                trainable=trainable, virtual_batch_size=virtual_batch_size, 
                adjustment=adjustment, name=name, **kwargs)
        self.to_fix_fn  = to_fix_fn
    def call(self, inputs, training):
        inputs = super().call(inputs, training=training)
        inputs = self.to_fix_fn(inputs)
        return input

class quantizeMaxPool2D(keras.layers.MaxPool2D):
    def __init__(self, pool_size, strides, 
                    padding, data_format,
                    to_fix_fn, **kwargs):
        super().__init__(pool_size=pool_size, strides=strides, 
                    padding=padding, data_format=data_format, **kwargs)
        self.to_fix_fn = to_fix_fn
    
    def call(self, inputs):
        inputs = super().call(inputs)
        inputs = self.to_fix_fn(inputs,self)
        return inputs


class quantizeDense(keras.layers.Dense):
    def __init__(self, units, kernel_regularizer,
                to_fix_fn,kernel_initializer=None, *args,**kwargs):
        super().__init__(units=units, kernel_regularizer = kernel_regularizer,
                        kernel_initializer=kernel_initializer, 
                        *args,**kwargs)
        self.to_fix_fn = to_fix_fn
    
    def call(self, inputs):
        inputs = super().call(inputs)
        inputs = self.to_fix_fn(inputs,self)
        return inputs

class quantizeSoftmax(keras.layers.Softmax):
    def __init__(self, to_fix_fn, **kwargs):
        super().__init__(**kwargs)
        self.to_fix_fn = to_fix_fn
    def call(self, inputs):
        inputs = super().call(inputs)
        inputs = self.to_fix_fn(inputs,self)
        return inputs


class ModelQunatize(object):
    def __init__(self) -> None:
        """
        wlfl_list: [[wl0,fl0],[wl1,fl1] ... ]
        """
        super().__init__()
        self.layer_counter = 0 
        self.full_precision_mode = True
        self.quantize_layer_extract_mode = False
        self.wlfl_list = [] 
        self.quantize_layers = []
        self.quantize_layers_outshape = []  
    def set_wlfl_list(self,wlfl):
        self.wlfl_list = wlfl
    
    def to_fix_fn(self,x,layer = None):
        if self.quantize_layer_extract_mode:
            #print('quantize layer extracting')
            self.quantize_layers.append(layer)
            self.quantize_layers_outshape.append(x.shape.as_list()[1:])

        elif self.full_precision_mode:
            #print('Full precision layer: {}'.format(self.layer_counter))
            pass 
        else:
            #print('Quantize layer: {}'.format(self.layer_counter))
            #wl,fl = self.wlfl_list[self.layer_counter]
            wl = max(self.wlfl_list.layer(self.layer_counter).kernel_wl,
                        self.wlfl_list.layer(self.layer_counter).bias_wl)
            fl = max(self.wlfl_list.layer(self.layer_counter).kernel_fl,
                    self.wlfl_list.layer(self.layer_counter).bias_fl)
            #print('layer counter {}'.format(self.layer_counter))
            #print("wl: {}  fl:{}".format(wl,fl))
            x = to_fixpoint(x,wl+8,fl)

        self.layer_counter += 1 
        return x 
    
    def set_quantize_mode(self):
        self.full_precision_mode = False 
        self.quantize_layer_extract_mode = False 
    def set_full_precision_mode(self):
        self.full_precision_mode = True  
        self.quantize_layer_extract_mode = False 
    def set_quantize_layer_extrac_mode(self):
        self.full_precision_mode = False 
        self.quantize_layer_extract_mode = True 
    
    def reset(self):
        self.layer_counter = 0 
    def reset_layers_list(self):
        self.quantize_layers = [] 
        self.quantize_layers_outshape = []
   
    def get_quantize_layers(self):
        return self.quantize_layers
    def get_quantize_layers_outshape(self):
        return self.quantize_layers_outshape
    def get_layer_count(self):
        return self.layer_counter

