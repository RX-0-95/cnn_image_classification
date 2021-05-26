import re 
import tensorflow as tf 


@tf.custom_gradient
def to_fixpoint(value,wl=8,fl=4):
    
    #Quantize the variabl
    #value: tf.tensor 
    #wl: world length 
    #fl: fractional lenght 
    #return: qunatized tf.tensor 
    
    #max value 
    #TODO:round to nearest and stochastic roundting
    min_val = tf.cast(-2**(wl-fl-1),tf.float32)
    precision = tf.cast(2**(-fl),tf.float32)
    max_val = tf.cast(-min_val-precision,tf.float32)
    #print('max val:{},min_val:{}'.format(max_val,min_val))
    value_q = tf.convert_to_tensor(value,dtype=tf.float32)
    value_q = tf.math.round(value/precision)*precision
    value_q = tf.clip_by_value(value_q,min_val,max_val)
    if wl <=0:
        value_q = tf.zeros_like(value_q)
    def grad(upstream):
        return upstream,0.0,0.0
    return value_q,grad


def quantize_weights_bias(model,weight_wl=8,wieght_fl=8, bias_wl=16,bias_fl=8):
    for var in model.trainable_variables:
        if re.search('bias',var.name): 
            #print('find bias')
            var.assign(to_fixpoint(var, bias_wl,bias_fl))
        else:
            #print('find wieght')
            var.assign(to_fixpoint(var,weight_wl,wieght_fl))


def copy_weight(target_model,source_model,copy = True):
    for i in range(len(target_model.trainable_variables)):
        target_model.trainable_variables[i].assign(source_model.trainable_variables[i])
"""
def copy_weight(target_model,source_model,copy = True):
    for i in range(len(target_model.trainable_variables)):
        if copy:
            source_weight = tf.identity(source_model.trainable_variables[i])
        else:
            source_weight = source_model.trainable_variables[i]
        target_model.trainable_variables[i].assign(source_weight)
"""
def quantize_var(var,wl,fl):
    var.assign(to_fixpoint(var,wl,fl))

