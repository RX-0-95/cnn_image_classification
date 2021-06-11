from numpy import kaiser
from tensorflow import keras
from tensorflow.python.ops.control_flow_ops import Assert
import quantize_util as qu
import quantize_layers as ql 
import os
import tensorflow as tf

def regularized_padded_conv(to_fix_fn,*args, **kwargs):
    return ql.quantizeConv2D(*args, **kwargs, padding='same', kernel_regularizer=_regularizer,
                                  kernel_initializer='he_normal', use_bias=False,to_fix_fn=to_fix_fn)


def bn_relu(x,to_fix_fn):
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def shortcut(x,to_fix_fn, filters, stride, mode):
    if x.shape[-1] == filters:
        return x
    elif mode == 'B':
        return regularized_padded_conv(to_fix_fn,filters, 1, strides=stride)(x)
    elif mode == 'B_original':
        x = regularized_padded_conv(to_fix_fn,filters, 1, strides=stride)(x)
        return tf.keras.layers.BatchNormalization()(x)
    elif mode == 'A':
        return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride>1 else x,
                      paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])
    else:
        raise KeyError("Parameter shortcut_type not recognized!")
    

def original_block(x,to_fix_fn, filters, stride=1, **kwargs):
    c1 = regularized_padded_conv(to_fix_fn,filters, 3, strides=stride)(x)
    c2 = regularized_padded_conv(to_fix_fn,filters, 3)(bn_relu(c1,to_fix_fn))
    c2 = tf.keras.layers.BatchNormalization()(c2)
    
    mode = 'B_original' if _shortcut_type == 'B' else _shortcut_type
    x = shortcut(x,to_fix_fn, filters, stride, mode=mode)
    return tf.keras.layers.ReLU()(x + c2)
    
    
def preactivation_block(x,to_fix_fn, filters, stride=1, preact_block=False):
    flow = bn_relu(x,to_fix_fn)
    if preact_block:
        x = flow
        
    c1 = regularized_padded_conv(to_fix_fn,filters, 3, strides=stride)(flow)
    if _dropout:
        c1 = tf.keras.layers.Dropout(_dropout)(c1)
        
    c2 = regularized_padded_conv(to_fix_fn,filters, 3)(bn_relu(c1,to_fix_fn))
    x = shortcut(x,to_fix_fn, filters, stride, mode=_shortcut_type)
    return x + c2


def bootleneck_block(x, to_fix_fn,filters, stride=1, preact_block=False):
    flow = bn_relu(x,to_fix_fn)
    if preact_block:
        x = flow
         
    c1 = regularized_padded_conv(to_fix_fn,filters//_bootleneck_width, 1)(flow)
    c2 = regularized_padded_conv(to_fix_fn,filters//_bootleneck_width, 3, strides=stride)(bn_relu(c1))
    c3 = regularized_padded_conv(to_fix_fn,filters, 1)(bn_relu(c2,to_fix_fn))
    x = shortcut(x, to_fix_fn,filters, stride, mode=_shortcut_type)
    return x + c3


def group_of_blocks(x, to_fix_fn, block_type, num_blocks, filters, stride, block_idx=0):
    global _preact_shortcuts
    preact_block = True if _preact_shortcuts or block_idx == 0 else False

    x = block_type(x, to_fix_fn,filters, stride, preact_block=preact_block)
    for i in range(num_blocks-1):
        x = block_type(x,to_fix_fn,filters)
    return x



class Resnet(keras.Model):
    def __init__(self,input_shape, n_classes,quantizer, l2_reg=1e-4, group_sizes=(2, 2, 2), 
            features=(16, 32, 64), strides=(1, 2, 2),shortcut_type='B', 
            block_type='preactivated', first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
           dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True,*args, **kwargs):
        super().__init__(*args, **kwargs)
        global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, _bootleneck_width, _preact_shortcuts
        _bootleneck_width = bootleneck_width # used in ResNeXts and bootleneck blocks
        _regularizer = tf.keras.regularizers.l2(l2_reg)
        _shortcut_type = shortcut_type # used in blocks
        _cardinality = cardinality # used in ResNeXts
        _dropout = dropout # used in Wide ResNets
        _preact_shortcuts = preact_shortcuts
        block_types = {'preactivated': preactivation_block,
                   'bootleneck': bootleneck_block,
                   'original': original_block}
        self.quantizer = quantizer 
        self.to_fix_fn = self.quantizer.to_fix_fn
        self._input_shape = input_shape 
        selected_block = block_types[block_type]
        inputs = tf.keras.layers.Input(shape=input_shape)
        flow = regularized_padded_conv(to_fix_fn =self.to_fix_fn, **first_conv)(inputs)
        
        if block_type == 'original':
            flow = bn_relu(flow,to_fix_fn=self.to_fix_fn)
        
        for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
            flow = group_of_blocks(flow,
                                to_fix_fn=self.to_fix_fn,
                                block_type=selected_block,
                                num_blocks=group_size,
                                block_idx=block_idx,
                                filters=feature,
                                stride=stride)
        
        if block_type != 'original':
            flow = bn_relu(flow,to_fix_fn=self.to_fix_fn)
        
        flow = tf.keras.layers.GlobalAveragePooling2D()(flow)
        #outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(flow)
        self.dense = ql.quantizeDense(units = n_classes,kernel_regularizer=_regularizer,to_fix_fn=self.to_fix_fn)
        outputs = self.dense(flow)
    
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
    

    def set_full_precision(self):
        self.quantizer.set_full_precision_mode()

    def set_quantize(self):
        self.quantizer.set_quantize_mode()
    def get_quantizable_layer_count(self):
        self.quantizer.reset()
        dummy_input_shape = list((1,*(self._input_shape)))
        dummy_input = tf.zeros(dummy_input_shape)
        self.forward(dummy_input)
        layer_count = self.quantizer.get_layer_count()
        self.quantizer.reset() 
        return layer_count

    def get_quantizable_layers(self):
        """
        Return: [[layer_object, outputshape of the layer] ...[ ]]
        """
        self.quantizer.reset()
        self.quantizer.reset_layers_list() 
        self.quantizer.set_quantize_layer_extrac_mode() 
        dummy_input_shape = list((1,*(self._input_shape)))
        dummy_input = tf.zeros(dummy_input_shape)
        self.forward(dummy_input)
        layers_list = self.quantizer.get_quantize_layers()
        self.quantizer.set_full_precision_mode()
        self.quantizer.reset() 
        return layers_list      
    def forward(self,inputs):
        return self.model(inputs)

    def call(self, inputs, training=False):
        outputs = self.forward(inputs)
        self.quantizer.reset()
        return outputs

class Resnet20(Resnet):
    def __init__(self,quantizer, input_shape=(32,32,3), n_classes=10, l2_reg=1e-4, 
                group_sizes=(3,3,3), features=(16,32,64), 
                strides=(1,2,2),shortcut_type='A', block_type='original', 
                first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, 
                dropout=0, cardinality=1, bootleneck_width = 4, 
                preact_shortcuts = False, *args, **kwargs):
        super().__init__(input_shape, n_classes,quantizer=quantizer, l2_reg=l2_reg, 
                    group_sizes=group_sizes, features=features, 
                    strides=strides, shortcut_type=shortcut_type, 
                    block_type=block_type, first_conv=first_conv, 
                    dropout=dropout, cardinality=cardinality, 
                    bootleneck_width=bootleneck_width, 
                    preact_shortcuts=preact_shortcuts, *args, **kwargs)

def load_weights_func(model, model_name):
    try: model.load_weights(os.path.join('saved_models', model_name + '.tf'))
    except tf.errors.NotFoundError: print("No weights found for this model!")
    return model


def cifar_resnet20(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False):
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(3, 3, 3), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet20')
    return model




