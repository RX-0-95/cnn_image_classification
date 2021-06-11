#example of use the qunatizer layer and quantizer 
import quantize_layers as ql 
from tensorflow import keras
import tensorflow as tf 
from tensorflow.keras.regularizers import l2

class Quantize_Lenet5(keras.Model):
    def __init__(self,out_channel,quantizer, 
                options={},input_shape= (32,32,3),
                *args, **kwargs):
        super(Quantize_Lenet5,self).__init__(**kwargs)
        #self.input_layer = tf.keras.layers.Input(input_shape)
        #Get options from kwargs 
        self._input_shape = input_shape
        _options = options.copy() 
        conv1_out = _options.pop('conv1_out',18)
        conv2_out = _options.pop('conv2_out',32)
        fc0_out = _options.pop('fc0_out',120)
        fc1_out = _options.pop('fc1_out',80)
        conv_regularizaion = _options.pop('conv_reg',0.00)
        activation = _options.pop('activation','none')
        self.quantize =  _options.pop('quantize',False)
        self.quantize_wl = _options.pop('quantize_wl',8)
        self.quantize_fl = _options.pop('quantize_fl',4)
        self.quantizer = quantizer
        self.to_fix_fn = self.quantizer.to_fix_fn
        #self.to_fixpoint = partial(to_fixpoint,wl=self.quantize_wl,fl=self.quantize_fl)
        if len(_options) >0:
            extra = ', '.join('"%s"' % k for k in list(_options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra) 
        
        initializer= tf.initializers.VarianceScaling(scale=0.002)
        _regularizer = tf.keras.regularizers.l2(1e-4)
        self.conv1 = ql.quantizeConv2D(conv1_out, [5,5], strides=(1,1),padding='VALID',
                                        kernel_initializer=initializer,activation=tf.nn.relu,
                                        kernel_regularizer=l2(conv_regularizaion),
                                        to_fix_fn=self.to_fix_fn)

        self.conv2 = ql.quantizeConv2D(conv2_out, [5,5], strides=(1,1), padding='VALID',
                                        kernel_initializer=initializer, activation=tf.nn.relu,
                                        kernel_regularizer=l2(conv_regularizaion),
                                        to_fix_fn=self.to_fix_fn)

        self.maxpool = keras.layers.MaxPool2D(pool_size=(2,2),strides=None, padding='VALID',
                                        data_format=None)
        self.flatten = keras.layers.Flatten()
        self.fc0 = ql.quantizeDense(fc0_out,kernel_initializer=initializer,
                                    kernel_regularizer=_regularizer,
                                    to_fix_fn=self.to_fix_fn)
        self.fc1 = ql.quantizeDense(fc1_out,kernel_initializer=initializer,
                                    kernel_regularizer=_regularizer,
                                    to_fix_fn=self.to_fix_fn)
        self.fc_out = ql.quantizeDense(out_channel,kernel_initializer=initializer,
                                        kernel_regularizer=_regularizer,
                                        to_fix_fn=self.to_fix_fn)
        #self.softmax = keras.layers.Softmax()
        self.activation = keras.layers.ReLU()
        #self.out = self.call(tf.zeros(self._input_shape))

    def set_full_precision(self):
        self.quantizer.set_full_precision_mode()

    def set_quantize(self):
        self.quantizer.set_quantize_mode()

    def forward(self,x,training=False):
        scores = None 
        #print("---- 0 -----")
        #x = self.to_fix_fn(x)
        #x = self.quantizer.to_fix_fn(x)
        #print("---- 1 -----")
        x = self.conv1(x)
        #print("---- 2 -----")
        x = self.maxpool(x)
        x = self.activation(x)
        #print("---- 3 -----")
        x = self.conv2(x)
        #print("---- 4 -----")
        x = self.maxpool(x)
        x = self.activation(x)
        x = self.flatten(x)
        #print("---- 5 -----")
        x = self.fc0(x)
        x = self.activation(x)
        #print("---- 5 -----")
        x = self.fc1(x)
        x = self.activation(x)
        #print("---- 7 -----")
        scores = self.fc_out(x)
        #print("---- 8 -----")
        # scores = self.softmax(x)
        return scores
    
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


    def call(self,x,training=False):
        scores = self.forward(x,training)
        self.quantizer.reset() 
        return scores