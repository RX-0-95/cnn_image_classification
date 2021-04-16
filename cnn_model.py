from re import S
import tensorflow as tf 
from tensorflow import keras
from tensorflow.python.framework.func_graph import flatten 
from tensorflow.keras.regularizers import l2
class Lennet5(tf.keras.Model):
    def __init__(self, in_channel,out_channel, options={},input_shape= (32,32,3),*args, **kwargs):
        super(Lennet5,self).__init__(**kwargs)
        self.input_layer = tf.keras.layers.Input(input_shape)
        #Get options from kwargs 
        _options = options.copy() 
        conv1_out = _options.pop('conv1_out',18)
        conv2_out = _options.pop('conv2_out',32)
        fc0_out = _options.pop('fc0_out',120)
        fc1_out = _options.pop('fc1_out',80)
        conv_regularizaion = _options.pop('conv_reg',0.00)
        activation = _options.pop('activation','none')
        if len(_options) >0:
            extra = ', '.join('"%s"' % k for k in list(_options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra) 
        
        initializer= tf.initializers.VarianceScaling(scale=0.002)
       
        self.conv1 = keras.layers.Conv2D(conv1_out, [5,5], strides=(1,1),padding='VALID',
                                        kernel_initializer=initializer,activation=tf.nn.relu,
                                        kernel_regularizer=l2(conv_regularizaion))

        self.conv2 = keras.layers.Conv2D(conv2_out, [5,5], strides=(1,1), padding='VALID',
                                        kernel_initializer=initializer, activation=tf.nn.relu,
                                        kernel_regularizer=l2(conv_regularizaion))

        self.maxpool = keras.layers.MaxPool2D(pool_size=(2,2),strides=None, padding='VALID',data_format=None)
        self.flatten = keras.layers.Flatten()
        self.fc0 = keras.layers.Dense(fc0_out,kernel_initializer=initializer)
        self.fc1 = keras.layers.Dense(fc1_out,kernel_initializer=initializer)
        self.fc_out = keras.layers.Dense(out_channel,kernel_initializer=initializer)
        self.softmax = keras.layers.Softmax() 
        self.activation = keras.layers.ReLU()

        self.out = self.call(self.input_layer)   

    def conv_block(self,x):
        pass 
      

    def call(self, x,training=False):
        scores = None 
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.fc0(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc_out(x)
        scores = self.softmax(x)
        return scores

class Conv3net(tf.keras.Model):
  def __init__(self, in_channel,out_channel, options={},input_shape= (32,32,3),*args, **kwargs):
        super(Conv3net,self).__init__(**kwargs)
        self.input_layer = tf.keras.layers.Input(input_shape)
        #Get options from kwargs 
        _options = options.copy() 
        kernel_size = _options.pop('kernel_size',3)
        conv1_out = _options.pop('conv1_out',18)
        conv2_out = _options.pop('conv2_out',32)
        conv3_out = _options.pop('conv3_out',64)
        
        fc0_out = _options.pop('fc0_out',120)
        fc1_out = _options.pop('fc1_out',80)
        conv_regularizaion = _options.pop('conv_reg',0.00)
        activation = _options.pop('activation','none')
        if len(_options) >0:
            extra = ', '.join('"%s"' % k for k in list(_options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra) 
        
        initializer= tf.initializers.VarianceScaling(scale=0.002)
       
        self.conv1 = keras.layers.Conv2D(conv1_out, [kernel_size,kernel_size], strides=(1,1),padding='SAME',
                                        kernel_initializer=initializer,activation=tf.nn.relu,
                                        kernel_regularizer=l2(conv_regularizaion))

        self.conv2 = keras.layers.Conv2D(conv2_out, [kernel_size,kernel_size], strides=(1,1), padding='SAME',
                                        kernel_initializer=initializer, activation=tf.nn.relu,
                                        kernel_regularizer=l2(conv_regularizaion))
        self.conv3 = keras.layers.Conv2D(conv3_out, [kernel_size,kernel_size], strides=(1,1), padding='SAME',
                                      kernel_initializer=initializer, activation=tf.nn.relu,
                                      kernel_regularizer=l2(conv_regularizaion))
        

        self.maxpool = keras.layers.MaxPool2D(pool_size=(2,2),strides=None, padding='VALID',data_format=None)
        self.flatten = keras.layers.Flatten()
        self.fc0 = keras.layers.Dense(fc0_out,kernel_initializer=initializer)
        self.fc1 = keras.layers.Dense(fc1_out,kernel_initializer=initializer)
        self.fc_out = keras.layers.Dense(out_channel,kernel_initializer=initializer)
        self.softmax = keras.layers.Softmax() 
        self.activation = keras.layers.ReLU()

        self.out = self.call(self.input_layer)   

  def call(self, x,training=False):
      scores = None 
      x = self.conv1(x)
      x = self.maxpool(x)
      x = self.activation(x)
      x = self.conv2(x)
      x = self.maxpool(x)
      x = self.activation(x)
      x = self.conv3(x)
      x = self.maxpool(x)
      x = self.activation(x)

      x = self.flatten(x)
      x = self.fc0(x)
      x = self.activation(x)
      x = self.fc1(x)
      x = self.activation(x)
      x = self.fc_out(x)
      scores = self.softmax(x)
      return scores



class ThreeLayerConvNet(tf.keras.Model):
  def __init__(self, channel_1, channel_2, num_classes):
    super(ThreeLayerConvNet, self).__init__()

    initializer = tf.initializers.VarianceScaling(scale=0.002)
    self.conv1 = tf.keras.layers.Conv2D(channel_1, [5,5], [1,1], padding='valid',
                                     kernel_initializer=initializer,
                                     activation=tf.nn.relu)
    self.conv2 = tf.keras.layers.Conv2D(channel_2, [3,3], [1,1], padding='valid',
                                     kernel_initializer=initializer,
                                     activation=tf.nn.relu)
    self.fc = tf.keras.layers.Dense(num_classes, kernel_initializer=initializer)
    self.flatten = tf.keras.layers.Flatten()   
    self.softmax = tf.keras.layers.Softmax()     
    
  def call(self, x, training=False):
    scores = None

    padding = tf.constant([[0,0],[2,2],[2,2],[0,0]])
    x = tf.pad(x, padding, 'CONSTANT')
    x = self.conv1(x)
    padding = tf.constant([[0,0],[1,1],[1,1],[0,0]])
    x = tf.pad(x, padding, 'CONSTANT')
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.fc(x)
    scores = self.softmax(x)
       
    return scores