
from os import name
from random import shuffle
from numpy.lib.npyio import save
import tensorflow as tf 
import numpy as np
from tensorflow import keras 
import os 
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers 
class Solver(object):
    """
    Solver for tensorflow
    """
    def __init__(self,model,data_set,options={},verbose=True, plot_graph=True,save_model=True,
        *args,**kwargs) -> None:
        super().__init__()
        self.model  = model
        self.verbose = verbose
        self.plot_graph = plot_graph
        self.save_mode = save_model
        #unpack the options 
        self.options = options.copy() 
        self.device = self.options.pop('device','/device:GPU:0')
        self.save_dir = self.options.pop('save_dir','saved_model')
        self.save_model_name = self.options.pop('save_model_name','trained')
        self.save_every_num_epoch = self.options.pop('save_every_num_epoch',5)
        self.optimizer_type = self.options.pop('optimizer','adam')
        self.learning_rate = self.options.pop('lr',0.0001)
        self.num_epochs = self.options.pop('epoch_num',25)
        self.train_batch_size = self.options.pop('train_batch_size',32)
        self.logs = {
            'train_acc': [],
            'train_loss': [],
            'val_acc': [],
            'val_loss': [],
            } 

        #_train_val_split_ratio = _options.pop('train_val_split_ratio',0.8)
        #_val_batch_size = _options.pop('val_batch_size',8)
        #_weight_decay = _options.pop('weight_decay',0.001)
        #_scheduler_factor = _options.pop('scheduler_factor',0.8)
        #_scheduler_patience = _options.pop('scheduler_patience',50)
        #_save_flag = _options.pop('save_flag',False)
        #_min_save_epoch = _options.pop('min_save_epoch',50)
        #_model_dir = _options.pop('model_dir','model')
        #_model_name= _options.pop('model_name','model.pth')
        
        # Check if there is unexpected options 
        if len(self.options) >0:
            extra = ', '.join('"%s"' % k for k in list(self.options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra)

        # construct the data set 
        X_train = data_set['train_data']
        y_train = data_set['train_label']
        X_val = data_set['val_data']
        y_val = data_set['val_label']
        
        self.train_data = Dataset(X_train,y_train,self.train_batch_size,shuffle=True)
        self.val_data = Dataset(X_val,y_val,batch_size=X_val.shape[0])
        

        #self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model,label)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = self.gen_optimizer()
        

    def gen_optimizer(self):
        if self.optimizer_type =='adam':
            opt = tf.optimizers.Adam(learning_rate=self.learning_rate)
        if self.optimizer_type == 'sgd':
            opt = tf.optimizers.SGD(learning_rate = self.learning_rate)
        return opt 

    def plot_logs(self,logs):
        fig,(ax1,ax2) = plt.subplots(1,2) 
        ax1.plot(logs['train_loss'],label='Train')
        ax1.plot(logs['val_loss'],label = 'Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('iter(unit 100 iters)')
        ax1.set_ylabel('loss')
        ax2.plot(logs['train_acc'],label = 'Train')
        ax2.plot(logs['val_acc'],label = 'Validation')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('iter(unit 100 iters)')
        ax2.set_ylabel('acc')
        fig.legend() 
        plt.show() 
    
    
    def test_model(self, X,y):
        """
        Return: 
        Test_loss 
        Test_accuracy 
        """
        with tf.device(self.device):
            test_loss =tf.keras.metrics.Mean(name='test_loss')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
            scores = self.model(X)
            loss = self.loss_fn(y, scores)
            test_loss.update_state(loss)
            test_accuracy.update_state(y, scores)
            template = 'Loss: {},Test Accuracy: {}'     
            print(template.format(test_loss.result(),
                test_accuracy.result()*100))
            return test_loss.result(), test_accuracy.result() 

    def predict(self,X):
        scores = self.model(X)
        predict_label = tf.argmax(scores,axis=1).numpy()
        return predict_label

    def train(self):
        #self.logs.clear()
        model =self.model
        with tf.device(self.device):
            train_loss =tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

            val_loss = tf.keras.metrics.Mean(name='val_loss')
            val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
            t = 0 
            for epoch in range(self.num_epochs):
                train_loss.reset_states()
                train_accuracy.reset_states() 

                for x_np, y_np in self.train_data:
                    with tf.GradientTape() as tape:
                        #tape.watch(model.trainable_variables)
                        scores = model(x_np)
                        #print(scores)
                        loss = self.loss_fn(y_np, scores)
                        gradients = tape.gradient(loss, model.trainable_variables)
                        #print(gradients)
                        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        
                        # Update the metrics
                        train_loss.update_state(loss)
                        train_accuracy.update_state(y_np, scores)

                        
                        #
                        if t % 100 == 0:
                            val_loss.reset_states()
                            val_accuracy.reset_states()
                            for test_x,test_y in self.val_data:
                                val_scores = model(test_x)
                                prediction = tf.math.argmax(val_scores,axis=1)
                                loss_t = self.loss_fn(test_y,val_scores)
                                val_loss.update_state(loss_t)
                                val_accuracy.update_state(test_y,val_scores)   
                                self.logs['val_acc'].append(val_accuracy.result()*100)
                                self.logs['val_loss'].append(val_loss.result())
                                self.logs['train_acc'].append(train_accuracy.result()*100)
                                self.logs['train_loss'].append(train_loss.result())
                            if self.verbose:
                                template = 'Iteration {}, Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'     
                                print(template.format(t, str(epoch+1)+'/'+str(self.num_epochs),
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    val_loss.result(),
                                    val_accuracy.result()*100))
                    t+= 1
                #Save model  
                if self.save_mode:
                    if epoch% self.save_every_num_epoch == 0 or epoch == self.num_epochs:
                        if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                        save_path = os.path.join(self.save_dir,self.save_model_name+'_'+str(epoch))
                        tf.saved_model.save(model,save_path)
            if  self.plot_graph:
                self.plot_logs(self.logs)

class Dataset(object):
    def __init__(self,X,y,batch_size, shuffle= False) -> None:
        super().__init__()
        """
        Construct a Dataset object to iterate over data X and labels y
        
        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N,B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B],self.y[i:i+B]) for i in range(0,N,B))

    