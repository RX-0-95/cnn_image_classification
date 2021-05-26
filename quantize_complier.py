
import copy
import enum
from os import terminal_size
from pickle import FLOAT, TRUE
from traceback import walk_stack
import numpy as np
from numpy.core.numeric import full
from numpy.random import shuffle
from tensorflow.python.keras.backend import dtype
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.ops.array_ops import sparse_mask 
import terminal_ui as tui 
import quantize_util as qu
import tensorflow as tf 
import collections
import solver as sl
FLOAT_BIT = 32


class DataShuffler(object):
    def __init__(self,X:np.array,y:np.array,sample_size=5000):
        super().__init__()
        assert X.shape[0] == y.shape[0], 'X and y have different size'
        self.data_len = X.shape[0]
        self.sample_size = sample_size 
        if self.sample_size > self.data_len:
            self.sample_size = self.data_len
        self.sample_idx = np.arange(self.data_len)
        self.X = X
        self.y = y 
    def sample(self,shuffle=False):
        if not shuffle:
            return self.X[self.sample_idx[0:self.sample_size]],\
                        self.y[self.sample_idx[0:self.sample_size]]
        if shuffle:
            np.random.shuffle(self.sample_idx)
            return self.X[self.sample_idx[0:self.sample_size]],\
                        self.y[self.sample_idx[0:self.sample_size]]
        
class LayerWlfl(object):
    def __init__(self,k_wl=0,k_fl=0,b_wl=0,b_fl =0) -> None:
        super().__init__()
        self.kernel_wl = k_wl
        self.kernel_fl = k_fl
        self.bias_wl = b_wl
        self.bias_fl = b_fl
    
    def __str__(self):
        #str = super().__str__() 
        str = ''
        str += "[{},{},{},{}]".format(self.kernel_wl,
                        self.kernel_fl,self.bias_wl,self.bias_fl)
        return str

class LayerAccLossPerBit(object):
    def __init__(self):
        super().__init__()
        self.kernel_apb = None
        self.bias_apb = None
    def __str__(self) -> str:
        str = ""
        str += "[{},{}]".format(self.kernel_apb,self.bias_apb)
        return str 

class LayerParamSize(object):
    def __init__(self):
        super().__init__()
        self.kernel_size = None 
        self.bias_size = None 
    def __str__(self):
        str = "[{},{}]".format(self.kernel_size,self.bias_size)
        return str 
class LayerList(object):
    def __init__(self,layer_num,list_type) -> None:
        super().__init__()
        self.layers_num = layer_num
        self.list = [] 
        for _ in range(self.layers_num):
            layer_info = list_type() 
            self.list.append(layer_info)
    
    def layer(self,num):
        return self.list[num]

    def __str__(self) -> str:
        str = super().__str__()
        str += "["
        for layer_info in self.list:
            str += layer_info.__str__()
        str += "]"
        return str 
    def __len__(self):
        return len(self.list)   
    def __iter__(self):
        return self.list.__iter__()  

class WlflList(LayerList):
    def __init__(self, layer_num, list_type=LayerWlfl) -> None:
        super().__init__(layer_num, list_type)
    
    def set_all_to(self,wl,fl):
        for each_wl_fl in self.list:
            each_wl_fl.kernel_wl = wl 
            each_wl_fl.kernel_fl = fl
            each_wl_fl.bias_wl = wl
            each_wl_fl.bias_fl = fl
    
    def save(self,file_name):
        f = open(file_name,'w')
        for layer_info in self.list:
            write_str = "{},{},{},{}".format(layer_info.kernel_wl,
                                            layer_info.kernel_fl,
                                            layer_info.bias_wl,
                                            layer_info.bias_fl)
            f.write(write_str+"\n")
        f.close()
    def load(self,file_name):
        f = open(file_name,'r')
        lines = f.readlines() 
        assert self.layers_num == len(lines), 'The layer size not match the size in {}'.format(file_name)
        for i,line in enumerate(lines):
            kernel_wl,kernel_fl, bias_wl,bias_fl = line.split(',')
            self.layer(i).kernel_wl = int(kernel_wl)
            self.layer(i).kernel_fl = int(kernel_fl)
            self.layer(i).bias_wl = int(bias_wl)
            self.layer(i).bias_fl = int(bias_fl)

class ParamSizeList(LayerList):
    def __init__(self, layer_num, list_type=LayerParamSize):
        super().__init__(layer_num, list_type)

class AccuracyLossPerBitList(LayerList):
    def __init__(self, layer_num, list_type=LayerAccLossPerBit) -> None:
        super().__init__(layer_num, list_type)
    
    #TODO: Figure out a way to sort the list 
    def get_smallest_apb(self):
        opt_apb = None 
        opt_layer = 0 
        opt_type = 'kernel'
        for i, layer_apb in enumerate(self.list):
            apb = layer_apb.kernel_apb
            if opt_apb is None:
                opt_apb = apb
            else:
                if apb < opt_apb:
                    opt_apb = apb
                    opt_layer = i 
                    opt_type = 'kernel'
            apb = layer_apb.bias_apb
            if apb is not None:
                if apb < opt_apb:
                    opt_apb = apb
                    opt_layer = i 
                    opt_type = 'bias'
        return opt_type, opt_layer,opt_apb

    def get_largest_compress(self,param_size_list:ParamSizeList,max_loss=0.01):
        max_compress = 0.0
        opt_layer = 0 
        opt_type = 'kernel'
        for i, layer_apb in enumerate(self.list):
            apb = layer_apb.kernel_apb
            layer_param = param_size_list.layer(i)
            kernel_size = layer_param.kernel_size
            layer_loss = apb*kernel_size
            if (layer_loss < max_loss) and kernel_size > max_compress:
                max_compress = kernel_size
                opt_layer = i 
                opt_type = 'kernel'
            apb = layer_apb.bias_apb
            if apb is not None:
                bias_size = layer_param.bias_size
                layer_loss = apb*bias_size
                if (layer_loss < max_loss) and bias_size > max_compress:
                    max_compress = bias_size
                    opt_layer = i 
                    opt_type = 'bias'
        return opt_type, opt_layer,max_compress
        


class FlSerachAgent(object):
    def __init__(self,model,tuner_fn,acc_fn) -> None:
        """
        model: nn model 
        tunner_fn: callback function holder to train the model for one batch input:(model)
        acc_fn: callback function holder to retun the accuracy of the model, input: (model)
        """
        super().__init__()
        self.model = model
        self.tune_callback = tuner_fn
        self.acc_callback = acc_fn
        
    def search_fl(self,data,wl,mid=None,max_fl=16,min_fl = -16):
        """
        Bisection serach
        return: best fl
        """
        #FIXME: auto gen max_fl and min_fl based on wl and data
        max_fl = max_fl
        min_fl = min_fl
        right = max_fl
        left = min_fl
        if mid ==None:
            mid = (right + left)//2
        midp = mid+1
        midm = mid-1 
        while (right - left) != 1:
            #print('midp: {} midm:{}'.format(midp,midm))
            valp = self.quantize_abs_mean_error(data,wl,midp)
            valm = self.quantize_abs_mean_error(data,wl,midm)
            if valm > valp:
                left = mid
            elif valm < valp:
                right = mid
            else:
                midp = min(midp+1,max_fl)
                midm = max(midm-1,min_fl)
                if midp == max_fl and midm == min_fl:
                    return mid
                continue
            mid = (right+left)//2
            midp = mid + 1 
            midm = mid-1
            #print('left:{} mid: {} right: {}'.format(left,mid,right))
        val_r = self.quantize_abs_mean_error(data,wl,right)
        val_l = self.quantize_abs_mean_error(data,wl,left)
        if val_l<val_r:
            return left
        else:
            return right

    def search_layers_fl(self,layers,wlfl_list:WlflList):
        """
        Find best fl for each layers, and assign to wlfl_list
        if no bias, then fl is 0 
        """
        assert len(wlfl_list)==len(layers), \
            "The layer number not compatiable with wlfl list length"
        for i,wlfl in enumerate(wlfl_list):
            opt_kernel_fl = self.search_fl(layers[i].kernel,wlfl.kernel_wl)
            wlfl.kernel_fl = opt_kernel_fl
            if layers[i].bias is not None:
                opt_bias_fl = self.search_fl(layers[i].bias,wlfl.bias_wl)
                wlfl.bias_fl = opt_bias_fl
            else:
                wlfl.bias_fl = 0 

    def serarch_layers_fl_and_get_accuracy(self,layers,wlfl_list:WlflList,shuffle=False):
        self.search_layers_fl(layers,wlfl_list)
        self.apply_wlfl_to_layers(layers,wlfl_list)
        return self.model_accuracy(shuffle)

    def initial_guess(self,layer_data,wl):
        return wl//2 

    def tunne_model(self,tunne_threshold = 3):
        return self.tune_callback(self.model)
    
    def model_accuracy(self,shuffle=False):
        loss, acc = self.acc_callback(self.model,shuffle) 
        return acc 
    def quantize_abs_mean_error(self,data,wl,fl):
        quantized_data= qu.to_fixpoint(data,wl,fl).numpy()
        error = np.mean(np.abs(data-quantized_data))
        return error

    def quantize_abs_mean_errors(self,data,wl,fls):
        errors = [] 
        for fl in fls:
            quantized_data= qu.to_fixpoint(data,wl,fl).numpy()
            error = np.mean(np.abs(data-quantized_data))
            errors.append(error)
        return errors 

    def _apply_wlfl_to_layer(self,layer,layer_wlfl:LayerWlfl):
        """
        Apply wl and fl to one layer's kernel and bias 
        """

        layer.kernel.assign(qu.to_fixpoint(layer.kernel,
                                layer_wlfl.kernel_wl,
                                layer_wlfl.kernel_fl))
        if layer.bias is not None:
            layer.bias.assign(qu.to_fixpoint(layer.bias,
                                layer_wlfl.bias_wl,
                                layer_wlfl.bias_fl))
        
    def apply_wlfl_to_layers(self,layers,wlfl_list:WlflList):
        assert len(wlfl_list)==len(layers), \
            "The layer number not compatiable with wlfl list length"
        for i,wlfl in enumerate(wlfl_list):
            self._apply_wlfl_to_layer(layers[i],wlfl)

    
        


class WLSearchAgent(object):
    def __init__(self,model,full_model,quantize_layers, wl_agent_opt = {},search_policy='BVL') -> None:
        super().__init__()
        self.model = model

        _opt = copy.deepcopy(wl_agent_opt)
        self.init_fl = _opt.pop('init_fl',6)    
        # Check if there is unexpected options 
        if len(_opt) >0:
            extra = ', '.join('"%s"' % k for k in list(self.options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra)
    
        self.full_mode = full_model
        #call back functio return the accuarcy of the model in % 
        #self.acc_callback = acc_fn
        self.quantize_layers = quantize_layers
        #parameter_size_list: 
        self.parameter_size,self.parameter_size_list = self.get_parameter_size()
        # store full precision trainable parameter: [[layer1_kerenl, layer1_bias], ..[] ]
        self.layers_param_list = self.extract_layers_trainable_variable()

        self.accuracy_loss_per_bit_list = AccuracyLossPerBitList(
                                        layer_num=len(self.quantize_layers))
    
        self.wl_list = WlflList(len(self.quantize_layers))
        self.wl_list.set_all_to(32,16)

    
    def init_search(self):
        """
        for each trainable variable in quantize_layers, get max of the parameter
        and calcuate the least interger bit
        """
        self.restore_model_weight() 
        for i,layer_param in enumerate(self.layers_param_list):
            kernel = layer_param[0]
            bias = layer_param[1]
            kernel_max= tf.math.reduce_max(tf.math.abs(kernel))
            #print(kernel_max)
            int_bits = self.get_intger_bits(kernel_max)
            kernel_wl = int_bits+self.init_fl
            self.wl_list.layer(i).kernel_wl = kernel_wl
            if bias is not None:
                bias_max = tf.math.reduce_max(tf.math.abs(bias))
                int_bits = self.get_intger_bits(bias_max)
                bias_wl = int_bits+self.init_fl
                self.wl_list.layer(i).bias_wl = kernel_wl
            else:
                self.wl_list.layer(i).bias_wl = 0 
            
    
    def fine_search(self,fl_agent:FlSerachAgent,acc_loss_threshold = 0.02,search_mode='loss_first'):
        self.restore_model_weight()
        accuracy_loss = 0.0 
        prev_layer = 0 
        prev_type = 'kernel'
        if search_mode == 'loss_first':
            while (accuracy_loss < acc_loss_threshold):
                init_acc = fl_agent.model_accuracy(shuffle=True)
                self.update_accuracy_loss_per_bit_list(fl_agent)
                var_type, layer_num,acc_loss_per_bit = self.accuracy_loss_per_bit_list.get_smallest_apb() 
                #calculate the accuracy loss update the wl list
                prev_layer = layer_num
                if var_type == 'kernel': 
                    #accuracy_loss += max(self.parameter_size_list.layer(layer_num).kernel_size*acc_loss_per_bit,0)
                    self.wl_list.layer(layer_num).kernel_wl -=1    
                    prev_type = 'kernel'
                elif var_type =='bias':
                    #accuracy_loss += max(self.parameter_size_list.layer(layer_num).bias_size*acc_loss_per_bit,0)
                    self.wl_list.layer(layer_num).bias_wl -= 1
                    prev_type = 'bias'
                else:
                    raise KeyError(
                        "Var type should be either kernel or bias but get {}".format(var_type))
                fl_agent.apply_wlfl_to_layers(self.quantize_layers,self.wl_list)
                model_acc = fl_agent.model_accuracy(shuffle=False) 
                accuracy_loss = init_acc - model_acc
                self.restore_model_weight()
                compression_rate = self.get_compression_rate()
                print("reduce wl on layer {} {},acc_loss: {}, current accuracy:{}, compression rate: {}"
                            .format(layer_num,var_type,accuracy_loss,model_acc,compression_rate))
                print(self.wl_list)            
                print(self.accuracy_loss_per_bit_list)
            """
            #restore the wlfl before exceeding the threshold 
            if prev_type == 'kernel':
                self.wl_list.layer(prev_layer).kernel_wl += 1 
            elif prev_type == 'bias':
                self.wl_list.layer(prev_layer).bias_wl -= 1 
            """
        elif search_mode == 'compress_first':
            while (accuracy_loss < acc_loss_threshold):
                init_acc = fl_agent.model_accuracy(shuffle=True)
                self.update_accuracy_loss_per_bit_list(fl_agent)
                var_type,layer_num,reduce_bit = self.accuracy_loss_per_bit_list.get_largest_compress(
                                                    self.parameter_size_list,max_loss=0.005)
                prev_layer = layer_num
                if var_type == 'kernel':
                    self.wl_list.layer(layer_num).kernel_wl -=1
                    prev_type = 'kernel'
                elif var_type =='bias':
                    self.wl_list.layer(layer_num).bias_wl -= 1
                    prev_type = 'bias'
                else:
                    raise KeyError(
                        "Var type should be either kernel or bias but get {}".format(var_type))
                fl_agent.apply_wlfl_to_layers(self.quantize_layers,self.wl_list)
                model_acc = fl_agent.model_accuracy() 
                accuracy_loss = init_acc - model_acc
                compression_rate = self.get_compression_rate()
                self.restore_model_weight()
                print("reduce wl on layer {} {},acc_loss: {}, current accuracy:{}, compression rate: {}"
                            .format(layer_num,var_type,accuracy_loss,model_acc,compression_rate))
                print(self.wl_list)
            """
            if prev_type == 'kernel':
                self.wl_list.layer(prev_layer).kernel_wl += 1 
            elif prev_type == 'bias':
                self.wl_list.layer(prev_layer).bias_wl -= 1             
            """ 

        else:
            raise KeyError('{} is not valid mode'.format(search_mode))
        self.restore_model_weight()
        return prev_type, prev_layer

    def update_accuracy_loss_per_bit_list(self,fl_agent:FlSerachAgent,increase_wl=False):
        self.restore_model_weight()
        ori_acc = fl_agent.model_accuracy(shuffle=True)
        #print('Original Acc: {}'.format(ori_acc))
        for i,wlfl in enumerate(self.wl_list):
            
            #print(self.wl_list)
            if increase_wl:
                wlfl.kernel_wl +=1
            else:
                wlfl.kernel_wl -=1
            #print(self.wl_list)
            acc = fl_agent.serarch_layers_fl_and_get_accuracy(
                                    self.quantize_layers,self.wl_list,
                                    shuffle=False)
            self.restore_model_weight()
            #acc_loss = max(ori_acc - acc,0) 
            acc_loss = ori_acc - acc 
            
            #print("layer acc: {}".format(acc))
            acc_loss_per_bit = acc_loss/self.parameter_size_list.layer(i).kernel_size
            self.accuracy_loss_per_bit_list.layer(i).kernel_apb = acc_loss_per_bit
            if increase_wl:
                wlfl.kernel_wl -=1
            else:
                wlfl.kernel_wl +=1
            if self.quantize_layers[i].bias is not None:
                if increase_wl:
                    wlfl.bias_wl +=1
                else:
                    wlfl.bias_wl -=1
                if wlfl.bias_wl == -1:
                    #FIXME: prevent wl reach negative, fix it properly in the future 
                    self.accuracy_loss_per_bit_list.layer(i).bias_apb = 1

                else:
                    acc = fl_agent.serarch_layers_fl_and_get_accuracy(
                                        self.quantize_layers,self.wl_list,
                                        shuffle=False)
                    self.restore_model_weight()
                    #acc_loss = max(ori_acc - acc,0)
                    acc_loss = ori_acc - acc
                    acc_loss_per_bit = acc_loss/self.parameter_size_list.layer(i).bias_size
                    self.accuracy_loss_per_bit_list.layer(i).bias_apb = acc_loss_per_bit
                if increase_wl:
                    wlfl.bias_wl -=1
                else:
                    wlfl.bias_wl +=1 
        
    def get_intger_bits(self,val):
        binary_val = '{0:b}'.format(int(val))
        return len(binary_val)+1

    def get_compression_rate(self):
        """
        use the data in self.wl_list calculate the compression 
        rate of the modle 
        return: #% of the paramenter comapre the full wl, size(in bits)
        """
        bits = 0 
        for i,layer_param_size in enumerate(self.parameter_size_list):
            kernel_size,bias_size = layer_param_size.kernel_size, layer_param_size.bias_size
            layer_wlfl= self.wl_list.layer(i)
            kernel_wl = layer_wlfl.kernel_wl 
            bias_wl = layer_wlfl.bias_wl 
            bits += int(kernel_wl*kernel_size + bias_wl*bias_size)
        compression_rate = bits/(self.parameter_size*FLOAT_BIT)
        return compression_rate,bits

    def get_parameter_size(self):
        """
        return: total_size, parameter_size_list_for each layer
        """
        param_size_list = ParamSizeList(len(self.quantize_layers)) 
        total_size = 0 
        #param_count_list = []
        for i,layer in enumerate(self.quantize_layers):
            kernel_size = layer.kernel.numpy().reshape(-1).shape[0]
            total_size += kernel_size
            #print(layer.bias)
            if layer.bias is not None: 
                bias_size = layer.bias.numpy().reshape(-1).shape[0]
                total_size += bias_size
            else:
                bias_size = 0 
            #param_count_list.append([kernel_size,bias_size])
            param_size_list.layer(i).kernel_size = kernel_size
            param_size_list.layer(i).bias_size = bias_size

        return total_size,param_size_list

    def extract_layers_trainable_variable(self):
        layers_param = [] 
        for layer in self.quantize_layers:
            kernel=  tf.identity(layer.kernel)
            #print(layer.bias)
            if layer.bias is not None: 
                bias = tf.identity(layer.bias)
            else:
                bias = None 
            layers_param.append([kernel,bias])
        return layers_param
           
    def restore_model_weight(self):
        """
        Restore the model weight to full precision
        """
        """
        assert len(self.quantize_layers)==len(self.layers_param_list),\
            "restore weight failed,length of paramter list should equal to layer length"
        for i,layer in enumerate(self.quantize_layers):
            layer.kernel.assign(self.layers_param_list[i][0])
            if layer.bias is not None:
                layer.bias.assign(self.layers_param_list[i][1])
        """
        qu.copy_weight(self.model,self.full_mode)
    def update_model_weight(self):
        qu.copy_weight(self.full_mode, self.model)

    def get_quantize_layers(self):
        return self.quantize_layers
    def get_wl_list(self):
        return self.wl_list
        

    
class QunatizeComplier(object):
    def __init__(self,quantize_model,full_precision_model,quantizable_layers,
                tunner_fn,acc_fn,verbose = True,max_loss = 0.02,target_compression=0.1):
        super().__init__()
        self.q_model = quantize_model
        self.full_model = full_precision_model
        self.quantize_layers = quantizable_layers
        self.wl_agent = WLSearchAgent(self.q_model,self.full_model,self.quantize_layers)
        self.fl_agent = FlSerachAgent(self.q_model,tunner_fn,acc_fn)
        self.verbose = verbose
        self.corase_search_loss_threshold = max_loss
        self.max_loss = max_loss
        self.target_compression_rate = target_compression
        self.tune_tril = 8
        self.fine_search_loss_step = max_loss/3

    def quantize_model(self):
        # init info
        self.wl_agent.restore_model_weight()
        init_acc = self.fl_agent.model_accuracy(shuffle=True) 
        if self.verbose:
            print("Full percision model accuracy: {}".format(init_acc))
        
        #coarse search 
        wlfl_list = self.wl_agent.get_wl_list() 
        if self.verbose:
            print('************Coarse search begin************')
        self.wl_agent.init_search()
        self.fl_agent.search_layers_fl(self.quantize_layers,wlfl_list)
        self.fl_agent.apply_wlfl_to_layers(self.quantize_layers,wlfl_list)
        #print(wlfl_list)
        corase_acc = self.fl_agent.model_accuracy() 
        self.wl_agent.restore_model_weight() 
        if self.verbose:
            print('Coarse seach done, compression rate:{},quantize model accuracy: {}'.format(
                                self.wl_agent.get_compression_rate(),corase_acc))
        
        #tunne model
        
        if init_acc - corase_acc > self.corase_search_loss_threshold:
            tunned_acc = self.tune_model(init_acc,max_loss=self.corase_search_loss_threshold)
            if tunned_acc: 
                if self.verbose:
                    print('qunatize model accuracy after tunned: {}'.format(tunned_acc)) 
            else:
                if self.verbose:
                    print('qunatization end after corase seach')
                return self.q_model, self.wl_agent.get_wl_list() 
    
        if self.verbose:
            print('\n************Fine search begin************\n')
        
        ####load list : Delete later
        #self.wl_agent.get_wl_list().load('complier_tmp/3deep_compressed_resnet20_wlfl.txt')
        ####

        #fine search 
        fine_search_end = False 
        while not fine_search_end:
            print('In compression search+++++++++++++++++++++++++++')
            last_type,last_layer = self.wl_agent.fine_search(fl_agent=self.fl_agent,
                                acc_loss_threshold=self.fine_search_loss_step,
                                search_mode='compress_first')
            
            compress_rate,total_bits =  self.wl_agent.get_compression_rate() 
            print('Compression rate:{}'.format(compress_rate))
            print('Traget Compression rate:{}'.format(self.target_compression_rate))

            if compress_rate < self.target_compression_rate:
                print("----------Ahieve compress rate ---------")
                fine_search_end = True
                break
            else:
                if self.verbose:
                    print('------Fine tune the model------')
                tune_acc = self.tune_model(init_acc,self.max_loss)
                if tune_acc:
                    pass 
                else:
                    fine_search_end = True
                    #restore the prev wlfl before the acc unrestorable
                    print("!!!!!Can't Compress Any More!")
                    if last_type == 'kernel':
                        self.wl_agent.get_wl_list().layer(last_layer).kernel_wl +=1
                    elif last_type == 'bias':
                        self.wl_agent.get_wl_list().layer(last_layer).bias_wl +=1

        print("Start loss search++++++++++++++++++++++++++++")
        fine_search_end = False 
        while not fine_search_end:
            last_type,last_layer = self.wl_agent.fine_search(fl_agent=self.fl_agent,
                                acc_loss_threshold=self.fine_search_loss_step,
                                search_mode='loss_first')
            
            compress_rate,total_bits =  self.wl_agent.get_compression_rate() 
            if compress_rate <= self.target_compression_rate:
                fine_search_end = True
                break
            else:
                if self.verbose:
                    print('------Fine tune the model------')
                tune_acc = self.tune_model(init_acc,self.max_loss)
                if tune_acc:
                    continue
                else:
                    fine_search_end = True
                    #restore the prev wlfl before the acc unrestorable
                    print("!!Loss seach end !!")
                    if last_type == 'kernel':
                        self.wl_agent.get_wl_list().layer(last_layer).kernel_wl +=1
                    elif last_type == 'bias':
                        self.wl_agent.get_wl_list().layer(last_layer).bias_wl +=1
        
        
        
        #final tunne 
        self.fl_agent.search_layers_fl(self.quantize_layers,self.wl_agent.get_wl_list())
        if self.verbose:
            print('=========Finial Tunne=========')
        tune_acc = self.tune_model(init_acc,self.max_loss/10)

        return self.q_model, self.wl_agent.get_wl_list()
        

    def tune_model(self,init_acc,max_loss):
        self.wl_agent.restore_model_weight()
        for _ in range(self.tune_tril):
            self.fl_agent.tunne_model()
            qu.copy_weight(self.full_model,self.q_model)
            tunned_acc = self.apply_wlfl_and_get_accuracy()
            if init_acc-tunned_acc <= max_loss:
                self.wl_agent.restore_model_weight()
                return tunned_acc 
        return False

    def apply_wlfl_and_get_accuracy(self):
        #self.wl_agent.restore_model_weight()
        self.fl_agent.apply_wlfl_to_layers(self.quantize_layers,
                                self.wl_agent.get_wl_list())
        acc = self.fl_agent.model_accuracy()
        #self.wl_agent.restore_model_weight()
        return acc 
    
    def save_quantize_model(self):
        pass