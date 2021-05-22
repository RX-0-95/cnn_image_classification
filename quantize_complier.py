
import copy
import enum
from pickle import FLOAT
from traceback import walk_stack
import numpy as np
from numpy.core.numeric import full 
import terminal_ui as tui 
import quantize_util as qu
import tensorflow as tf 
import collections
FLOAT_BIT = 32
class Recorder(object):
    def __init__(self) -> None:
        super().__init__()

class LayerWlfl(object):
    def __init__(self,k_wl=0,k_fl=0,b_wl=0,b_fl =0) -> None:
        super().__init__()
        self.kernel_wl = k_wl
        self.kernel_fl = k_fl
        self.bias_wl = b_wl
        self.bias_fl = b_fl
  
class WlflList(object):
    def __init__(self,layer_num) -> None:
        super().__init__()
        self.layers_num = layer_num
        self.wlfl_list = [] 
        for _ in range(self.layers_num):
            layer_wl_fl = LayerWlfl() 
            self.wlfl_list.append(layer_wl_fl)
    
    def layer(self,num):
        return self.wlfl_list[num]
    def set_all_to(self,wl,fl):
        for each_wl_fl in self.wlfl_list:
            each_wl_fl.kernel_wl = wl 
            each_wl_fl.kernel_fl = fl
            each_wl_fl.bias_wl = wl
            each_wl_fl.bias_fl = fl
    def __str__(self) -> str:
        str = super().__str__()
        for wl_fl in self.wlfl_list:
            str += '[{},{},{},{}]'.\
                    format(wl_fl.kernel_wl,
                        wl_fl.kernel_fl,wl_fl.bias_wl,wl_fl.bias_fl)
        return str 
    def __len__(self):
        return len(self.wlfl_list)   
    def __iter__(self):
        return self.wlfl_list.__iter__()  


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
        
    def search_fl(self,data,wl,mid=None):
        """
        Bisection serach
        return: best fl
        """
        #FIXME: auto gen max_fl and min_fl based on wl and data
        max_fl = 32
        min_fl = -32
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

    def initial_guess(self,layer_data,wl):
        return wl//2 

    def tunne_model(self,tunne_threshold = 3):
        self.tune_callback(self.model)
    
    def model_accuracy(self):
        return self.acc_callback(self.model) 

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
    def __init__(self,model,quantize_layers, wl_agent_opt = {},search_policy='BVL') -> None:
        super().__init__()
        self.model = model

        _opt = copy.deepcopy(wl_agent_opt)
        self.init_fl = _opt.pop('init_fl',6)    
        # Check if there is unexpected options 
        if len(_opt) >0:
            extra = ', '.join('"%s"' % k for k in list(self.options.keys()))
            raise ValueError('Unrecognized arguments in options%s' % extra)
    
       
        #call back functio return the accuarcy of the model in % 
        #self.acc_callback = acc_fn
        self.quantize_layers = quantize_layers
        #parameter_size_list: [[kenel_size, bais_size] ... []]
        self.parameter_size,self.parameter_size_list = self.get_parameter_size()
        # store full precision trainable parameter: [[layer1_kerenl, layer1_bias], ..[] ]
        self.layers_param_list = self.extract_layers_trainable_variable()
    
        self.wl_list = WlflList(len(self.quantize_layers))
        self.wl_list.set_all_to(32,16)

    
    def init_search(self):
        """
        for each trainable variable in quantize_layers, get max of the parameter
        and calcuate the least interger bit
        """
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
            kernel_size,bias_size = layer_param_size
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
        total_size = 0 
        param_count_list = []
        for layer in self.quantize_layers:
            kernel_size = layer.kernel.numpy().reshape(-1).shape[0]
            total_size += kernel_size
            #print(layer.bias)
            if layer.bias is not None: 
                bias_size = layer.bias.numpy().reshape(-1).shape[0]
                total_size += bias_size
            else:
                bias_size = 0 
            param_count_list.append([kernel_size,bias_size])
        return total_size,param_count_list

    def extract_layers_trainable_variable(self):
        layers_param = [] 
        for layer in self.quantize_layers:
            kernel= layer.kernel
            #print(layer.bias)
            if layer.bias is not None: 
                bias = layer.bias
            else:
                bias = None 
            layers_param.append([kernel,bias])
        return layers_param
           
    def restore_model_weight(self):
        """
        Restore the model weight to full precision
        """
        assert len(self.quantize_layers)==len(self.layers_param_list),\
            "restore weight failed,length of paramter list should equal to layer length"
        for i,layer in enumerate(self.quantize_layers):
            layer[i].kernel.assign(self.layers_param_list[i][0])
            if layer[i].bias is not None:
                layer[i].bias.assign(self.layers_param_list[i][1])
            
             

    def get_quantize_layers(self):
        return self.quantize_layers
    def get_wl_list(self):
        return self.wl_list
        

    
