
import numpy as np 
import terminal_ui as tui 
import quantize_util as qu
import collections

class Recorder(object):
    def __init__(self) -> None:
        super().__init__()

class FlSerachAgent(object):
    def __init__(self,model,tuner_fn,acc_fn) -> None:
        """
        model: nn model 
        tunner_fn: callback function holder to train the model for one batch 
        acc_fn: callback function holder to retun the accuracy of the model 
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

    def initial_guess(self,layer_data,wl):
        return wl//2 

    def tunne_model(self,tunne_threshold = 3):
        self.tune_callback()
    
    def model_accuracy(self):
        return self.acc_callback() 


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



class WLSearchAgent(object):
    def __init__(self,model,init_wl = 32) -> None:
        super().__init__()
        self.model = model
        

    
