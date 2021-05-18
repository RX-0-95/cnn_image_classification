from time import sleep, time

class TerminalBar(object):
    def __init__(self,name='Bar',max=50,length = 25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name 
        self.max =max
        self.length = length
        self.progress = 0 
        #default filler and spacer 
        self.filler = '='
        self.spacer = '*'
        self.bonder_l = ' |'
        self.bonder_r = '| '
        self.front_descriptor = ''
        self.back_descriptor = ''
    
    def update_front_descriptor(self, front_des):
        self.front_descriptor = front_des 
    def append_front_decriptor(self,app_front_des):
        self.front_descriptor += app_front_des

    def update_back_descriptor(self,back_des):
        self.back_descriptor = back_des
    def append_back_descriptor(self,app_back_des):
        self.back_descriptor += app_back_des
    @property 
    def filler_length(self):
        filler_len = int((self.progress/self.max) * self.length) 
        return filler_len

    @property
    def str(self):
        filler_len = self.filler_length
        str = self.front_descriptor + self.bonder_l +\
                self.filler*filler_len + self.spacer*(self.length-filler_len)+\
                self.bonder_r + self.back_descriptor
        return str 

    def next(self):
        """
        Add one to the progress and print to terminal 
        """
        self.progress += 1 
        #print(self.progress)
        #print(self.filler_length)
        if self.progress >= self.max:
            ender = ''
        else:
            ender = ''
        print('\r',self.str,end= '')
        

    def set_filler(self,filler):
        self.filler=filler 
    def set_spacer(self,spacer):
        self.set_spacer =spacer 
    
    def set_bounder(self,bonder_r,bonder_l):
        self.bonder_r = bonder_r
        self.bonder_l = bonder_l
    
    def finish(self):
        """
        Force the bar to end 
        """
        print('')
        
    
class KerasProgressBar(object):
    def __init__(self, max_iter, bar_length=50, *args, **kwargs):
        super().__init__()
        self.max_iter = max_iter 
        self.bar = TerminalBar(max=max_iter, name='KerasProgressBar')
        self.iter_counter = 0 
        self.front_des_len = 2*len(str(max_iter))+1
        self.train_acc = 'NA' 
        self.val_acc = 'NA'
        self.train_loss = 'NA' 
        self.val_loss = 'NA' 

    def write_iter_info(self):
        iter_info = str(self.iter_counter)+'/'+ str(self.max_iter)
        iter_info = iter_info.rjust(self.front_des_len)
        #update to Bar 
        self.bar.update_front_descriptor(iter_info)
    def update_train_acc_loss(self,train_acc,train_loss):
        self.train_acc = train_acc
        self.train_loss = train_loss
    def update_val_acc_loss(self,val_acc,val_loss):
        self.val_acc = val_acc 
        self.val_loss = val_loss
    def write_train_acc_loss_info(self):
        train_str = ' - train_acc: {:>6} - train_loss: {:>6}'.\
            format(self.train_acc,self.train_loss)
        self.bar.update_back_descriptor(train_str)
    def write_val_acc_loss_info(self):
        val_str = ' - val_acc: {:<5} - val_loss: {:<5}'.\
            format(self.val_acc,self.val_loss)
        self.bar.append_back_descriptor(val_str)
    
    #write info to bas descriptor  
    def write_info(self):
        self.write_iter_info() 
        self.write_train_acc_loss_info() 
        self.write_val_acc_loss_info() 
    def next(self):
        self.iter_counter+=1 
        self.write_info() 
        self.bar.next()

    def finish(self):
        self.write_info() 
        self.bar.finish() 
    
        
    #def update_front_descriptor(self):
        


if __name__ == '__main__':
    """
    for j in range(0 , 2):
        Bar = TerminalBar(max=20)
        for i in range(0,20):
            sleep(0.5)
            #print(Bar.progress)
            Bar.next()
    """
    for j in range(0,2):
        Bar = KerasProgressBar(max_iter=20)
        for i in range(0,20):
            Bar.update_train_acc_loss(1+i,2+i)
            sleep(0.5)
            Bar.next()
        Bar.finish()

    

