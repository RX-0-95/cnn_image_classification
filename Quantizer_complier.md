# Qunatizer_complier

## Search Alogrithm 

### Alog1: Exhaustive search (binary search)
Change the WL for each layer 
Record the accuracy change
Find out with layer has higher percision prority
* Shallow Serach 

* Deep Search 


## Give WLS Module 

### Give WLs, find FLs 
        fl_search(model,WL)
use least abs mean search for FLs, use bisection search 
max fl = 32-wl
min fl = -32-wl 

* input: 
    - Model (full percision)
    - WL for each layer (list)

* output: 
    - FL for each layer (list)

### Give all qunatize layers and find fls, and assgin to WlflList

        search_layers_fl(self,quantize_layers,wlfl_list)


fl_search range: -16-
recorder(dict) to record each layer's opt fl for 

### Apply wl and fl to the model weights 
        apply_wlfl_to_layers()

### Retrain the model for one batch (?)


## WL Search Module
### Course search 
1. Decrease the wl of each layer so the total test acc will not decrease by 1%(course threshold)

#### setp:
1. Initialize wl: 
    for each var: get the max abs element, and apply get bit number that can get satisfy the max number 
    eg. max var: 128.999 
    2^7 = 128 -> need (7+1) bit
Give the initialize floting point be 6-> initial wl: integer bit + fl bit 

2. 
Replace the wl of each layer spearately use the initialze wl 
if the precision change is greate than (course thresold: 1%) increase the wl of the layer and try until satisfy. 
The best fl is found by 

3. Apply wl to all layers, 


### Fine search 
use list to store acc_loss_vs_bit for each layers 

1. Init acc_loss per bit list 
get_model_acc use fl agent
for all layers:
    decrease the wl of the layer by 1 
    pass to fl_agent, serach for best fl 
    get model accuracy loss 
    calculate acc_loss per bit and write to list

3. Acc loss threshold calculator 
    Native: hard threshold 
    Advnace: compression rate and acc_loss

2. 
while True (some accuracy loss threshold):
   a.  get smallest acc_loss_per_bit: kernel or bias 
    use param_size calc acc_loss

    b. if greater than threshold, find second smallest. so on 

    c. reduce wl of layer(kernel/bias) by 1

    d. pass to fl update the acc_loss_per_bit list. 
    
    e. calc compression rate

    e. repeat until loss threshold reached 


### Retrain the model 
Create dynamic list: record acc/bit lost, and compress the layer with least acc/bit lost, and update the list.



1. Run weight length from 

* input: 
    - model
    - tunner: Callback function to train (tunner) the model 
    - test_data set 
    - WL 
    - FL 
* output:
    - Restored Accuarcy, Original accuracy
    - Model size


## Helper functiomn 
1. 
                quantize_var(var,wl,fl)
var: trainable variabel : (kernel or bias)
wl: word length 
fl: flaoting lenght 

if the size of the var is 0, then return do nothing 
if not assgn the wl and fl to the var 


# Report 

## Introduction 1/4

## Qunatization method 

### Tech 1/4
### Layer 1/4

## Search Heruistic 
### FL agent: 1/2
### WL ageent: 1/2 

## Result: 1/2
## Discussion: 1/4
