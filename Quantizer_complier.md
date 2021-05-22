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



fl_search range: -16-
recorder(dict) to record each layer's opt fl for 
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

