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
    - Model 
    - WL for each layer (list)

* output: 
    - FL for each layer (list)


fl_search range: -16-
recorder(dict) to record each layer's opt fl for 


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
