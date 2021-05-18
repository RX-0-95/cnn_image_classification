## Intro 
Intel NX with in-fabric Ai tensor blocks able to reach peak performance 143 int8 TOPS. (14 nm FPGA)

12 nm Nvidia GPUs T4 with tensor core: 130 int8 TOPS

The actual performance depends only only on the peak performance, but the entire work flow. 

This paper provide a method to evaluation of Intel's AI-optimized FPGA stratix 10 NX. Use the method compare the performance of NV T4 V100 to the Intel FPGA.

Tool Chain: allow user use pure software to program FPGA AI tensor blocks without FPGA EDA tools in the loop. 

Result: 24X faster than T4, 12X faster then V100 at batch6

Advantage: 100Gbps ethernet allow for remote access to 10x and 2x less system overhead latency then local access to V100 GPU vis 128Gbps PCIe for short and long sequence RNN. 

* Start:
Hello everyone, 


## intrp
GPU is cool and useful. tera operation per second (TOPS) is increasing thanks for the high demanding. Some are even higher than 820 int8 TOPS. 

Problem with the peak TOPs measure:
* peak performance is not always attainable, in most of the time, the system is not operate at it's peak performance. 

Utilization of Tensor: (affect by 2 fact)
* mapping of given workload to the avaliable compute units 
* end to end data transfer overhead bring data in and out the chip 

Actual achievable performance is important 
In this paper the Intel team develop a method to compare the performance of their Stratix 10 NX AI-optimized FPGA with Nvidia's AI-optmized GPUs
###V

 Setup for the evaluation 
* enhance the Microsft;s Brainwave NPU: a state-o-art commeical AI soft procesor and make it efficiently use the testr blocks in Startix 10NX 
* Compare to baseline NPU implemented use previous gen product stratix 10 GX/MX which has no tensor blocks. 
* Compare to Nvidia GPUS with tensor cores, T4 and V100. 
in real-time DL workload 
### test networks:
MLPS , RNNs, GRUs, LSTMs 

## Backgroud 
90% of Google DL datacenter workload in 2017 are MLPS and RNNS. 
MLPs: Staked fully connected layers (graph), most simply and widely used in the DL. 

RNNs: Sequence inputs as sentences. Number of matrix multiplications followed by vector element-wise operations. Most use in NLP.
    vanilla RNNs, LSTM, GRUs. 
    (Fromulas)
Those are high frequecy operation is those RNNs, as we can see from those opertations, most of them are vector to matrix multiplication. Later I will show that the Stratix NX10 FPGA is design to be good at the vector matrix multiplication. 



### NPU architecture and Toolchain 
Brainwave NPU: Design by Microsoft target for low-latency batch-1 DL inference. 
The basedline NPU is re-implementation of the MS's Brainwave NPU, with the modification on int8 precision instead of MS's custom flaoting point formats. (graph)
Five main parts:
1. Matrix-vector multiplicartion unit(MVU): T tiles followed by an inter-tile adder reduction tree; Each tile contains a vector register file (VRF) to store the input vectors and D dot production engines (DPEs)
DPEs tightly coupled with matrix register file (MRF), stores the model weight. 

The external Vector register file (eVRF) enables skipping the MVU for instruction that do not have a matrix-vector operation. eVRF is followed by two multi-function units (MFUs) 

Multi-function units (MFUs): element-wise activation. addition and multiplication.

Loader (LD): communicates with the outside world via the input/output FIFOs, and can write back the pipeline results to any of the architecture's VRFs


The team also implement a complete NPU software toolchain so that developer can write NPU programs in domain-specific laguage (DSL) (figure). A simulator of the NPU is also avaliable, so developers can rapidly experiment with different NPU programs without actual complied FPGA or use FPGA CAD tools. 


## Startix 10 NX; AI optimized FPGA 
Advantage: 


### Startix 10 NX AI Tensor Block
* New tensor block introduces data reuse register banks, enable fitting a significantly higher number of int8 multipliers than any prior work, while keep same number of input output port of DSP block. 
* In short, the data tranfer overhead inisde the NPU reduces alot. 
It also 
* Support various numberical format such as int8/int 4, single-precision floating point (fp32), blocking floating point and brain floaing point (bfloat24/bfloat16). This paper fouce in the int8 tesnsor mode. 

Now lets take look of it tensor block
On the left is the sketch of the tensor block in Statix NX10. 

* Tensorblock (graph Fig2)
- 3 dot product units, each with 8x8 multipliers, and three optional accumulators. 
- Two banks of ping-pong data reuse registers used to store operands and can be populated throught either the block's data inout prot or dedicated chain from the tensor block. (data feed)
- One tensort block at the begining of the chain is used as input bypass to load inputs to the ping-pong register chain of the tensor blocks below. (cascade In)
- The reset of the chain each dot product recieve (30) operands from the previous tensorblock, and 10 operands directly from the input and broadcast to all three dot product. 
- When first bank is feeding operands to the dot product, the second bank can be loaded from the previous tensor. 
- (fig3) Color circle represent data shift in ping-pong registers. uncolored shapes are input vectors to the dat port of the tensor block being brocast the all the dot product unit. 
- Dash box: dot product unit is doing it's job. 
- The dot product unit is busy all the time. (key point)


## Enhanced NPU for STRATIX 10 NX
### NPU Architecture and Toolchain Enhancements 
* Matrix-vector Multiplication Mappting: (fig4 (a))
  Matrix vector multiplication unit(MVU), we need use 2 Dot produce engine to to do that. The matrix split into T column blocks, each tile is responsible for one column block. 
* Inroder to use the new tensor blocks in Stratix 10 NX, rember, each tensor block has 3 dot product engine. The intel team reorganize the NPU's MVU. load blocks of 3 different input vectors (to feed 3 dot product unit) to one of the register bank and reuse them across enough matrix rows to hid the latency of loading the next 3 blocks of the input, some when dot product unit is perform dot product, the other bank register is load data.

* One problem of this data flow is the we need a place to store the output of the DPEs, and add them together latter 

#### Accululator Design:
The acccululator are no longer simple register and adder, the solution is a BRAM-based accumulator. (fig 4c). The precision of the adder in done in int32 precision. Bascially they store the partial result in the BRAM and add the result together later. 

#### Daisy Chain Tiles:


#### ISA Toolchain 
As we see that the NPU is better when perform batch-3 operations, the team design new instructions for the batch-3 operations. The instruction now contain 3 fields for storing the address of the 3 operatnd vector instead of 1, so it's more efficent in instruction set. Their C++ performance simulator also modified ot reflect the new architecture, allow accurate performace estimation for developers. Their complier automatically take care of the operation mapping, so developer does not need to care about the low-level control sequencing for the tensor blocks. 


### Implementation Results 
(Table I) shows the implementation of Straix 10 NX device compared th MX2100 and GX2800 with no tensor blocks. As we can see from the Table, the Straix 10NX has much higher multiplier count compared to other two. 3600 tensor blocks are used by the NPU on NX, only 2240(57%) are used in tensor mode by MVU. The remaining tensor blocks are used to implement the MFU;s int32 element-wise multiplication (800 blocks) and (560)blocks are used as input bypass to feed the data reuse registers to the tesor block chains. 


Table II show the batch-6 latency of the NPU on Tratix 10NX  device on real-time DL workloads. The latency is measured at the cycle starting from consuming the inputs for the input, excuting all NPU instasuctions and writting the outputs the output. The result shows that the performance of the NPU increase when the problem size increase and reach 32.4 TOPs on RNN with size 1792. The latency are very low, the highest latency for GRU workload at 256 step size takes only 1.1ms. GRU are ususally used in Nueraul language processing, the average sentence lenght in Engilish is 15-20 words, which means the NPU can run inference over 11,6000 - 15,500 sentences per second. 

## Core Compute Becnh mark 
It's not that persvice to see how good is the product without compare with it's competitors. Team intel, for course has their enhanced NPU on the Stratix 10NX AI-optimized FPGA against Team Nvidia's T4 and V100, T4 and V100 are NVs AI optimized GPUs. T4 has 320 tensor core and V100 has 640 tensor cores. 

### Square matrix-matrix multiplication (GEMM). 

The reseach team perfrom GPU-micro benchmark use square matrix-matrix multilication, each device is updaed to the latest stable library. On Nvidia deviece, the reserach team use Nvidia's offical and highly optimized cuDNN kernel for the DL workload. To make the game fare, they exhaustively run all possible configuration for NVdiva device in precision, tensor core setting, compute style, and peak the best achieved performance to compare to the NPU on Stratix 10NX. 

(fig6) shows the GEMM benchmark of T4 and V100. The red solid line represnt the performance when tensor core is on, and the blue line is tensor core disable. It is clear that the tensor core siginificaly increase the performance however, in both V100 and T4, the tensor core are not close to their peak performance. One worth to noticed that, the tensor core on T4 and V100 does not support fp32 percision, instead fp32 data is conveted into f16 before executing. This overhead of the data conversion decasse the tesnor core performance, so we can see that the tensor core has much higher performance when it's deal with fp16 data compare the fp32. 

Beacuse the tesnsor on the Straitix NX 10 is desinged for vecto to matrix multiplication
The team perform GEMM bechmark on NPU by keeping one matrix persiostent on-chip and streaming in the ohter matrix as squence of row vectors. The plot here compre the NPU performace on Stratix 10NX to the T4 GPU with int8 tesnsor cores. The team claim that despite thier tensor blocks are design for matrix vector operation, they still get similar performance in matrix to matrix operator compare to T4. 

#### Preformance Compariosn on AI workloads 
(Fig 7) Compare the NPU, T4,V100's real-time DL workload. The read line is T4, green line is V100 and the blue line is NPU. As we see from the graph, the performace of the NPU is much higher than V100 and T4 when the batch size is small. The NPU has it's best performance when batch is 6, which make sense beacuse the tensor core has 3 MVU, and 2 cores work at same time. At batch-6, the performance of NPU is 24.2x and 11.7 times higer than T4 and V100. Also, NPU performe better when the bath size can be divisable by 6. At high batch, The NPU aslo perform better or equvalent to T4 and still competable to V100.  

The right top plot here shows the geomean unilization of the NPU compared to both GPU at different batch size. The NPU reach 37.1% utiliztion compare to 1.52% and 3% of utiliztion of T4 and V100. The reason that the GPU has such low unitliztion at low batch is beacause their tensor cores are not diectly connected to each other. Each GPU tensor core has to send it's patrial result to a gloabal memory and synchronize wit other tensor cores to get combine result. Other operations require the combined result such as activation need to read the result from the global memory, thus higher the bactch size, high bath size can amortize the overhead, but even at batch size of 256, the GPU utilization is still only 13.3% and 17.8% in T4 and V100. On the other hand, the dasiy chain connection of the tensor blocks in the NPU allow partial result directly pass the next tensor blocks, allevaiting the need to communcation through memory. 

### System-level Behchamarking 
Compare the end to end system performance use NPU and GPU.

GPU base inference consist the host CPU connect to GPU card via PCIe interface. 

On the NUP side, a embedded ethernet interface drectily recieve inputs from remote client. The spec ofth machine they use in the test are show in this figure. (Table IV)

#### End to End Ai inference Application 

The End-to-End sysetm is the entire work flow of the DL inference Application. It consisit One-time intiliation, Prepare inputs, Send input to accelerator, Accelerator execution, send back result. The FPGA sysetm use ethernet as connection to host, threfore the host CPU needs to access its NIC (Network interface card) with optimizaed software libaries. 

On the Nvida side, the team use NV's recommanded method for intercation between CPI and GPU inorder to reach best performance. 

#### Data movement Efficiency Characterization 
Data transfer is insentail in the DL inference, low speed data trasnfer can be a bottle neck for the DL inference even with powerful acceleration device. The Stratix 10NX board use 100G enternet communicate to CPU. To evaluate the data movement efficiency, the team measure the host-to-deivce and device-to-host data transfer performances separately and take the average. the result shown in (FIG8). The V100 GPU achieve highst bandwidth since the PCIe interface proivde higer peak bandwidth. the FPGA achieves the best utiliztion of its up to 90% of the peak bandwidth of the 100G ethernet, where V100 only utilize up to 80% of its peak 128 Gbps PCIe bandwidth. 

#### End-to-end performance Comparsion on AI workloads. 
The figure 9 shows the system-level execution time of RNN workloads, the x axis are the size of RNN, and y axis is the Execttion time in milli seconds. The FPGA ssytem has above 15 times faster than GPU system in RNN with short sequence RNN and above 5 times after with long sequence RNN. 

### Powe Analysis
GPU power is measure with Nv's SMI tool and FPGA is measure with high resolution power memter. The T4 power is in the range of 27-45w and the V100 in 35-72W. When GPUs running worlad thath achiveves high utilization, T4 goes up to 70W and V100 goes up to 190W. 

On the Stratix 10NX, the power range is 54-70W without aby speical colling solutions.Noticed that in same amount of time, the FGPA does much more DL workload compare to T4 and V100. 
As a result the Stratix 10NX NPU running at batch-6 inference achieves 12-16x and 8x-12 times higher average energy efficiency than T4 and V100 respectivley. 

## Conclusion
As a conclusion, the team presented the first evaluation of the performance of Itenl's AI-optimized Startix 10NX FPGA with tensor blocks compared to latest AI-optimized GPUs. They use their FPGA implement Brainwave NPU and achieves 3.5x performance then the baseline NPU without tesor block on Stratix 10MX. In common DL workload, the utilization of the NPU reach 80%. The NPU at batch-6 delievers 24x 12x higher core cpmpute perfoprmance on averge compare the T4 and V100 respectivly. At higer batch size such as 256, the NPU sill achieves 58% higher perfirnace than T4 and only 30% less than V100. Finally they build an ene-to-end system for NPU and GPUs and compare their AI inference. The result show that the FPGA;s intergrage 100G enterhnet result int 10x and x times less overhead compared to the 128Gnps PCIe interface on the V100 GPU for RNN worrload. 