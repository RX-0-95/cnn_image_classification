??	
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
?
lennet5_1/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelennet5_1/conv2d_2/kernel
?
-lennet5_1/conv2d_2/kernel/Read/ReadVariableOpReadVariableOplennet5_1/conv2d_2/kernel*&
_output_shapes
:*
dtype0
?
lennet5_1/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelennet5_1/conv2d_2/bias

+lennet5_1/conv2d_2/bias/Read/ReadVariableOpReadVariableOplennet5_1/conv2d_2/bias*
_output_shapes
:*
dtype0
?
lennet5_1/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namelennet5_1/conv2d_3/kernel
?
-lennet5_1/conv2d_3/kernel/Read/ReadVariableOpReadVariableOplennet5_1/conv2d_3/kernel*&
_output_shapes
: *
dtype0
?
lennet5_1/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namelennet5_1/conv2d_3/bias

+lennet5_1/conv2d_3/bias/Read/ReadVariableOpReadVariableOplennet5_1/conv2d_3/bias*
_output_shapes
: *
dtype0
?
lennet5_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?x*)
shared_namelennet5_1/dense_3/kernel
?
,lennet5_1/dense_3/kernel/Read/ReadVariableOpReadVariableOplennet5_1/dense_3/kernel*
_output_shapes
:	?x*
dtype0
?
lennet5_1/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*'
shared_namelennet5_1/dense_3/bias
}
*lennet5_1/dense_3/bias/Read/ReadVariableOpReadVariableOplennet5_1/dense_3/bias*
_output_shapes
:x*
dtype0
?
lennet5_1/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xP*)
shared_namelennet5_1/dense_4/kernel
?
,lennet5_1/dense_4/kernel/Read/ReadVariableOpReadVariableOplennet5_1/dense_4/kernel*
_output_shapes

:xP*
dtype0
?
lennet5_1/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*'
shared_namelennet5_1/dense_4/bias
}
*lennet5_1/dense_4/bias/Read/ReadVariableOpReadVariableOplennet5_1/dense_4/bias*
_output_shapes
:P*
dtype0
?
lennet5_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P
*)
shared_namelennet5_1/dense_5/kernel
?
,lennet5_1/dense_5/kernel/Read/ReadVariableOpReadVariableOplennet5_1/dense_5/kernel*
_output_shapes

:P
*
dtype0
?
lennet5_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_namelennet5_1/dense_5/bias
}
*lennet5_1/dense_5/bias/Read/ReadVariableOpReadVariableOplennet5_1/dense_5/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*? 
value? B?  B? 
?
	conv1
	conv2
maxpool
flatten
fc0
fc1

fc_out
softmax
	
activation

regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
 trainable_variables
!	variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
R
5regularization_losses
6trainable_variables
7	variables
8	keras_api
R
9regularization_losses
:trainable_variables
;	variables
<	keras_api
 
F
0
1
2
3
#4
$5
)6
*7
/8
09
F
0
1
2
3
#4
$5
)6
*7
/8
09
?

regularization_losses
=layer_metrics
>layer_regularization_losses
trainable_variables
?metrics
	variables

@layers
Anon_trainable_variables
 
VT
VARIABLE_VALUElennet5_1/conv2d_2/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElennet5_1/conv2d_2/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
Blayer_metrics
Clayer_regularization_losses
trainable_variables
Dmetrics
	variables

Elayers
Fnon_trainable_variables
VT
VARIABLE_VALUElennet5_1/conv2d_3/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElennet5_1/conv2d_3/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
Glayer_metrics
Hlayer_regularization_losses
trainable_variables
Imetrics
	variables

Jlayers
Knon_trainable_variables
 
 
 
?
regularization_losses
Llayer_metrics
Mlayer_regularization_losses
trainable_variables
Nmetrics
	variables

Olayers
Pnon_trainable_variables
 
 
 
?
regularization_losses
Qlayer_metrics
Rlayer_regularization_losses
 trainable_variables
Smetrics
!	variables

Tlayers
Unon_trainable_variables
SQ
VARIABLE_VALUElennet5_1/dense_3/kernel%fc0/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElennet5_1/dense_3/bias#fc0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
?
%regularization_losses
Vlayer_metrics
Wlayer_regularization_losses
&trainable_variables
Xmetrics
'	variables

Ylayers
Znon_trainable_variables
SQ
VARIABLE_VALUElennet5_1/dense_4/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElennet5_1/dense_4/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
+regularization_losses
[layer_metrics
\layer_regularization_losses
,trainable_variables
]metrics
-	variables

^layers
_non_trainable_variables
VT
VARIABLE_VALUElennet5_1/dense_5/kernel(fc_out/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElennet5_1/dense_5/bias&fc_out/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
?
1regularization_losses
`layer_metrics
alayer_regularization_losses
2trainable_variables
bmetrics
3	variables

clayers
dnon_trainable_variables
 
 
 
?
5regularization_losses
elayer_metrics
flayer_regularization_losses
6trainable_variables
gmetrics
7	variables

hlayers
inon_trainable_variables
 
 
 
?
9regularization_losses
jlayer_metrics
klayer_regularization_losses
:trainable_variables
lmetrics
;	variables

mlayers
nnon_trainable_variables
 
 
 
?
0
1
2
3
4
5
6
7
	8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lennet5_1/conv2d_2/kernellennet5_1/conv2d_2/biaslennet5_1/conv2d_3/kernellennet5_1/conv2d_3/biaslennet5_1/dense_3/kernellennet5_1/dense_3/biaslennet5_1/dense_4/kernellennet5_1/dense_4/biaslennet5_1/dense_5/kernellennet5_1/dense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_1548297
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-lennet5_1/conv2d_2/kernel/Read/ReadVariableOp+lennet5_1/conv2d_2/bias/Read/ReadVariableOp-lennet5_1/conv2d_3/kernel/Read/ReadVariableOp+lennet5_1/conv2d_3/bias/Read/ReadVariableOp,lennet5_1/dense_3/kernel/Read/ReadVariableOp*lennet5_1/dense_3/bias/Read/ReadVariableOp,lennet5_1/dense_4/kernel/Read/ReadVariableOp*lennet5_1/dense_4/bias/Read/ReadVariableOp,lennet5_1/dense_5/kernel/Read/ReadVariableOp*lennet5_1/dense_5/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_1548882
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelennet5_1/conv2d_2/kernellennet5_1/conv2d_2/biaslennet5_1/conv2d_3/kernellennet5_1/conv2d_3/biaslennet5_1/dense_3/kernellennet5_1/dense_3/biaslennet5_1/dense_4/kernellennet5_1/dense_4/biaslennet5_1/dense_5/kernellennet5_1/dense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_1548922??
?
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1547964

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????x2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
~
)__inference_dense_3_layer_call_fn_1548719

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_15479442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548802

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????x2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
+__inference_lennet5_1_layer_call_fn_1548600
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lennet5_1_layer_call_and_return_conditional_losses_15481612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?O
?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548575
x+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
re_lu_1/ReluRelu max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
re_lu_1/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2d_3/Relu?
max_pooling2d_1/MaxPool_1MaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool_1?
re_lu_1/Relu_1Relu"max_pooling2d_1/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_1/Relu_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapere_lu_1/Relu_1:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_3/BiasAddt
re_lu_1/Relu_2Reludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
re_lu_1/Relu_2?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulre_lu_1/Relu_2:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_4/BiasAddt
re_lu_1/Relu_3Reludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
re_lu_1/Relu_3?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulre_lu_1/Relu_3:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_5/BiasAdd}
softmax_1/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_1/Softmax?
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_2/kernel/Regularizer/SquareSquareClennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,lennet5_1/conv2d_2/kernel/Regularizer/Square?
+lennet5_1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_2/kernel/Regularizer/Const?
)lennet5_1/conv2d_2/kernel/Regularizer/SumSum0lennet5_1/conv2d_2/kernel/Regularizer/Square:y:04lennet5_1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/Sum?
+lennet5_1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_2/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_2/kernel/Regularizer/mulMul4lennet5_1/conv2d_2/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/mul?
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_3/kernel/Regularizer/SquareSquareClennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,lennet5_1/conv2d_3/kernel/Regularizer/Square?
+lennet5_1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_3/kernel/Regularizer/Const?
)lennet5_1/conv2d_3/kernel/Regularizer/SumSum0lennet5_1/conv2d_3/kernel/Regularizer/Square:y:04lennet5_1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/Sum?
+lennet5_1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_3/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_3/kernel/Regularizer/mulMul4lennet5_1/conv2d_3/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/mul?
IdentityIdentitysoftmax_1/Softmax:softmax:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp<^lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<^lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2z
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2z
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548782

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548001

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????P2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1548648

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_2/kernel/Regularizer/SquareSquareClennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,lennet5_1/conv2d_2/kernel/Regularizer/Square?
+lennet5_1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_2/kernel/Regularizer/Const?
)lennet5_1/conv2d_2/kernel/Regularizer/SumSum0lennet5_1/conv2d_2/kernel/Regularizer/Square:y:04lennet5_1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/Sum?
+lennet5_1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_2/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_2/kernel/Regularizer/mulMul4lennet5_1/conv2d_2/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp<^lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1547845

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_2/kernel/Regularizer/SquareSquareClennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,lennet5_1/conv2d_2/kernel/Regularizer/Square?
+lennet5_1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_2/kernel/Regularizer/Const?
)lennet5_1/conv2d_2/kernel/Regularizer/SumSum0lennet5_1/conv2d_2/kernel/Regularizer/Square:y:04lennet5_1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/Sum?
+lennet5_1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_2/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_2/kernel/Regularizer/mulMul4lennet5_1/conv2d_2/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp<^lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
E
)__inference_re_lu_1_layer_call_fn_1548807

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15479642
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????x:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?"
?
 __inference__traced_save_1548882
file_prefix8
4savev2_lennet5_1_conv2d_2_kernel_read_readvariableop6
2savev2_lennet5_1_conv2d_2_bias_read_readvariableop8
4savev2_lennet5_1_conv2d_3_kernel_read_readvariableop6
2savev2_lennet5_1_conv2d_3_bias_read_readvariableop7
3savev2_lennet5_1_dense_3_kernel_read_readvariableop5
1savev2_lennet5_1_dense_3_bias_read_readvariableop7
3savev2_lennet5_1_dense_4_kernel_read_readvariableop5
1savev2_lennet5_1_dense_4_bias_read_readvariableop7
3savev2_lennet5_1_dense_5_kernel_read_readvariableop5
1savev2_lennet5_1_dense_5_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc0/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc0/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB(fc_out/kernel/.ATTRIBUTES/VARIABLE_VALUEB&fc_out/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_lennet5_1_conv2d_2_kernel_read_readvariableop2savev2_lennet5_1_conv2d_2_bias_read_readvariableop4savev2_lennet5_1_conv2d_3_kernel_read_readvariableop2savev2_lennet5_1_conv2d_3_bias_read_readvariableop3savev2_lennet5_1_dense_3_kernel_read_readvariableop1savev2_lennet5_1_dense_3_bias_read_readvariableop3savev2_lennet5_1_dense_4_kernel_read_readvariableop1savev2_lennet5_1_dense_4_bias_read_readvariableop3savev2_lennet5_1_dense_5_kernel_read_readvariableop1savev2_lennet5_1_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*x
_input_shapesg
e: ::: : :	?x:x:xP:P:P
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	?x: 

_output_shapes
:x:$ 

_output_shapes

:xP: 

_output_shapes
:P:$	 

_output_shapes

:P
: 


_output_shapes
:
:

_output_shapes
: 
?
E
)__inference_re_lu_1_layer_call_fn_1548797

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15478672
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1547913

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_3_layer_call_and_return_conditional_losses_1547944

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?P
?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548354
input_1+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinput_1&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
re_lu_1/ReluRelu max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
re_lu_1/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2d_3/Relu?
max_pooling2d_1/MaxPool_1MaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool_1?
re_lu_1/Relu_1Relu"max_pooling2d_1/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_1/Relu_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapere_lu_1/Relu_1:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_3/BiasAddt
re_lu_1/Relu_2Reludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
re_lu_1/Relu_2?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulre_lu_1/Relu_2:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_4/BiasAddt
re_lu_1/Relu_3Reludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
re_lu_1/Relu_3?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulre_lu_1/Relu_3:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_5/BiasAdd}
softmax_1/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_1/Softmax?
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_2/kernel/Regularizer/SquareSquareClennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,lennet5_1/conv2d_2/kernel/Regularizer/Square?
+lennet5_1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_2/kernel/Regularizer/Const?
)lennet5_1/conv2d_2/kernel/Regularizer/SumSum0lennet5_1/conv2d_2/kernel/Regularizer/Square:y:04lennet5_1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/Sum?
+lennet5_1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_2/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_2/kernel/Regularizer/mulMul4lennet5_1/conv2d_2/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/mul?
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_3/kernel/Regularizer/SquareSquareClennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,lennet5_1/conv2d_3/kernel/Regularizer/Square?
+lennet5_1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_3/kernel/Regularizer/Const?
)lennet5_1/conv2d_3/kernel/Regularizer/SumSum0lennet5_1/conv2d_3/kernel/Regularizer/Square:y:04lennet5_1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/Sum?
+lennet5_1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_3/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_3/kernel/Regularizer/mulMul4lennet5_1/conv2d_3/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/mul?
IdentityIdentitysoftmax_1/Softmax:softmax:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp<^lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<^lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2z
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2z
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?O
?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548518
x+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
re_lu_1/ReluRelu max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
re_lu_1/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2d_3/Relu?
max_pooling2d_1/MaxPool_1MaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool_1?
re_lu_1/Relu_1Relu"max_pooling2d_1/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_1/Relu_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapere_lu_1/Relu_1:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_3/BiasAddt
re_lu_1/Relu_2Reludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
re_lu_1/Relu_2?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulre_lu_1/Relu_2:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_4/BiasAddt
re_lu_1/Relu_3Reludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
re_lu_1/Relu_3?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulre_lu_1/Relu_3:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_5/BiasAdd}
softmax_1/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_1/Softmax?
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_2/kernel/Regularizer/SquareSquareClennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,lennet5_1/conv2d_2/kernel/Regularizer/Square?
+lennet5_1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_2/kernel/Regularizer/Const?
)lennet5_1/conv2d_2/kernel/Regularizer/SumSum0lennet5_1/conv2d_2/kernel/Regularizer/Square:y:04lennet5_1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/Sum?
+lennet5_1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_2/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_2/kernel/Regularizer/mulMul4lennet5_1/conv2d_2/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/mul?
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_3/kernel/Regularizer/SquareSquareClennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,lennet5_1/conv2d_3/kernel/Regularizer/Square?
+lennet5_1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_3/kernel/Regularizer/Const?
)lennet5_1/conv2d_3/kernel/Regularizer/SumSum0lennet5_1/conv2d_3/kernel/Regularizer/Square:y:04lennet5_1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/Sum?
+lennet5_1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_3/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_3/kernel/Regularizer/mulMul4lennet5_1/conv2d_3/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/mul?
IdentityIdentitysoftmax_1/Softmax:softmax:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp<^lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<^lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2z
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2z
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?

*__inference_conv2d_3_layer_call_fn_1548689

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15478922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_3_layer_call_and_return_conditional_losses_1548710

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_5_layer_call_and_return_conditional_losses_1548748

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_1548829H
Dlennet5_1_conv2d_3_kernel_regularizer_square_readvariableop_resource
identity??;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDlennet5_1_conv2d_3_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype02=
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_3/kernel/Regularizer/SquareSquareClennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,lennet5_1/conv2d_3/kernel/Regularizer/Square?
+lennet5_1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_3/kernel/Regularizer/Const?
)lennet5_1/conv2d_3/kernel/Regularizer/SumSum0lennet5_1/conv2d_3/kernel/Regularizer/Square:y:04lennet5_1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/Sum?
+lennet5_1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_3/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_3/kernel/Regularizer/mulMul4lennet5_1/conv2d_3/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/mul?
IdentityIdentity-lennet5_1/conv2d_3/kernel/Regularizer/mul:z:0<^lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1548695

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1548297
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_15478122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?P
?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548411
input_1+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinput_1&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
re_lu_1/ReluRelu max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
re_lu_1/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dre_lu_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2d_3/Relu?
max_pooling2d_1/MaxPool_1MaxPoolconv2d_3/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool_1?
re_lu_1/Relu_1Relu"max_pooling2d_1/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_1/Relu_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapere_lu_1/Relu_1:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_3/BiasAddt
re_lu_1/Relu_2Reludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
re_lu_1/Relu_2?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulre_lu_1/Relu_2:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_4/BiasAddt
re_lu_1/Relu_3Reludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
re_lu_1/Relu_3?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulre_lu_1/Relu_3:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_5/BiasAdd}
softmax_1/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_1/Softmax?
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_2/kernel/Regularizer/SquareSquareClennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,lennet5_1/conv2d_2/kernel/Regularizer/Square?
+lennet5_1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_2/kernel/Regularizer/Const?
)lennet5_1/conv2d_2/kernel/Regularizer/SumSum0lennet5_1/conv2d_2/kernel/Regularizer/Square:y:04lennet5_1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/Sum?
+lennet5_1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_2/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_2/kernel/Regularizer/mulMul4lennet5_1/conv2d_2/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/mul?
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_3/kernel/Regularizer/SquareSquareClennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,lennet5_1/conv2d_3/kernel/Regularizer/Square?
+lennet5_1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_3/kernel/Regularizer/Const?
)lennet5_1/conv2d_3/kernel/Regularizer/SumSum0lennet5_1/conv2d_3/kernel/Regularizer/Square:y:04lennet5_1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/Sum?
+lennet5_1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_3/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_3/kernel/Regularizer/mulMul4lennet5_1/conv2d_3/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/mul?
IdentityIdentitysoftmax_1/Softmax:softmax:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp<^lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<^lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2z
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2z
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
+__inference_lennet5_1_layer_call_fn_1548625
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lennet5_1_layer_call_and_return_conditional_losses_15482352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?	
?
D__inference_dense_5_layer_call_and_return_conditional_losses_1548018

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
?
D__inference_dense_4_layer_call_and_return_conditional_losses_1547981

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
?
D__inference_dense_4_layer_call_and_return_conditional_losses_1548729

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1548680

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
Relu?
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_3/kernel/Regularizer/SquareSquareClennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,lennet5_1/conv2d_3/kernel/Regularizer/Square?
+lennet5_1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_3/kernel/Regularizer/Const?
)lennet5_1/conv2d_3/kernel/Regularizer/SumSum0lennet5_1/conv2d_3/kernel/Regularizer/Square:y:04lennet5_1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/Sum?
+lennet5_1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_3/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_3/kernel/Regularizer/mulMul4lennet5_1/conv2d_3/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp<^lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
#__inference__traced_restore_1548922
file_prefix.
*assignvariableop_lennet5_1_conv2d_2_kernel.
*assignvariableop_1_lennet5_1_conv2d_2_bias0
,assignvariableop_2_lennet5_1_conv2d_3_kernel.
*assignvariableop_3_lennet5_1_conv2d_3_bias/
+assignvariableop_4_lennet5_1_dense_3_kernel-
)assignvariableop_5_lennet5_1_dense_3_bias/
+assignvariableop_6_lennet5_1_dense_4_kernel-
)assignvariableop_7_lennet5_1_dense_4_bias/
+assignvariableop_8_lennet5_1_dense_5_kernel-
)assignvariableop_9_lennet5_1_dense_5_bias
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc0/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc0/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB(fc_out/kernel/.ATTRIBUTES/VARIABLE_VALUEB&fc_out/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp*assignvariableop_lennet5_1_conv2d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp*assignvariableop_1_lennet5_1_conv2d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_lennet5_1_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_lennet5_1_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp+assignvariableop_4_lennet5_1_dense_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp)assignvariableop_5_lennet5_1_dense_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp+assignvariableop_6_lennet5_1_dense_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp)assignvariableop_7_lennet5_1_dense_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_lennet5_1_dense_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_lennet5_1_dense_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
+__inference_lennet5_1_layer_call_fn_1548461
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lennet5_1_layer_call_and_return_conditional_losses_15482352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?

*__inference_conv2d_2_layer_call_fn_1548657

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15478452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
~
)__inference_dense_5_layer_call_fn_1548757

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_15480182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
E
)__inference_re_lu_1_layer_call_fn_1548787

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15479132
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548772

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????P2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1547892

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
Relu?
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_3/kernel/Regularizer/SquareSquareClennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,lennet5_1/conv2d_3/kernel/Regularizer/Square?
+lennet5_1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_3/kernel/Regularizer/Const?
)lennet5_1/conv2d_3/kernel/Regularizer/SumSum0lennet5_1/conv2d_3/kernel/Regularizer/Square:y:04lennet5_1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/Sum?
+lennet5_1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_3/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_3/kernel/Regularizer/mulMul4lennet5_1/conv2d_3/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp<^lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_1_layer_call_fn_1547824

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15478182
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548792

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_re_lu_1_layer_call_fn_1548777

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15480012
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?K
?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548161
x
conv2d_2_1548115
conv2d_2_1548117
conv2d_3_1548122
conv2d_3_1548124
dense_3_1548130
dense_3_1548132
dense_4_1548136
dense_4_1548138
dense_5_1548142
dense_5_1548144
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallxconv2d_2_1548115conv2d_2_1548117*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15478452"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15478182!
max_pooling2d_1/PartitionedCall?
re_lu_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15478672
re_lu_1/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_3_1548122conv2d_3_1548124*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15478922"
 conv2d_3/StatefulPartitionedCall?
!max_pooling2d_1/PartitionedCall_1PartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15478182#
!max_pooling2d_1/PartitionedCall_1?
re_lu_1/PartitionedCall_1PartitionedCall*max_pooling2d_1/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15479132
re_lu_1/PartitionedCall_1?
flatten_1/PartitionedCallPartitionedCall"re_lu_1/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_15479262
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_1548130dense_3_1548132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_15479442!
dense_3/StatefulPartitionedCall?
re_lu_1/PartitionedCall_2PartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15479642
re_lu_1/PartitionedCall_2?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"re_lu_1/PartitionedCall_2:output:0dense_4_1548136dense_4_1548138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_15479812!
dense_4/StatefulPartitionedCall?
re_lu_1/PartitionedCall_3PartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15480012
re_lu_1/PartitionedCall_3?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"re_lu_1/PartitionedCall_3:output:0dense_5_1548142dense_5_1548144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_15480182!
dense_5/StatefulPartitionedCall?
softmax_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_softmax_1_layer_call_and_return_conditional_losses_15480392
softmax_1/PartitionedCall?
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_1548115*&
_output_shapes
:*
dtype02=
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_2/kernel/Regularizer/SquareSquareClennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,lennet5_1/conv2d_2/kernel/Regularizer/Square?
+lennet5_1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_2/kernel/Regularizer/Const?
)lennet5_1/conv2d_2/kernel/Regularizer/SumSum0lennet5_1/conv2d_2/kernel/Regularizer/Square:y:04lennet5_1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/Sum?
+lennet5_1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_2/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_2/kernel/Regularizer/mulMul4lennet5_1/conv2d_2/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/mul?
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_1548122*&
_output_shapes
: *
dtype02=
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_3/kernel/Regularizer/SquareSquareClennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,lennet5_1/conv2d_3/kernel/Regularizer/Square?
+lennet5_1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_3/kernel/Regularizer/Const?
)lennet5_1/conv2d_3/kernel/Regularizer/SumSum0lennet5_1/conv2d_3/kernel/Regularizer/Square:y:04lennet5_1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/Sum?
+lennet5_1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_3/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_3/kernel/Regularizer/mulMul4lennet5_1/conv2d_3/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/mul?
IdentityIdentity"softmax_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall<^lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<^lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2z
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2z
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
G
+__inference_softmax_1_layer_call_fn_1548767

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_softmax_1_layer_call_and_return_conditional_losses_15480392
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
+__inference_lennet5_1_layer_call_fn_1548436
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lennet5_1_layer_call_and_return_conditional_losses_15481612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1547926

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
G
+__inference_flatten_1_layer_call_fn_1548700

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_15479262
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?K
?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548235
x
conv2d_2_1548189
conv2d_2_1548191
conv2d_3_1548196
conv2d_3_1548198
dense_3_1548204
dense_3_1548206
dense_4_1548210
dense_4_1548212
dense_5_1548216
dense_5_1548218
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallxconv2d_2_1548189conv2d_2_1548191*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15478452"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15478182!
max_pooling2d_1/PartitionedCall?
re_lu_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15478672
re_lu_1/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_3_1548196conv2d_3_1548198*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15478922"
 conv2d_3/StatefulPartitionedCall?
!max_pooling2d_1/PartitionedCall_1PartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15478182#
!max_pooling2d_1/PartitionedCall_1?
re_lu_1/PartitionedCall_1PartitionedCall*max_pooling2d_1/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15479132
re_lu_1/PartitionedCall_1?
flatten_1/PartitionedCallPartitionedCall"re_lu_1/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_15479262
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_1548204dense_3_1548206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_15479442!
dense_3/StatefulPartitionedCall?
re_lu_1/PartitionedCall_2PartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15479642
re_lu_1/PartitionedCall_2?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"re_lu_1/PartitionedCall_2:output:0dense_4_1548210dense_4_1548212*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_15479812!
dense_4/StatefulPartitionedCall?
re_lu_1/PartitionedCall_3PartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_15480012
re_lu_1/PartitionedCall_3?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"re_lu_1/PartitionedCall_3:output:0dense_5_1548216dense_5_1548218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_15480182!
dense_5/StatefulPartitionedCall?
softmax_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_softmax_1_layer_call_and_return_conditional_losses_15480392
softmax_1/PartitionedCall?
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_1548189*&
_output_shapes
:*
dtype02=
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_2/kernel/Regularizer/SquareSquareClennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,lennet5_1/conv2d_2/kernel/Regularizer/Square?
+lennet5_1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_2/kernel/Regularizer/Const?
)lennet5_1/conv2d_2/kernel/Regularizer/SumSum0lennet5_1/conv2d_2/kernel/Regularizer/Square:y:04lennet5_1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/Sum?
+lennet5_1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_2/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_2/kernel/Regularizer/mulMul4lennet5_1/conv2d_2/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/mul?
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_1548196*&
_output_shapes
: *
dtype02=
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_3/kernel/Regularizer/SquareSquareClennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,lennet5_1/conv2d_3/kernel/Regularizer/Square?
+lennet5_1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_3/kernel/Regularizer/Const?
)lennet5_1/conv2d_3/kernel/Regularizer/SumSum0lennet5_1/conv2d_3/kernel/Regularizer/Square:y:04lennet5_1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/Sum?
+lennet5_1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_3/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_3/kernel/Regularizer/mulMul4lennet5_1/conv2d_3/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_3/kernel/Regularizer/mul?
IdentityIdentity"softmax_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall<^lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp<^lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2z
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2z
;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
?
__inference_loss_fn_0_1548818H
Dlennet5_1_conv2d_2_kernel_regularizer_square_readvariableop_resource
identity??;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDlennet5_1_conv2d_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02=
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
,lennet5_1/conv2d_2/kernel/Regularizer/SquareSquareClennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,lennet5_1/conv2d_2/kernel/Regularizer/Square?
+lennet5_1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+lennet5_1/conv2d_2/kernel/Regularizer/Const?
)lennet5_1/conv2d_2/kernel/Regularizer/SumSum0lennet5_1/conv2d_2/kernel/Regularizer/Square:y:04lennet5_1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/Sum?
+lennet5_1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+lennet5_1/conv2d_2/kernel/Regularizer/mul/x?
)lennet5_1/conv2d_2/kernel/Regularizer/mulMul4lennet5_1/conv2d_2/kernel/Regularizer/mul/x:output:02lennet5_1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lennet5_1/conv2d_2/kernel/Regularizer/mul?
IdentityIdentity-lennet5_1/conv2d_2/kernel/Regularizer/mul:z:0<^lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp;lennet5_1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp
?
~
)__inference_dense_4_layer_call_fn_1548738

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_15479812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1547818

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?C
?
"__inference__wrapped_model_1547812
input_15
1lennet5_1_conv2d_2_conv2d_readvariableop_resource6
2lennet5_1_conv2d_2_biasadd_readvariableop_resource5
1lennet5_1_conv2d_3_conv2d_readvariableop_resource6
2lennet5_1_conv2d_3_biasadd_readvariableop_resource4
0lennet5_1_dense_3_matmul_readvariableop_resource5
1lennet5_1_dense_3_biasadd_readvariableop_resource4
0lennet5_1_dense_4_matmul_readvariableop_resource5
1lennet5_1_dense_4_biasadd_readvariableop_resource4
0lennet5_1_dense_5_matmul_readvariableop_resource5
1lennet5_1_dense_5_biasadd_readvariableop_resource
identity??)lennet5_1/conv2d_2/BiasAdd/ReadVariableOp?(lennet5_1/conv2d_2/Conv2D/ReadVariableOp?)lennet5_1/conv2d_3/BiasAdd/ReadVariableOp?(lennet5_1/conv2d_3/Conv2D/ReadVariableOp?(lennet5_1/dense_3/BiasAdd/ReadVariableOp?'lennet5_1/dense_3/MatMul/ReadVariableOp?(lennet5_1/dense_4/BiasAdd/ReadVariableOp?'lennet5_1/dense_4/MatMul/ReadVariableOp?(lennet5_1/dense_5/BiasAdd/ReadVariableOp?'lennet5_1/dense_5/MatMul/ReadVariableOp?
(lennet5_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp1lennet5_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(lennet5_1/conv2d_2/Conv2D/ReadVariableOp?
lennet5_1/conv2d_2/Conv2DConv2Dinput_10lennet5_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
lennet5_1/conv2d_2/Conv2D?
)lennet5_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2lennet5_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)lennet5_1/conv2d_2/BiasAdd/ReadVariableOp?
lennet5_1/conv2d_2/BiasAddBiasAdd"lennet5_1/conv2d_2/Conv2D:output:01lennet5_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
lennet5_1/conv2d_2/BiasAdd?
lennet5_1/conv2d_2/ReluRelu#lennet5_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
lennet5_1/conv2d_2/Relu?
!lennet5_1/max_pooling2d_1/MaxPoolMaxPool%lennet5_1/conv2d_2/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2#
!lennet5_1/max_pooling2d_1/MaxPool?
lennet5_1/re_lu_1/ReluRelu*lennet5_1/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
lennet5_1/re_lu_1/Relu?
(lennet5_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp1lennet5_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(lennet5_1/conv2d_3/Conv2D/ReadVariableOp?
lennet5_1/conv2d_3/Conv2DConv2D$lennet5_1/re_lu_1/Relu:activations:00lennet5_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
lennet5_1/conv2d_3/Conv2D?
)lennet5_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2lennet5_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)lennet5_1/conv2d_3/BiasAdd/ReadVariableOp?
lennet5_1/conv2d_3/BiasAddBiasAdd"lennet5_1/conv2d_3/Conv2D:output:01lennet5_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
lennet5_1/conv2d_3/BiasAdd?
lennet5_1/conv2d_3/ReluRelu#lennet5_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
lennet5_1/conv2d_3/Relu?
#lennet5_1/max_pooling2d_1/MaxPool_1MaxPool%lennet5_1/conv2d_3/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2%
#lennet5_1/max_pooling2d_1/MaxPool_1?
lennet5_1/re_lu_1/Relu_1Relu,lennet5_1/max_pooling2d_1/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
lennet5_1/re_lu_1/Relu_1?
lennet5_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
lennet5_1/flatten_1/Const?
lennet5_1/flatten_1/ReshapeReshape&lennet5_1/re_lu_1/Relu_1:activations:0"lennet5_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lennet5_1/flatten_1/Reshape?
'lennet5_1/dense_3/MatMul/ReadVariableOpReadVariableOp0lennet5_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02)
'lennet5_1/dense_3/MatMul/ReadVariableOp?
lennet5_1/dense_3/MatMulMatMul$lennet5_1/flatten_1/Reshape:output:0/lennet5_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
lennet5_1/dense_3/MatMul?
(lennet5_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp1lennet5_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02*
(lennet5_1/dense_3/BiasAdd/ReadVariableOp?
lennet5_1/dense_3/BiasAddBiasAdd"lennet5_1/dense_3/MatMul:product:00lennet5_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
lennet5_1/dense_3/BiasAdd?
lennet5_1/re_lu_1/Relu_2Relu"lennet5_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
lennet5_1/re_lu_1/Relu_2?
'lennet5_1/dense_4/MatMul/ReadVariableOpReadVariableOp0lennet5_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02)
'lennet5_1/dense_4/MatMul/ReadVariableOp?
lennet5_1/dense_4/MatMulMatMul&lennet5_1/re_lu_1/Relu_2:activations:0/lennet5_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
lennet5_1/dense_4/MatMul?
(lennet5_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp1lennet5_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02*
(lennet5_1/dense_4/BiasAdd/ReadVariableOp?
lennet5_1/dense_4/BiasAddBiasAdd"lennet5_1/dense_4/MatMul:product:00lennet5_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
lennet5_1/dense_4/BiasAdd?
lennet5_1/re_lu_1/Relu_3Relu"lennet5_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
lennet5_1/re_lu_1/Relu_3?
'lennet5_1/dense_5/MatMul/ReadVariableOpReadVariableOp0lennet5_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02)
'lennet5_1/dense_5/MatMul/ReadVariableOp?
lennet5_1/dense_5/MatMulMatMul&lennet5_1/re_lu_1/Relu_3:activations:0/lennet5_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
lennet5_1/dense_5/MatMul?
(lennet5_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp1lennet5_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(lennet5_1/dense_5/BiasAdd/ReadVariableOp?
lennet5_1/dense_5/BiasAddBiasAdd"lennet5_1/dense_5/MatMul:product:00lennet5_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
lennet5_1/dense_5/BiasAdd?
lennet5_1/softmax_1/SoftmaxSoftmax"lennet5_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
lennet5_1/softmax_1/Softmax?
IdentityIdentity%lennet5_1/softmax_1/Softmax:softmax:0*^lennet5_1/conv2d_2/BiasAdd/ReadVariableOp)^lennet5_1/conv2d_2/Conv2D/ReadVariableOp*^lennet5_1/conv2d_3/BiasAdd/ReadVariableOp)^lennet5_1/conv2d_3/Conv2D/ReadVariableOp)^lennet5_1/dense_3/BiasAdd/ReadVariableOp(^lennet5_1/dense_3/MatMul/ReadVariableOp)^lennet5_1/dense_4/BiasAdd/ReadVariableOp(^lennet5_1/dense_4/MatMul/ReadVariableOp)^lennet5_1/dense_5/BiasAdd/ReadVariableOp(^lennet5_1/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2V
)lennet5_1/conv2d_2/BiasAdd/ReadVariableOp)lennet5_1/conv2d_2/BiasAdd/ReadVariableOp2T
(lennet5_1/conv2d_2/Conv2D/ReadVariableOp(lennet5_1/conv2d_2/Conv2D/ReadVariableOp2V
)lennet5_1/conv2d_3/BiasAdd/ReadVariableOp)lennet5_1/conv2d_3/BiasAdd/ReadVariableOp2T
(lennet5_1/conv2d_3/Conv2D/ReadVariableOp(lennet5_1/conv2d_3/Conv2D/ReadVariableOp2T
(lennet5_1/dense_3/BiasAdd/ReadVariableOp(lennet5_1/dense_3/BiasAdd/ReadVariableOp2R
'lennet5_1/dense_3/MatMul/ReadVariableOp'lennet5_1/dense_3/MatMul/ReadVariableOp2T
(lennet5_1/dense_4/BiasAdd/ReadVariableOp(lennet5_1/dense_4/BiasAdd/ReadVariableOp2R
'lennet5_1/dense_4/MatMul/ReadVariableOp'lennet5_1/dense_4/MatMul/ReadVariableOp2T
(lennet5_1/dense_5/BiasAdd/ReadVariableOp(lennet5_1/dense_5/BiasAdd/ReadVariableOp2R
'lennet5_1/dense_5/MatMul/ReadVariableOp'lennet5_1/dense_5/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
b
F__inference_softmax_1_layer_call_and_return_conditional_losses_1548039

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1547867

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_softmax_1_layer_call_and_return_conditional_losses_1548762

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????  <
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
	conv1
	conv2
maxpool
flatten
fc0
fc1

fc_out
softmax
	
activation

regularization_losses
trainable_variables
	variables
	keras_api

signatures
o__call__
p_default_save_signature
*q&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "Lennet5", "name": "lennet5_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Lennet5"}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 18, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 32, 32, 3]}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 18}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 14, 14, 18]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
regularization_losses
 trainable_variables
!	variables
"	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
z__call__
*{&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 120, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 800}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 800]}}
?

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
|__call__
*}&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 120]}}
?

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
~__call__
*&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 80]}}
?
5regularization_losses
6trainable_variables
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax_1", "trainable": true, "dtype": "float32", "axis": -1}}
?
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
0
?0
?1"
trackable_list_wrapper
f
0
1
2
3
#4
$5
)6
*7
/8
09"
trackable_list_wrapper
f
0
1
2
3
#4
$5
)6
*7
/8
09"
trackable_list_wrapper
?

regularization_losses
=layer_metrics
>layer_regularization_losses
trainable_variables
?metrics
	variables

@layers
Anon_trainable_variables
o__call__
p_default_save_signature
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
3:12lennet5_1/conv2d_2/kernel
%:#2lennet5_1/conv2d_2/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
Blayer_metrics
Clayer_regularization_losses
trainable_variables
Dmetrics
	variables

Elayers
Fnon_trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
3:1 2lennet5_1/conv2d_3/kernel
%:# 2lennet5_1/conv2d_3/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
Glayer_metrics
Hlayer_regularization_losses
trainable_variables
Imetrics
	variables

Jlayers
Knon_trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Llayer_metrics
Mlayer_regularization_losses
trainable_variables
Nmetrics
	variables

Olayers
Pnon_trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Qlayer_metrics
Rlayer_regularization_losses
 trainable_variables
Smetrics
!	variables

Tlayers
Unon_trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
+:)	?x2lennet5_1/dense_3/kernel
$:"x2lennet5_1/dense_3/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
%regularization_losses
Vlayer_metrics
Wlayer_regularization_losses
&trainable_variables
Xmetrics
'	variables

Ylayers
Znon_trainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
*:(xP2lennet5_1/dense_4/kernel
$:"P2lennet5_1/dense_4/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
+regularization_losses
[layer_metrics
\layer_regularization_losses
,trainable_variables
]metrics
-	variables

^layers
_non_trainable_variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
*:(P
2lennet5_1/dense_5/kernel
$:"
2lennet5_1/dense_5/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
1regularization_losses
`layer_metrics
alayer_regularization_losses
2trainable_variables
bmetrics
3	variables

clayers
dnon_trainable_variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
5regularization_losses
elayer_metrics
flayer_regularization_losses
6trainable_variables
gmetrics
7	variables

hlayers
inon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
9regularization_losses
jlayer_metrics
klayer_regularization_losses
:trainable_variables
lmetrics
;	variables

mlayers
nnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
+__inference_lennet5_1_layer_call_fn_1548436
+__inference_lennet5_1_layer_call_fn_1548600
+__inference_lennet5_1_layer_call_fn_1548625
+__inference_lennet5_1_layer_call_fn_1548461?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_1547812?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????  
?2?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548354
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548518
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548575
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548411?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_2_layer_call_fn_1548657?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1548648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_3_layer_call_fn_1548689?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1548680?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_pooling2d_1_layer_call_fn_1547824?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1547818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_flatten_1_layer_call_fn_1548700?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_1_layer_call_and_return_conditional_losses_1548695?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_3_layer_call_fn_1548719?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_3_layer_call_and_return_conditional_losses_1548710?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_4_layer_call_fn_1548738?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_4_layer_call_and_return_conditional_losses_1548729?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_5_layer_call_fn_1548757?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_5_layer_call_and_return_conditional_losses_1548748?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_softmax_1_layer_call_fn_1548767?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_softmax_1_layer_call_and_return_conditional_losses_1548762?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_re_lu_1_layer_call_fn_1548807
)__inference_re_lu_1_layer_call_fn_1548797
)__inference_re_lu_1_layer_call_fn_1548787
)__inference_re_lu_1_layer_call_fn_1548777?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548782
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548792
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548772
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548802?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_1548818?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_1548829?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
%__inference_signature_wrapper_1548297input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_1547812{
#$)*/08?5
.?+
)?&
input_1?????????  
? "3?0
.
output_1"?
output_1?????????
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1548648l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????
? ?
*__inference_conv2d_2_layer_call_fn_1548657_7?4
-?*
(?%
inputs?????????  
? " ???????????
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1548680l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

 
? ?
*__inference_conv2d_3_layer_call_fn_1548689_7?4
-?*
(?%
inputs?????????
? " ??????????

 ?
D__inference_dense_3_layer_call_and_return_conditional_losses_1548710]#$0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????x
? }
)__inference_dense_3_layer_call_fn_1548719P#$0?-
&?#
!?
inputs??????????
? "??????????x?
D__inference_dense_4_layer_call_and_return_conditional_losses_1548729\)*/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????P
? |
)__inference_dense_4_layer_call_fn_1548738O)*/?,
%?"
 ?
inputs?????????x
? "??????????P?
D__inference_dense_5_layer_call_and_return_conditional_losses_1548748\/0/?,
%?"
 ?
inputs?????????P
? "%?"
?
0?????????

? |
)__inference_dense_5_layer_call_fn_1548757O/0/?,
%?"
 ?
inputs?????????P
? "??????????
?
F__inference_flatten_1_layer_call_and_return_conditional_losses_1548695a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
+__inference_flatten_1_layer_call_fn_1548700T7?4
-?*
(?%
inputs????????? 
? "????????????
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548354q
#$)*/0<?9
2?/
)?&
input_1?????????  
p
? "%?"
?
0?????????

? ?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548411q
#$)*/0<?9
2?/
)?&
input_1?????????  
p 
? "%?"
?
0?????????

? ?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548518k
#$)*/06?3
,?)
#? 
x?????????  
p
? "%?"
?
0?????????

? ?
F__inference_lennet5_1_layer_call_and_return_conditional_losses_1548575k
#$)*/06?3
,?)
#? 
x?????????  
p 
? "%?"
?
0?????????

? ?
+__inference_lennet5_1_layer_call_fn_1548436d
#$)*/0<?9
2?/
)?&
input_1?????????  
p
? "??????????
?
+__inference_lennet5_1_layer_call_fn_1548461d
#$)*/0<?9
2?/
)?&
input_1?????????  
p 
? "??????????
?
+__inference_lennet5_1_layer_call_fn_1548600^
#$)*/06?3
,?)
#? 
x?????????  
p
? "??????????
?
+__inference_lennet5_1_layer_call_fn_1548625^
#$)*/06?3
,?)
#? 
x?????????  
p 
? "??????????
<
__inference_loss_fn_0_1548818?

? 
? "? <
__inference_loss_fn_1_1548829?

? 
? "? ?
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1547818?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_1_layer_call_fn_1547824?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548772X/?,
%?"
 ?
inputs?????????P
? "%?"
?
0?????????P
? ?
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548782h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548792h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
D__inference_re_lu_1_layer_call_and_return_conditional_losses_1548802X/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????x
? x
)__inference_re_lu_1_layer_call_fn_1548777K/?,
%?"
 ?
inputs?????????P
? "??????????P?
)__inference_re_lu_1_layer_call_fn_1548787[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
)__inference_re_lu_1_layer_call_fn_1548797[7?4
-?*
(?%
inputs?????????
? " ??????????x
)__inference_re_lu_1_layer_call_fn_1548807K/?,
%?"
 ?
inputs?????????x
? "??????????x?
%__inference_signature_wrapper_1548297?
#$)*/0C?@
? 
9?6
4
input_1)?&
input_1?????????  "3?0
.
output_1"?
output_1?????????
?
F__inference_softmax_1_layer_call_and_return_conditional_losses_1548762\3?0
)?&
 ?
inputs?????????


 
? "%?"
?
0?????????

? ~
+__inference_softmax_1_layer_call_fn_1548767O3?0
)?&
 ?
inputs?????????


 
? "??????????
