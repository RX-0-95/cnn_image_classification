??	
??
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
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
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
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8Ƹ
?
le_net_8/conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namele_net_8/conv2d_22/kernel
?
-le_net_8/conv2d_22/kernel/Read/ReadVariableOpReadVariableOple_net_8/conv2d_22/kernel*&
_output_shapes
:*
dtype0
?
le_net_8/conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namele_net_8/conv2d_22/bias

+le_net_8/conv2d_22/bias/Read/ReadVariableOpReadVariableOple_net_8/conv2d_22/bias*
_output_shapes
:*
dtype0
?
le_net_8/conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namele_net_8/conv2d_23/kernel
?
-le_net_8/conv2d_23/kernel/Read/ReadVariableOpReadVariableOple_net_8/conv2d_23/kernel*&
_output_shapes
:*
dtype0
?
le_net_8/conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namele_net_8/conv2d_23/bias

+le_net_8/conv2d_23/bias/Read/ReadVariableOpReadVariableOple_net_8/conv2d_23/bias*
_output_shapes
:*
dtype0
?
le_net_8/dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?x*)
shared_namele_net_8/dense_33/kernel
?
,le_net_8/dense_33/kernel/Read/ReadVariableOpReadVariableOple_net_8/dense_33/kernel*
_output_shapes
:	?x*
dtype0
?
le_net_8/dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*'
shared_namele_net_8/dense_33/bias
}
*le_net_8/dense_33/bias/Read/ReadVariableOpReadVariableOple_net_8/dense_33/bias*
_output_shapes
:x*
dtype0
?
le_net_8/dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*)
shared_namele_net_8/dense_34/kernel
?
,le_net_8/dense_34/kernel/Read/ReadVariableOpReadVariableOple_net_8/dense_34/kernel*
_output_shapes

:xT*
dtype0
?
le_net_8/dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*'
shared_namele_net_8/dense_34/bias
}
*le_net_8/dense_34/bias/Read/ReadVariableOpReadVariableOple_net_8/dense_34/bias*
_output_shapes
:T*
dtype0
?
le_net_8/dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
*)
shared_namele_net_8/dense_35/kernel
?
,le_net_8/dense_35/kernel/Read/ReadVariableOpReadVariableOple_net_8/dense_35/kernel*
_output_shapes

:T
*
dtype0
?
le_net_8/dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_namele_net_8/dense_35/bias
}
*le_net_8/dense_35/bias/Read/ReadVariableOpReadVariableOple_net_8/dense_35/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
	conv1
	conv2
fc1
fc2
fc3
flatten
softmax
tanh
	sigmoid

pool
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
h

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
R
.trainable_variables
/regularization_losses
0	variables
1	keras_api
R
2trainable_variables
3regularization_losses
4	variables
5	keras_api

6	keras_api

7	keras_api
R
8trainable_variables
9regularization_losses
:	variables
;	keras_api
F
0
1
2
3
4
5
"6
#7
(8
)9
 
F
0
1
2
3
4
5
"6
#7
(8
)9
?

<layers
=layer_metrics
trainable_variables
regularization_losses
>layer_regularization_losses
?metrics
	variables
@non_trainable_variables
 
VT
VARIABLE_VALUEle_net_8/conv2d_22/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEle_net_8/conv2d_22/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Alayers
Blayer_metrics
trainable_variables
regularization_losses
Clayer_regularization_losses
Dmetrics
	variables
Enon_trainable_variables
VT
VARIABLE_VALUEle_net_8/conv2d_23/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEle_net_8/conv2d_23/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Flayers
Glayer_metrics
trainable_variables
regularization_losses
Hlayer_regularization_losses
Imetrics
	variables
Jnon_trainable_variables
SQ
VARIABLE_VALUEle_net_8/dense_33/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEle_net_8/dense_33/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Klayers
Llayer_metrics
trainable_variables
regularization_losses
Mlayer_regularization_losses
Nmetrics
 	variables
Onon_trainable_variables
SQ
VARIABLE_VALUEle_net_8/dense_34/kernel%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEle_net_8/dense_34/bias#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
?

Players
Qlayer_metrics
$trainable_variables
%regularization_losses
Rlayer_regularization_losses
Smetrics
&	variables
Tnon_trainable_variables
SQ
VARIABLE_VALUEle_net_8/dense_35/kernel%fc3/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEle_net_8/dense_35/bias#fc3/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?

Ulayers
Vlayer_metrics
*trainable_variables
+regularization_losses
Wlayer_regularization_losses
Xmetrics
,	variables
Ynon_trainable_variables
 
 
 
?

Zlayers
[layer_metrics
.trainable_variables
/regularization_losses
\layer_regularization_losses
]metrics
0	variables
^non_trainable_variables
 
 
 
?

_layers
`layer_metrics
2trainable_variables
3regularization_losses
alayer_regularization_losses
bmetrics
4	variables
cnon_trainable_variables
 
 
 
 
 
?

dlayers
elayer_metrics
8trainable_variables
9regularization_losses
flayer_regularization_losses
gmetrics
:	variables
hnon_trainable_variables
F
0
1
2
3
4
5
6
7
	8

9
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1le_net_8/conv2d_22/kernelle_net_8/conv2d_22/biasle_net_8/conv2d_23/kernelle_net_8/conv2d_23/biasle_net_8/dense_33/kernelle_net_8/dense_33/biasle_net_8/dense_34/kernelle_net_8/dense_34/biasle_net_8/dense_35/kernelle_net_8/dense_35/bias*
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
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_5401239
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-le_net_8/conv2d_22/kernel/Read/ReadVariableOp+le_net_8/conv2d_22/bias/Read/ReadVariableOp-le_net_8/conv2d_23/kernel/Read/ReadVariableOp+le_net_8/conv2d_23/bias/Read/ReadVariableOp,le_net_8/dense_33/kernel/Read/ReadVariableOp*le_net_8/dense_33/bias/Read/ReadVariableOp,le_net_8/dense_34/kernel/Read/ReadVariableOp*le_net_8/dense_34/bias/Read/ReadVariableOp,le_net_8/dense_35/kernel/Read/ReadVariableOp*le_net_8/dense_35/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_5401774
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamele_net_8/conv2d_22/kernelle_net_8/conv2d_22/biasle_net_8/conv2d_23/kernelle_net_8/conv2d_23/biasle_net_8/dense_33/kernelle_net_8/dense_33/biasle_net_8/dense_34/kernelle_net_8/dense_34/biasle_net_8/dense_35/kernelle_net_8/dense_35/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_5401814??
?
?
__inference_loss_fn_0_5401710H
Dle_net_8_conv2d_22_kernel_regularizer_square_readvariableop_resource
identity??;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDle_net_8_conv2d_22_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_22/kernel/Regularizer/SquareSquareCle_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_22/kernel/Regularizer/Square?
+le_net_8/conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_22/kernel/Regularizer/Const?
)le_net_8/conv2d_22/kernel/Regularizer/SumSum0le_net_8/conv2d_22/kernel/Regularizer/Square:y:04le_net_8/conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/Sum?
+le_net_8/conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_22/kernel/Regularizer/mul/x?
)le_net_8/conv2d_22/kernel/Regularizer/mulMul4le_net_8/conv2d_22/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/mul?
IdentityIdentity-le_net_8/conv2d_22/kernel/Regularizer/mul:z:0<^le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp
?
M
1__inference_max_pooling2d_4_layer_call_fn_5400810

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
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_54008042
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
?	
?
E__inference_dense_33_layer_call_and_return_conditional_losses_5401631

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?x*
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?N
?
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401454
input_1,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity?? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp
ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Constd
PadPadinput_1Const:output:0*
T0*/
_output_shapes
:?????????$$2
Pad?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2DPad:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_22/BiasAdd?
max_pooling2d_4/MaxPoolMaxPoolconv2d_22/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1?
Pad_1Pad max_pooling2d_4/MaxPool:output:0Const_1:output:0*
T0*/
_output_shapes
:?????????2
Pad_1?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2DPad_1:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_23/BiasAdd?
max_pooling2d_4/MaxPool_1MaxPoolconv2d_23/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool_1u
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_11/Const?
flatten_11/ReshapeReshape"max_pooling2d_4/MaxPool_1:output:0flatten_11/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_11/Reshape?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMulflatten_11/Reshape:output:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_33/BiasAdd?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_33/BiasAdd:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_34/BiasAdd?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_34/BiasAdd:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/BiasAdd?
softmax_11/SoftmaxSoftmaxdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_11/Softmax?
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_22/kernel/Regularizer/SquareSquareCle_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_22/kernel/Regularizer/Square?
+le_net_8/conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_22/kernel/Regularizer/Const?
)le_net_8/conv2d_22/kernel/Regularizer/SumSum0le_net_8/conv2d_22/kernel/Regularizer/Square:y:04le_net_8/conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/Sum?
+le_net_8/conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_22/kernel/Regularizer/mul/x?
)le_net_8/conv2d_22/kernel/Regularizer/mulMul4le_net_8/conv2d_22/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/mul?
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_23/kernel/Regularizer/SquareSquareCle_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_23/kernel/Regularizer/Square?
+le_net_8/conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_23/kernel/Regularizer/Const?
)le_net_8/conv2d_23/kernel/Regularizer/SumSum0le_net_8/conv2d_23/kernel/Regularizer/Square:y:04le_net_8/conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/Sum?
+le_net_8/conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_23/kernel/Regularizer/mul/x?
)le_net_8/conv2d_23/kernel/Regularizer/mulMul4le_net_8/conv2d_23/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/mul?
IdentityIdentitysoftmax_11/Softmax:softmax:0!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp<^le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp<^le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2z
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp2z
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
F__inference_conv2d_23_layer_call_and_return_conditional_losses_5401612

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_23/kernel/Regularizer/SquareSquareCle_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_23/kernel/Regularizer/Square?
+le_net_8/conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_23/kernel/Regularizer/Const?
)le_net_8/conv2d_23/kernel/Regularizer/SumSum0le_net_8/conv2d_23/kernel/Regularizer/Square:y:04le_net_8/conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/Sum?
+le_net_8/conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_23/kernel/Regularizer/mul/x?
)le_net_8/conv2d_23/kernel/Regularizer/mulMul4le_net_8/conv2d_23/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp<^le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_23_layer_call_fn_5401621

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
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_23_layer_call_and_return_conditional_losses_54008672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_34_layer_call_and_return_conditional_losses_5401650

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????T2

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
E__inference_dense_34_layer_call_and_return_conditional_losses_5400934

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????T2

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
E__inference_dense_33_layer_call_and_return_conditional_losses_5400908

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?x*
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?N
?
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401294
x,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity?? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp
ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Const^
PadPadxConst:output:0*
T0*/
_output_shapes
:?????????$$2
Pad?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2DPad:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_22/BiasAdd?
max_pooling2d_4/MaxPoolMaxPoolconv2d_22/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1?
Pad_1Pad max_pooling2d_4/MaxPool:output:0Const_1:output:0*
T0*/
_output_shapes
:?????????2
Pad_1?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2DPad_1:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_23/BiasAdd?
max_pooling2d_4/MaxPool_1MaxPoolconv2d_23/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool_1u
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_11/Const?
flatten_11/ReshapeReshape"max_pooling2d_4/MaxPool_1:output:0flatten_11/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_11/Reshape?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMulflatten_11/Reshape:output:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_33/BiasAdd?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_33/BiasAdd:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_34/BiasAdd?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_34/BiasAdd:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/BiasAdd?
softmax_11/SoftmaxSoftmaxdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_11/Softmax?
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_22/kernel/Regularizer/SquareSquareCle_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_22/kernel/Regularizer/Square?
+le_net_8/conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_22/kernel/Regularizer/Const?
)le_net_8/conv2d_22/kernel/Regularizer/SumSum0le_net_8/conv2d_22/kernel/Regularizer/Square:y:04le_net_8/conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/Sum?
+le_net_8/conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_22/kernel/Regularizer/mul/x?
)le_net_8/conv2d_22/kernel/Regularizer/mulMul4le_net_8/conv2d_22/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/mul?
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_23/kernel/Regularizer/SquareSquareCle_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_23/kernel/Regularizer/Square?
+le_net_8/conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_23/kernel/Regularizer/Const?
)le_net_8/conv2d_23/kernel/Regularizer/SumSum0le_net_8/conv2d_23/kernel/Regularizer/Square:y:04le_net_8/conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/Sum?
+le_net_8/conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_23/kernel/Regularizer/mul/x?
)le_net_8/conv2d_23/kernel/Regularizer/mulMul4le_net_8/conv2d_23/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/mul?
IdentityIdentitysoftmax_11/Softmax:softmax:0!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp<^le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp<^le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2z
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp2z
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
?
*__inference_le_net_8_layer_call_fn_5401399
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
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_le_net_8_layer_call_and_return_conditional_losses_54011772
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
?

*__inference_dense_34_layer_call_fn_5401659

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
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_54009342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????T2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
*__inference_le_net_8_layer_call_fn_5401534
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
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_le_net_8_layer_call_and_return_conditional_losses_54011032
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
?	
?
E__inference_dense_35_layer_call_and_return_conditional_losses_5401669

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
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
:?????????T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_5401239
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
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_54007982
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
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_5400890

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_le_net_8_layer_call_fn_5401374
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
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_le_net_8_layer_call_and_return_conditional_losses_54011032
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
?

*__inference_dense_33_layer_call_fn_5401640

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
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_54009082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????x2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5401581

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd?
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_22/kernel/Regularizer/SquareSquareCle_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_22/kernel/Regularizer/Square?
+le_net_8/conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_22/kernel/Regularizer/Const?
)le_net_8/conv2d_22/kernel/Regularizer/SumSum0le_net_8/conv2d_22/kernel/Regularizer/Square:y:04le_net_8/conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/Sum?
+le_net_8/conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_22/kernel/Regularizer/mul/x?
)le_net_8/conv2d_22/kernel/Regularizer/mulMul4le_net_8/conv2d_22/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp<^le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????$$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?
?
F__inference_conv2d_23_layer_call_and_return_conditional_losses_5400867

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_23/kernel/Regularizer/SquareSquareCle_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_23/kernel/Regularizer/Square?
+le_net_8/conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_23/kernel/Regularizer/Const?
)le_net_8/conv2d_23/kernel/Regularizer/SumSum0le_net_8/conv2d_23/kernel/Regularizer/Square:y:04le_net_8/conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/Sum?
+le_net_8/conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_23/kernel/Regularizer/mul/x?
)le_net_8/conv2d_23/kernel/Regularizer/mulMul4le_net_8/conv2d_23/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp<^le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_11_layer_call_fn_5401689

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_54008902
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_5401721H
Dle_net_8_conv2d_23_kernel_regularizer_square_readvariableop_resource
identity??;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDle_net_8_conv2d_23_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_23/kernel/Regularizer/SquareSquareCle_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_23/kernel/Regularizer/Square?
+le_net_8/conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_23/kernel/Regularizer/Const?
)le_net_8/conv2d_23/kernel/Regularizer/SumSum0le_net_8/conv2d_23/kernel/Regularizer/Square:y:04le_net_8/conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/Sum?
+le_net_8/conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_23/kernel/Regularizer/mul/x?
)le_net_8/conv2d_23/kernel/Regularizer/mulMul4le_net_8/conv2d_23/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/mul?
IdentityIdentity-le_net_8/conv2d_23/kernel/Regularizer/mul:z:0<^le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp
?C
?
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401177
x
conv2d_22_5401133
conv2d_22_5401135
conv2d_23_5401141
conv2d_23_5401143
dense_33_5401148
dense_33_5401150
dense_34_5401153
dense_34_5401155
dense_35_5401158
dense_35_5401160
identity??!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp
ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Const^
PadPadxConst:output:0*
T0*/
_output_shapes
:?????????$$2
Pad?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCallPad:output:0conv2d_22_5401133conv2d_22_5401135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_54008322#
!conv2d_22/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_54008042!
max_pooling2d_4/PartitionedCall?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1?
Pad_1Pad(max_pooling2d_4/PartitionedCall:output:0Const_1:output:0*
T0*/
_output_shapes
:?????????2
Pad_1?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallPad_1:output:0conv2d_23_5401141conv2d_23_5401143*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_23_layer_call_and_return_conditional_losses_54008672#
!conv2d_23/StatefulPartitionedCall?
!max_pooling2d_4/PartitionedCall_1PartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_54008042#
!max_pooling2d_4/PartitionedCall_1?
flatten_11/PartitionedCallPartitionedCall*max_pooling2d_4/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_54008902
flatten_11/PartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_33_5401148dense_33_5401150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_54009082"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_5401153dense_34_5401155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_54009342"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_5401158dense_35_5401160*
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
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_54009602"
 dense_35/StatefulPartitionedCall?
softmax_11/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
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
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_softmax_11_layer_call_and_return_conditional_losses_54009812
softmax_11/PartitionedCall?
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_5401133*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_22/kernel/Regularizer/SquareSquareCle_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_22/kernel/Regularizer/Square?
+le_net_8/conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_22/kernel/Regularizer/Const?
)le_net_8/conv2d_22/kernel/Regularizer/SumSum0le_net_8/conv2d_22/kernel/Regularizer/Square:y:04le_net_8/conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/Sum?
+le_net_8/conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_22/kernel/Regularizer/mul/x?
)le_net_8/conv2d_22/kernel/Regularizer/mulMul4le_net_8/conv2d_22/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/mul?
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_5401141*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_23/kernel/Regularizer/SquareSquareCle_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_23/kernel/Regularizer/Square?
+le_net_8/conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_23/kernel/Regularizer/Const?
)le_net_8/conv2d_23/kernel/Regularizer/SumSum0le_net_8/conv2d_23/kernel/Regularizer/Square:y:04le_net_8/conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/Sum?
+le_net_8/conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_23/kernel/Regularizer/mul/x?
)le_net_8/conv2d_23/kernel/Regularizer/mulMul4le_net_8/conv2d_23/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/mul?
IdentityIdentity#softmax_11/PartitionedCall:output:0"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall<^le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp<^le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2z
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp2z
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
c
G__inference_softmax_11_layer_call_and_return_conditional_losses_5400981

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
?-
?
#__inference__traced_restore_5401814
file_prefix.
*assignvariableop_le_net_8_conv2d_22_kernel.
*assignvariableop_1_le_net_8_conv2d_22_bias0
,assignvariableop_2_le_net_8_conv2d_23_kernel.
*assignvariableop_3_le_net_8_conv2d_23_bias/
+assignvariableop_4_le_net_8_dense_33_kernel-
)assignvariableop_5_le_net_8_dense_33_bias/
+assignvariableop_6_le_net_8_dense_34_kernel-
)assignvariableop_7_le_net_8_dense_34_bias/
+assignvariableop_8_le_net_8_dense_35_kernel-
)assignvariableop_9_le_net_8_dense_35_bias
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc3/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp*assignvariableop_le_net_8_conv2d_22_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp*assignvariableop_1_le_net_8_conv2d_22_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_le_net_8_conv2d_23_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_le_net_8_conv2d_23_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp+assignvariableop_4_le_net_8_dense_33_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp)assignvariableop_5_le_net_8_dense_33_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp+assignvariableop_6_le_net_8_dense_34_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp)assignvariableop_7_le_net_8_dense_34_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_le_net_8_dense_35_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_le_net_8_dense_35_biasIdentity_9:output:0"/device:CPU:0*
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
?A
?
"__inference__wrapped_model_5400798
input_15
1le_net_8_conv2d_22_conv2d_readvariableop_resource6
2le_net_8_conv2d_22_biasadd_readvariableop_resource5
1le_net_8_conv2d_23_conv2d_readvariableop_resource6
2le_net_8_conv2d_23_biasadd_readvariableop_resource4
0le_net_8_dense_33_matmul_readvariableop_resource5
1le_net_8_dense_33_biasadd_readvariableop_resource4
0le_net_8_dense_34_matmul_readvariableop_resource5
1le_net_8_dense_34_biasadd_readvariableop_resource4
0le_net_8_dense_35_matmul_readvariableop_resource5
1le_net_8_dense_35_biasadd_readvariableop_resource
identity??)le_net_8/conv2d_22/BiasAdd/ReadVariableOp?(le_net_8/conv2d_22/Conv2D/ReadVariableOp?)le_net_8/conv2d_23/BiasAdd/ReadVariableOp?(le_net_8/conv2d_23/Conv2D/ReadVariableOp?(le_net_8/dense_33/BiasAdd/ReadVariableOp?'le_net_8/dense_33/MatMul/ReadVariableOp?(le_net_8/dense_34/BiasAdd/ReadVariableOp?'le_net_8/dense_34/MatMul/ReadVariableOp?(le_net_8/dense_35/BiasAdd/ReadVariableOp?'le_net_8/dense_35/MatMul/ReadVariableOp?
le_net_8/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             2
le_net_8/Const
le_net_8/PadPadinput_1le_net_8/Const:output:0*
T0*/
_output_shapes
:?????????$$2
le_net_8/Pad?
(le_net_8/conv2d_22/Conv2D/ReadVariableOpReadVariableOp1le_net_8_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(le_net_8/conv2d_22/Conv2D/ReadVariableOp?
le_net_8/conv2d_22/Conv2DConv2Dle_net_8/Pad:output:00le_net_8/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
le_net_8/conv2d_22/Conv2D?
)le_net_8/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp2le_net_8_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)le_net_8/conv2d_22/BiasAdd/ReadVariableOp?
le_net_8/conv2d_22/BiasAddBiasAdd"le_net_8/conv2d_22/Conv2D:output:01le_net_8/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
le_net_8/conv2d_22/BiasAdd?
 le_net_8/max_pooling2d_4/MaxPoolMaxPool#le_net_8/conv2d_22/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2"
 le_net_8/max_pooling2d_4/MaxPool?
le_net_8/Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2
le_net_8/Const_1?
le_net_8/Pad_1Pad)le_net_8/max_pooling2d_4/MaxPool:output:0le_net_8/Const_1:output:0*
T0*/
_output_shapes
:?????????2
le_net_8/Pad_1?
(le_net_8/conv2d_23/Conv2D/ReadVariableOpReadVariableOp1le_net_8_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(le_net_8/conv2d_23/Conv2D/ReadVariableOp?
le_net_8/conv2d_23/Conv2DConv2Dle_net_8/Pad_1:output:00le_net_8/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
le_net_8/conv2d_23/Conv2D?
)le_net_8/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp2le_net_8_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)le_net_8/conv2d_23/BiasAdd/ReadVariableOp?
le_net_8/conv2d_23/BiasAddBiasAdd"le_net_8/conv2d_23/Conv2D:output:01le_net_8/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
le_net_8/conv2d_23/BiasAdd?
"le_net_8/max_pooling2d_4/MaxPool_1MaxPool#le_net_8/conv2d_23/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2$
"le_net_8/max_pooling2d_4/MaxPool_1?
le_net_8/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
le_net_8/flatten_11/Const?
le_net_8/flatten_11/ReshapeReshape+le_net_8/max_pooling2d_4/MaxPool_1:output:0"le_net_8/flatten_11/Const:output:0*
T0*(
_output_shapes
:??????????2
le_net_8/flatten_11/Reshape?
'le_net_8/dense_33/MatMul/ReadVariableOpReadVariableOp0le_net_8_dense_33_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02)
'le_net_8/dense_33/MatMul/ReadVariableOp?
le_net_8/dense_33/MatMulMatMul$le_net_8/flatten_11/Reshape:output:0/le_net_8/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
le_net_8/dense_33/MatMul?
(le_net_8/dense_33/BiasAdd/ReadVariableOpReadVariableOp1le_net_8_dense_33_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02*
(le_net_8/dense_33/BiasAdd/ReadVariableOp?
le_net_8/dense_33/BiasAddBiasAdd"le_net_8/dense_33/MatMul:product:00le_net_8/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
le_net_8/dense_33/BiasAdd?
'le_net_8/dense_34/MatMul/ReadVariableOpReadVariableOp0le_net_8_dense_34_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02)
'le_net_8/dense_34/MatMul/ReadVariableOp?
le_net_8/dense_34/MatMulMatMul"le_net_8/dense_33/BiasAdd:output:0/le_net_8/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
le_net_8/dense_34/MatMul?
(le_net_8/dense_34/BiasAdd/ReadVariableOpReadVariableOp1le_net_8_dense_34_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02*
(le_net_8/dense_34/BiasAdd/ReadVariableOp?
le_net_8/dense_34/BiasAddBiasAdd"le_net_8/dense_34/MatMul:product:00le_net_8/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
le_net_8/dense_34/BiasAdd?
'le_net_8/dense_35/MatMul/ReadVariableOpReadVariableOp0le_net_8_dense_35_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02)
'le_net_8/dense_35/MatMul/ReadVariableOp?
le_net_8/dense_35/MatMulMatMul"le_net_8/dense_34/BiasAdd:output:0/le_net_8/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
le_net_8/dense_35/MatMul?
(le_net_8/dense_35/BiasAdd/ReadVariableOpReadVariableOp1le_net_8_dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(le_net_8/dense_35/BiasAdd/ReadVariableOp?
le_net_8/dense_35/BiasAddBiasAdd"le_net_8/dense_35/MatMul:product:00le_net_8/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
le_net_8/dense_35/BiasAdd?
le_net_8/softmax_11/SoftmaxSoftmax"le_net_8/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
le_net_8/softmax_11/Softmax?
IdentityIdentity%le_net_8/softmax_11/Softmax:softmax:0*^le_net_8/conv2d_22/BiasAdd/ReadVariableOp)^le_net_8/conv2d_22/Conv2D/ReadVariableOp*^le_net_8/conv2d_23/BiasAdd/ReadVariableOp)^le_net_8/conv2d_23/Conv2D/ReadVariableOp)^le_net_8/dense_33/BiasAdd/ReadVariableOp(^le_net_8/dense_33/MatMul/ReadVariableOp)^le_net_8/dense_34/BiasAdd/ReadVariableOp(^le_net_8/dense_34/MatMul/ReadVariableOp)^le_net_8/dense_35/BiasAdd/ReadVariableOp(^le_net_8/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2V
)le_net_8/conv2d_22/BiasAdd/ReadVariableOp)le_net_8/conv2d_22/BiasAdd/ReadVariableOp2T
(le_net_8/conv2d_22/Conv2D/ReadVariableOp(le_net_8/conv2d_22/Conv2D/ReadVariableOp2V
)le_net_8/conv2d_23/BiasAdd/ReadVariableOp)le_net_8/conv2d_23/BiasAdd/ReadVariableOp2T
(le_net_8/conv2d_23/Conv2D/ReadVariableOp(le_net_8/conv2d_23/Conv2D/ReadVariableOp2T
(le_net_8/dense_33/BiasAdd/ReadVariableOp(le_net_8/dense_33/BiasAdd/ReadVariableOp2R
'le_net_8/dense_33/MatMul/ReadVariableOp'le_net_8/dense_33/MatMul/ReadVariableOp2T
(le_net_8/dense_34/BiasAdd/ReadVariableOp(le_net_8/dense_34/BiasAdd/ReadVariableOp2R
'le_net_8/dense_34/MatMul/ReadVariableOp'le_net_8/dense_34/MatMul/ReadVariableOp2T
(le_net_8/dense_35/BiasAdd/ReadVariableOp(le_net_8/dense_35/BiasAdd/ReadVariableOp2R
'le_net_8/dense_35/MatMul/ReadVariableOp'le_net_8/dense_35/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
+__inference_conv2d_22_layer_call_fn_5401590

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
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_54008322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????$$::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?N
?
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401349
x,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity?? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp
ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Const^
PadPadxConst:output:0*
T0*/
_output_shapes
:?????????$$2
Pad?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2DPad:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_22/BiasAdd?
max_pooling2d_4/MaxPoolMaxPoolconv2d_22/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1?
Pad_1Pad max_pooling2d_4/MaxPool:output:0Const_1:output:0*
T0*/
_output_shapes
:?????????2
Pad_1?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2DPad_1:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_23/BiasAdd?
max_pooling2d_4/MaxPool_1MaxPoolconv2d_23/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool_1u
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_11/Const?
flatten_11/ReshapeReshape"max_pooling2d_4/MaxPool_1:output:0flatten_11/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_11/Reshape?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMulflatten_11/Reshape:output:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_33/BiasAdd?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_33/BiasAdd:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_34/BiasAdd?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_34/BiasAdd:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/BiasAdd?
softmax_11/SoftmaxSoftmaxdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_11/Softmax?
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_22/kernel/Regularizer/SquareSquareCle_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_22/kernel/Regularizer/Square?
+le_net_8/conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_22/kernel/Regularizer/Const?
)le_net_8/conv2d_22/kernel/Regularizer/SumSum0le_net_8/conv2d_22/kernel/Regularizer/Square:y:04le_net_8/conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/Sum?
+le_net_8/conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_22/kernel/Regularizer/mul/x?
)le_net_8/conv2d_22/kernel/Regularizer/mulMul4le_net_8/conv2d_22/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/mul?
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_23/kernel/Regularizer/SquareSquareCle_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_23/kernel/Regularizer/Square?
+le_net_8/conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_23/kernel/Regularizer/Const?
)le_net_8/conv2d_23/kernel/Regularizer/SumSum0le_net_8/conv2d_23/kernel/Regularizer/Square:y:04le_net_8/conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/Sum?
+le_net_8/conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_23/kernel/Regularizer/mul/x?
)le_net_8/conv2d_23/kernel/Regularizer/mulMul4le_net_8/conv2d_23/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/mul?
IdentityIdentitysoftmax_11/Softmax:softmax:0!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp<^le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp<^le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2z
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp2z
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
?
*__inference_le_net_8_layer_call_fn_5401559
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
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_le_net_8_layer_call_and_return_conditional_losses_54011772
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
?
?
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5400832

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd?
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_22/kernel/Regularizer/SquareSquareCle_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_22/kernel/Regularizer/Square?
+le_net_8/conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_22/kernel/Regularizer/Const?
)le_net_8/conv2d_22/kernel/Regularizer/SumSum0le_net_8/conv2d_22/kernel/Regularizer/Square:y:04le_net_8/conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/Sum?
+le_net_8/conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_22/kernel/Regularizer/mul/x?
)le_net_8/conv2d_22/kernel/Regularizer/mulMul4le_net_8/conv2d_22/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/mul?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp<^le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????$$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?C
?
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401103
x
conv2d_22_5401059
conv2d_22_5401061
conv2d_23_5401067
conv2d_23_5401069
dense_33_5401074
dense_33_5401076
dense_34_5401079
dense_34_5401081
dense_35_5401084
dense_35_5401086
identity??!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp
ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Const^
PadPadxConst:output:0*
T0*/
_output_shapes
:?????????$$2
Pad?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCallPad:output:0conv2d_22_5401059conv2d_22_5401061*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_22_layer_call_and_return_conditional_losses_54008322#
!conv2d_22/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_54008042!
max_pooling2d_4/PartitionedCall?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1?
Pad_1Pad(max_pooling2d_4/PartitionedCall:output:0Const_1:output:0*
T0*/
_output_shapes
:?????????2
Pad_1?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCallPad_1:output:0conv2d_23_5401067conv2d_23_5401069*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_23_layer_call_and_return_conditional_losses_54008672#
!conv2d_23/StatefulPartitionedCall?
!max_pooling2d_4/PartitionedCall_1PartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_54008042#
!max_pooling2d_4/PartitionedCall_1?
flatten_11/PartitionedCallPartitionedCall*max_pooling2d_4/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_54008902
flatten_11/PartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_33_5401074dense_33_5401076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_54009082"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_5401079dense_34_5401081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_54009342"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_5401084dense_35_5401086*
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
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_54009602"
 dense_35/StatefulPartitionedCall?
softmax_11/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
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
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_softmax_11_layer_call_and_return_conditional_losses_54009812
softmax_11/PartitionedCall?
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_22_5401059*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_22/kernel/Regularizer/SquareSquareCle_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_22/kernel/Regularizer/Square?
+le_net_8/conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_22/kernel/Regularizer/Const?
)le_net_8/conv2d_22/kernel/Regularizer/SumSum0le_net_8/conv2d_22/kernel/Regularizer/Square:y:04le_net_8/conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/Sum?
+le_net_8/conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_22/kernel/Regularizer/mul/x?
)le_net_8/conv2d_22/kernel/Regularizer/mulMul4le_net_8/conv2d_22/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/mul?
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_23_5401067*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_23/kernel/Regularizer/SquareSquareCle_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_23/kernel/Regularizer/Square?
+le_net_8/conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_23/kernel/Regularizer/Const?
)le_net_8/conv2d_23/kernel/Regularizer/SumSum0le_net_8/conv2d_23/kernel/Regularizer/Square:y:04le_net_8/conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/Sum?
+le_net_8/conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_23/kernel/Regularizer/mul/x?
)le_net_8/conv2d_23/kernel/Regularizer/mulMul4le_net_8/conv2d_23/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/mul?
IdentityIdentity#softmax_11/PartitionedCall:output:0"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall<^le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp<^le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2z
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp2z
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
h
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5400804

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
?
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_5401684

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_softmax_11_layer_call_fn_5401699

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
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_softmax_11_layer_call_and_return_conditional_losses_54009812
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
?
c
G__inference_softmax_11_layer_call_and_return_conditional_losses_5401694

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
?N
?
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401509
input_1,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity?? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp
ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Constd
PadPadinput_1Const:output:0*
T0*/
_output_shapes
:?????????$$2
Pad?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2DPad:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_22/BiasAdd?
max_pooling2d_4/MaxPoolMaxPoolconv2d_22/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1?
Pad_1Pad max_pooling2d_4/MaxPool:output:0Const_1:output:0*
T0*/
_output_shapes
:?????????2
Pad_1?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2DPad_1:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_23/BiasAdd?
max_pooling2d_4/MaxPool_1MaxPoolconv2d_23/BiasAdd:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool_1u
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_11/Const?
flatten_11/ReshapeReshape"max_pooling2d_4/MaxPool_1:output:0flatten_11/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_11/Reshape?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMulflatten_11/Reshape:output:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_33/BiasAdd?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_33/BiasAdd:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T2
dense_34/BiasAdd?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_34/BiasAdd:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/BiasAdd?
softmax_11/SoftmaxSoftmaxdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_11/Softmax?
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_22/kernel/Regularizer/SquareSquareCle_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_22/kernel/Regularizer/Square?
+le_net_8/conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_22/kernel/Regularizer/Const?
)le_net_8/conv2d_22/kernel/Regularizer/SumSum0le_net_8/conv2d_22/kernel/Regularizer/Square:y:04le_net_8/conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/Sum?
+le_net_8/conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_22/kernel/Regularizer/mul/x?
)le_net_8/conv2d_22/kernel/Regularizer/mulMul4le_net_8/conv2d_22/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_22/kernel/Regularizer/mul?
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp?
,le_net_8/conv2d_23/kernel/Regularizer/SquareSquareCle_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,le_net_8/conv2d_23/kernel/Regularizer/Square?
+le_net_8/conv2d_23/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+le_net_8/conv2d_23/kernel/Regularizer/Const?
)le_net_8/conv2d_23/kernel/Regularizer/SumSum0le_net_8/conv2d_23/kernel/Regularizer/Square:y:04le_net_8/conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/Sum?
+le_net_8/conv2d_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+le_net_8/conv2d_23/kernel/Regularizer/mul/x?
)le_net_8/conv2d_23/kernel/Regularizer/mulMul4le_net_8/conv2d_23/kernel/Regularizer/mul/x:output:02le_net_8/conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)le_net_8/conv2d_23/kernel/Regularizer/mul?
IdentityIdentitysoftmax_11/Softmax:softmax:0!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp<^le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp<^le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2z
;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_22/kernel/Regularizer/Square/ReadVariableOp2z
;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp;le_net_8/conv2d_23/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?	
?
E__inference_dense_35_layer_call_and_return_conditional_losses_5400960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
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
:?????????T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?

*__inference_dense_35_layer_call_fn_5401678

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
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_54009602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????T::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????T
 
_user_specified_nameinputs
?"
?
 __inference__traced_save_5401774
file_prefix8
4savev2_le_net_8_conv2d_22_kernel_read_readvariableop6
2savev2_le_net_8_conv2d_22_bias_read_readvariableop8
4savev2_le_net_8_conv2d_23_kernel_read_readvariableop6
2savev2_le_net_8_conv2d_23_bias_read_readvariableop7
3savev2_le_net_8_dense_33_kernel_read_readvariableop5
1savev2_le_net_8_dense_33_bias_read_readvariableop7
3savev2_le_net_8_dense_34_kernel_read_readvariableop5
1savev2_le_net_8_dense_34_bias_read_readvariableop7
3savev2_le_net_8_dense_35_kernel_read_readvariableop5
1savev2_le_net_8_dense_35_bias_read_readvariableop
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
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc3/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_le_net_8_conv2d_22_kernel_read_readvariableop2savev2_le_net_8_conv2d_22_bias_read_readvariableop4savev2_le_net_8_conv2d_23_kernel_read_readvariableop2savev2_le_net_8_conv2d_23_bias_read_readvariableop3savev2_le_net_8_dense_33_kernel_read_readvariableop1savev2_le_net_8_dense_33_bias_read_readvariableop3savev2_le_net_8_dense_34_kernel_read_readvariableop1savev2_le_net_8_dense_34_bias_read_readvariableop3savev2_le_net_8_dense_35_kernel_read_readvariableop1savev2_le_net_8_dense_35_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
e: :::::	?x:x:xT:T:T
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?x: 

_output_shapes
:x:$ 

_output_shapes

:xT: 

_output_shapes
:T:$	 

_output_shapes

:T
: 


_output_shapes
:
:

_output_shapes
: "?L
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
fc1
fc2
fc3
flatten
softmax
tanh
	sigmoid

pool
trainable_variables
regularization_losses
	variables
	keras_api

signatures
i_default_save_signature
*j&call_and_return_all_conditional_losses
k__call__"?
_tf_keras_model?{"class_name": "LeNet", "name": "le_net_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "LeNet"}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 36, 36, 3]}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 18, 18, 6]}}
?

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
*p&call_and_return_all_conditional_losses
q__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 120, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 1024]}}
?

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
*r&call_and_return_all_conditional_losses
s__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 84, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 120]}}
?

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
*t&call_and_return_all_conditional_losses
u__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 84}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 84]}}
?
.trainable_variables
/regularization_losses
0	variables
1	keras_api
*v&call_and_return_all_conditional_losses
w__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
2trainable_variables
3regularization_losses
4	variables
5	keras_api
*x&call_and_return_all_conditional_losses
y__call__"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax_11", "trainable": true, "dtype": "float32", "axis": -1}}
?
6	keras_api"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "tanh"}}
?
7	keras_api"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}
?
8trainable_variables
9regularization_losses
:	variables
;	keras_api
*z&call_and_return_all_conditional_losses
{__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
f
0
1
2
3
4
5
"6
#7
(8
)9"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
f
0
1
2
3
4
5
"6
#7
(8
)9"
trackable_list_wrapper
?

<layers
=layer_metrics
trainable_variables
regularization_losses
>layer_regularization_losses
?metrics
	variables
@non_trainable_variables
k__call__
i_default_save_signature
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
,
~serving_default"
signature_map
3:12le_net_8/conv2d_22/kernel
%:#2le_net_8/conv2d_22/bias
.
0
1"
trackable_list_wrapper
'
|0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Alayers
Blayer_metrics
trainable_variables
regularization_losses
Clayer_regularization_losses
Dmetrics
	variables
Enon_trainable_variables
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
3:12le_net_8/conv2d_23/kernel
%:#2le_net_8/conv2d_23/bias
.
0
1"
trackable_list_wrapper
'
}0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Flayers
Glayer_metrics
trainable_variables
regularization_losses
Hlayer_regularization_losses
Imetrics
	variables
Jnon_trainable_variables
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
+:)	?x2le_net_8/dense_33/kernel
$:"x2le_net_8/dense_33/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Klayers
Llayer_metrics
trainable_variables
regularization_losses
Mlayer_regularization_losses
Nmetrics
 	variables
Onon_trainable_variables
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
*:(xT2le_net_8/dense_34/kernel
$:"T2le_net_8/dense_34/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?

Players
Qlayer_metrics
$trainable_variables
%regularization_losses
Rlayer_regularization_losses
Smetrics
&	variables
Tnon_trainable_variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
*:(T
2le_net_8/dense_35/kernel
$:"
2le_net_8/dense_35/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?

Ulayers
Vlayer_metrics
*trainable_variables
+regularization_losses
Wlayer_regularization_losses
Xmetrics
,	variables
Ynon_trainable_variables
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Zlayers
[layer_metrics
.trainable_variables
/regularization_losses
\layer_regularization_losses
]metrics
0	variables
^non_trainable_variables
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

_layers
`layer_metrics
2trainable_variables
3regularization_losses
alayer_regularization_losses
bmetrics
4	variables
cnon_trainable_variables
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

dlayers
elayer_metrics
8trainable_variables
9regularization_losses
flayer_regularization_losses
gmetrics
:	variables
hnon_trainable_variables
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
f
0
1
2
3
4
5
6
7
	8

9"
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
'
|0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
}0"
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
?2?
"__inference__wrapped_model_5400798?
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
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401294
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401349
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401509
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401454?
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
?2?
*__inference_le_net_8_layer_call_fn_5401559
*__inference_le_net_8_layer_call_fn_5401399
*__inference_le_net_8_layer_call_fn_5401534
*__inference_le_net_8_layer_call_fn_5401374?
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
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5401581?
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
+__inference_conv2d_22_layer_call_fn_5401590?
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
F__inference_conv2d_23_layer_call_and_return_conditional_losses_5401612?
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
+__inference_conv2d_23_layer_call_fn_5401621?
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
E__inference_dense_33_layer_call_and_return_conditional_losses_5401631?
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
*__inference_dense_33_layer_call_fn_5401640?
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
E__inference_dense_34_layer_call_and_return_conditional_losses_5401650?
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
*__inference_dense_34_layer_call_fn_5401659?
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
E__inference_dense_35_layer_call_and_return_conditional_losses_5401669?
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
*__inference_dense_35_layer_call_fn_5401678?
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
G__inference_flatten_11_layer_call_and_return_conditional_losses_5401684?
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
,__inference_flatten_11_layer_call_fn_5401689?
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
G__inference_softmax_11_layer_call_and_return_conditional_losses_5401694?
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
,__inference_softmax_11_layer_call_fn_5401699?
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
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5400804?
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
1__inference_max_pooling2d_4_layer_call_fn_5400810?
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
__inference_loss_fn_0_5401710?
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
__inference_loss_fn_1_5401721?
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
%__inference_signature_wrapper_5401239input_1"?
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
"__inference__wrapped_model_5400798{
"#()8?5
.?+
)?&
input_1?????????  
? "3?0
.
output_1"?
output_1?????????
?
F__inference_conv2d_22_layer_call_and_return_conditional_losses_5401581l7?4
-?*
(?%
inputs?????????$$
? "-?*
#? 
0?????????  
? ?
+__inference_conv2d_22_layer_call_fn_5401590_7?4
-?*
(?%
inputs?????????$$
? " ??????????  ?
F__inference_conv2d_23_layer_call_and_return_conditional_losses_5401612l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
+__inference_conv2d_23_layer_call_fn_5401621_7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_dense_33_layer_call_and_return_conditional_losses_5401631]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????x
? ~
*__inference_dense_33_layer_call_fn_5401640P0?-
&?#
!?
inputs??????????
? "??????????x?
E__inference_dense_34_layer_call_and_return_conditional_losses_5401650\"#/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????T
? }
*__inference_dense_34_layer_call_fn_5401659O"#/?,
%?"
 ?
inputs?????????x
? "??????????T?
E__inference_dense_35_layer_call_and_return_conditional_losses_5401669\()/?,
%?"
 ?
inputs?????????T
? "%?"
?
0?????????

? }
*__inference_dense_35_layer_call_fn_5401678O()/?,
%?"
 ?
inputs?????????T
? "??????????
?
G__inference_flatten_11_layer_call_and_return_conditional_losses_5401684a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
,__inference_flatten_11_layer_call_fn_5401689T7?4
-?*
(?%
inputs?????????
? "????????????
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401294k
"#()6?3
,?)
#? 
x?????????  
p
? "%?"
?
0?????????

? ?
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401349k
"#()6?3
,?)
#? 
x?????????  
p 
? "%?"
?
0?????????

? ?
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401454q
"#()<?9
2?/
)?&
input_1?????????  
p
? "%?"
?
0?????????

? ?
E__inference_le_net_8_layer_call_and_return_conditional_losses_5401509q
"#()<?9
2?/
)?&
input_1?????????  
p 
? "%?"
?
0?????????

? ?
*__inference_le_net_8_layer_call_fn_5401374^
"#()6?3
,?)
#? 
x?????????  
p
? "??????????
?
*__inference_le_net_8_layer_call_fn_5401399^
"#()6?3
,?)
#? 
x?????????  
p 
? "??????????
?
*__inference_le_net_8_layer_call_fn_5401534d
"#()<?9
2?/
)?&
input_1?????????  
p
? "??????????
?
*__inference_le_net_8_layer_call_fn_5401559d
"#()<?9
2?/
)?&
input_1?????????  
p 
? "??????????
<
__inference_loss_fn_0_5401710?

? 
? "? <
__inference_loss_fn_1_5401721?

? 
? "? ?
L__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_5400804?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_4_layer_call_fn_5400810?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_signature_wrapper_5401239?
"#()C?@
? 
9?6
4
input_1)?&
input_1?????????  "3?0
.
output_1"?
output_1?????????
?
G__inference_softmax_11_layer_call_and_return_conditional_losses_5401694\3?0
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
? 
,__inference_softmax_11_layer_call_fn_5401699O3?0
)?&
 ?
inputs?????????


 
? "??????????
