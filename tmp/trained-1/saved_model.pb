??
??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
(three_layer_conv_net_13/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(three_layer_conv_net_13/conv2d_26/kernel
?
<three_layer_conv_net_13/conv2d_26/kernel/Read/ReadVariableOpReadVariableOp(three_layer_conv_net_13/conv2d_26/kernel*&
_output_shapes
:*
dtype0
?
&three_layer_conv_net_13/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&three_layer_conv_net_13/conv2d_26/bias
?
:three_layer_conv_net_13/conv2d_26/bias/Read/ReadVariableOpReadVariableOp&three_layer_conv_net_13/conv2d_26/bias*
_output_shapes
:*
dtype0
?
(three_layer_conv_net_13/conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(three_layer_conv_net_13/conv2d_27/kernel
?
<three_layer_conv_net_13/conv2d_27/kernel/Read/ReadVariableOpReadVariableOp(three_layer_conv_net_13/conv2d_27/kernel*&
_output_shapes
:*
dtype0
?
&three_layer_conv_net_13/conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&three_layer_conv_net_13/conv2d_27/bias
?
:three_layer_conv_net_13/conv2d_27/bias/Read/ReadVariableOpReadVariableOp&three_layer_conv_net_13/conv2d_27/bias*
_output_shapes
:*
dtype0
?
'three_layer_conv_net_13/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@
*8
shared_name)'three_layer_conv_net_13/dense_13/kernel
?
;three_layer_conv_net_13/dense_13/kernel/Read/ReadVariableOpReadVariableOp'three_layer_conv_net_13/dense_13/kernel*
_output_shapes
:	?@
*
dtype0
?
%three_layer_conv_net_13/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%three_layer_conv_net_13/dense_13/bias
?
9three_layer_conv_net_13/dense_13/bias/Read/ReadVariableOpReadVariableOp%three_layer_conv_net_13/dense_13/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
	conv1
	conv2
fc
flatten
softmax
regularization_losses
	variables
trainable_variables
		keras_api


signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
 	keras_api
R
!regularization_losses
"	variables
#trainable_variables
$	keras_api
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?
regularization_losses

%layers
	variables
&layer_metrics
'metrics
trainable_variables
(non_trainable_variables
)layer_regularization_losses
 
ec
VARIABLE_VALUE(three_layer_conv_net_13/conv2d_26/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE&three_layer_conv_net_13/conv2d_26/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

*layers
+layer_metrics
	variables
,metrics
trainable_variables
-non_trainable_variables
.layer_regularization_losses
ec
VARIABLE_VALUE(three_layer_conv_net_13/conv2d_27/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE&three_layer_conv_net_13/conv2d_27/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

/layers
0layer_metrics
	variables
1metrics
trainable_variables
2non_trainable_variables
3layer_regularization_losses
a_
VARIABLE_VALUE'three_layer_conv_net_13/dense_13/kernel$fc/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE%three_layer_conv_net_13/dense_13/bias"fc/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

4layers
5layer_metrics
	variables
6metrics
trainable_variables
7non_trainable_variables
8layer_regularization_losses
 
 
 
?
regularization_losses

9layers
:layer_metrics
	variables
;metrics
trainable_variables
<non_trainable_variables
=layer_regularization_losses
 
 
 
?
!regularization_losses

>layers
?layer_metrics
"	variables
@metrics
#trainable_variables
Anon_trainable_variables
Blayer_regularization_losses
#
0
1
2
3
4
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1(three_layer_conv_net_13/conv2d_26/kernel&three_layer_conv_net_13/conv2d_26/bias(three_layer_conv_net_13/conv2d_27/kernel&three_layer_conv_net_13/conv2d_27/bias'three_layer_conv_net_13/dense_13/kernel%three_layer_conv_net_13/dense_13/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1054989
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename<three_layer_conv_net_13/conv2d_26/kernel/Read/ReadVariableOp:three_layer_conv_net_13/conv2d_26/bias/Read/ReadVariableOp<three_layer_conv_net_13/conv2d_27/kernel/Read/ReadVariableOp:three_layer_conv_net_13/conv2d_27/bias/Read/ReadVariableOp;three_layer_conv_net_13/dense_13/kernel/Read/ReadVariableOp9three_layer_conv_net_13/dense_13/bias/Read/ReadVariableOpConst*
Tin

2*
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
 __inference__traced_save_1055302
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(three_layer_conv_net_13/conv2d_26/kernel&three_layer_conv_net_13/conv2d_26/bias(three_layer_conv_net_13/conv2d_27/kernel&three_layer_conv_net_13/conv2d_27/bias'three_layer_conv_net_13/dense_13/kernel%three_layer_conv_net_13/dense_13/bias*
Tin
	2*
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
#__inference__traced_restore_1055330??
?%
?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055147
x,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity?? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp
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
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2DPad:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_26/Relu?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1
Pad_1Padconv2d_26/Relu:activations:0Const_1:output:0*
T0*/
_output_shapes
:?????????""2
Pad_1?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2DPad_1:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/Reluu
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_13/Const?
flatten_13/ReshapeReshapeconv2d_27/Relu:activations:0flatten_13/Const:output:0*
T0*(
_output_shapes
:??????????@2
flatten_13/Reshape?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@
*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMulflatten_13/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_13/BiasAdd?
softmax_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_13/Softmax?
IdentityIdentitysoftmax_13/Softmax:softmax:0!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
?
%__inference_signature_wrapper_1054989
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_10547442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?%
?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055020
input_1,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity?? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp
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
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2DPad:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_26/Relu?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1
Pad_1Padconv2d_26/Relu:activations:0Const_1:output:0*
T0*/
_output_shapes
:?????????""2
Pad_1?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2DPad_1:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/Reluu
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_13/Const?
flatten_13/ReshapeReshapeconv2d_27/Relu:activations:0flatten_13/Const:output:0*
T0*(
_output_shapes
:??????????@2
flatten_13/Reshape?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@
*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMulflatten_13/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_13/BiasAdd?
softmax_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_13/Softmax?
IdentityIdentitysoftmax_13/Softmax:softmax:0!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
c
G__inference_flatten_13_layer_call_and_return_conditional_losses_1054812

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
9__inference_three_layer_conv_net_13_layer_call_fn_1055164
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_10549132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?	
?
E__inference_dense_13_layer_call_and_return_conditional_losses_1054830

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@
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
identityIdentity:output:0*/
_input_shapes
:??????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
 __inference__traced_save_1055302
file_prefixG
Csavev2_three_layer_conv_net_13_conv2d_26_kernel_read_readvariableopE
Asavev2_three_layer_conv_net_13_conv2d_26_bias_read_readvariableopG
Csavev2_three_layer_conv_net_13_conv2d_27_kernel_read_readvariableopE
Asavev2_three_layer_conv_net_13_conv2d_27_bias_read_readvariableopF
Bsavev2_three_layer_conv_net_13_dense_13_kernel_read_readvariableopD
@savev2_three_layer_conv_net_13_dense_13_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB$fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Csavev2_three_layer_conv_net_13_conv2d_26_kernel_read_readvariableopAsavev2_three_layer_conv_net_13_conv2d_26_bias_read_readvariableopCsavev2_three_layer_conv_net_13_conv2d_27_kernel_read_readvariableopAsavev2_three_layer_conv_net_13_conv2d_27_bias_read_readvariableopBsavev2_three_layer_conv_net_13_dense_13_kernel_read_readvariableop@savev2_three_layer_conv_net_13_dense_13_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
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

identity_1Identity_1:output:0*X
_input_shapesG
E: :::::	?@
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?@
: 

_output_shapes
:
:

_output_shapes
: 
?8
?
"__inference__wrapped_model_1054744
input_1D
@three_layer_conv_net_13_conv2d_26_conv2d_readvariableop_resourceE
Athree_layer_conv_net_13_conv2d_26_biasadd_readvariableop_resourceD
@three_layer_conv_net_13_conv2d_27_conv2d_readvariableop_resourceE
Athree_layer_conv_net_13_conv2d_27_biasadd_readvariableop_resourceC
?three_layer_conv_net_13_dense_13_matmul_readvariableop_resourceD
@three_layer_conv_net_13_dense_13_biasadd_readvariableop_resource
identity??8three_layer_conv_net_13/conv2d_26/BiasAdd/ReadVariableOp?7three_layer_conv_net_13/conv2d_26/Conv2D/ReadVariableOp?8three_layer_conv_net_13/conv2d_27/BiasAdd/ReadVariableOp?7three_layer_conv_net_13/conv2d_27/Conv2D/ReadVariableOp?7three_layer_conv_net_13/dense_13/BiasAdd/ReadVariableOp?6three_layer_conv_net_13/dense_13/MatMul/ReadVariableOp?
three_layer_conv_net_13/ConstConst*
_output_shapes

:*
dtype0*9
value0B."                             2
three_layer_conv_net_13/Const?
three_layer_conv_net_13/PadPadinput_1&three_layer_conv_net_13/Const:output:0*
T0*/
_output_shapes
:?????????$$2
three_layer_conv_net_13/Pad?
7three_layer_conv_net_13/conv2d_26/Conv2D/ReadVariableOpReadVariableOp@three_layer_conv_net_13_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7three_layer_conv_net_13/conv2d_26/Conv2D/ReadVariableOp?
(three_layer_conv_net_13/conv2d_26/Conv2DConv2D$three_layer_conv_net_13/Pad:output:0?three_layer_conv_net_13/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2*
(three_layer_conv_net_13/conv2d_26/Conv2D?
8three_layer_conv_net_13/conv2d_26/BiasAdd/ReadVariableOpReadVariableOpAthree_layer_conv_net_13_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8three_layer_conv_net_13/conv2d_26/BiasAdd/ReadVariableOp?
)three_layer_conv_net_13/conv2d_26/BiasAddBiasAdd1three_layer_conv_net_13/conv2d_26/Conv2D:output:0@three_layer_conv_net_13/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2+
)three_layer_conv_net_13/conv2d_26/BiasAdd?
&three_layer_conv_net_13/conv2d_26/ReluRelu2three_layer_conv_net_13/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2(
&three_layer_conv_net_13/conv2d_26/Relu?
three_layer_conv_net_13/Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2!
three_layer_conv_net_13/Const_1?
three_layer_conv_net_13/Pad_1Pad4three_layer_conv_net_13/conv2d_26/Relu:activations:0(three_layer_conv_net_13/Const_1:output:0*
T0*/
_output_shapes
:?????????""2
three_layer_conv_net_13/Pad_1?
7three_layer_conv_net_13/conv2d_27/Conv2D/ReadVariableOpReadVariableOp@three_layer_conv_net_13_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7three_layer_conv_net_13/conv2d_27/Conv2D/ReadVariableOp?
(three_layer_conv_net_13/conv2d_27/Conv2DConv2D&three_layer_conv_net_13/Pad_1:output:0?three_layer_conv_net_13/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2*
(three_layer_conv_net_13/conv2d_27/Conv2D?
8three_layer_conv_net_13/conv2d_27/BiasAdd/ReadVariableOpReadVariableOpAthree_layer_conv_net_13_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8three_layer_conv_net_13/conv2d_27/BiasAdd/ReadVariableOp?
)three_layer_conv_net_13/conv2d_27/BiasAddBiasAdd1three_layer_conv_net_13/conv2d_27/Conv2D:output:0@three_layer_conv_net_13/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2+
)three_layer_conv_net_13/conv2d_27/BiasAdd?
&three_layer_conv_net_13/conv2d_27/ReluRelu2three_layer_conv_net_13/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2(
&three_layer_conv_net_13/conv2d_27/Relu?
(three_layer_conv_net_13/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2*
(three_layer_conv_net_13/flatten_13/Const?
*three_layer_conv_net_13/flatten_13/ReshapeReshape4three_layer_conv_net_13/conv2d_27/Relu:activations:01three_layer_conv_net_13/flatten_13/Const:output:0*
T0*(
_output_shapes
:??????????@2,
*three_layer_conv_net_13/flatten_13/Reshape?
6three_layer_conv_net_13/dense_13/MatMul/ReadVariableOpReadVariableOp?three_layer_conv_net_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@
*
dtype028
6three_layer_conv_net_13/dense_13/MatMul/ReadVariableOp?
'three_layer_conv_net_13/dense_13/MatMulMatMul3three_layer_conv_net_13/flatten_13/Reshape:output:0>three_layer_conv_net_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2)
'three_layer_conv_net_13/dense_13/MatMul?
7three_layer_conv_net_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp@three_layer_conv_net_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype029
7three_layer_conv_net_13/dense_13/BiasAdd/ReadVariableOp?
(three_layer_conv_net_13/dense_13/BiasAddBiasAdd1three_layer_conv_net_13/dense_13/MatMul:product:0?three_layer_conv_net_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2*
(three_layer_conv_net_13/dense_13/BiasAdd?
*three_layer_conv_net_13/softmax_13/SoftmaxSoftmax1three_layer_conv_net_13/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2,
*three_layer_conv_net_13/softmax_13/Softmax?
IdentityIdentity4three_layer_conv_net_13/softmax_13/Softmax:softmax:09^three_layer_conv_net_13/conv2d_26/BiasAdd/ReadVariableOp8^three_layer_conv_net_13/conv2d_26/Conv2D/ReadVariableOp9^three_layer_conv_net_13/conv2d_27/BiasAdd/ReadVariableOp8^three_layer_conv_net_13/conv2d_27/Conv2D/ReadVariableOp8^three_layer_conv_net_13/dense_13/BiasAdd/ReadVariableOp7^three_layer_conv_net_13/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::2t
8three_layer_conv_net_13/conv2d_26/BiasAdd/ReadVariableOp8three_layer_conv_net_13/conv2d_26/BiasAdd/ReadVariableOp2r
7three_layer_conv_net_13/conv2d_26/Conv2D/ReadVariableOp7three_layer_conv_net_13/conv2d_26/Conv2D/ReadVariableOp2t
8three_layer_conv_net_13/conv2d_27/BiasAdd/ReadVariableOp8three_layer_conv_net_13/conv2d_27/BiasAdd/ReadVariableOp2r
7three_layer_conv_net_13/conv2d_27/Conv2D/ReadVariableOp7three_layer_conv_net_13/conv2d_27/Conv2D/ReadVariableOp2r
7three_layer_conv_net_13/dense_13/BiasAdd/ReadVariableOp7three_layer_conv_net_13/dense_13/BiasAdd/ReadVariableOp2p
6three_layer_conv_net_13/dense_13/MatMul/ReadVariableOp6three_layer_conv_net_13/dense_13/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?

?
F__inference_conv2d_27_layer_call_and_return_conditional_losses_1055212

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????""::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????""
 
_user_specified_nameinputs
?
H
,__inference_flatten_13_layer_call_fn_1055251

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
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_13_layer_call_and_return_conditional_losses_10548122
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
H
,__inference_softmax_13_layer_call_fn_1055261

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
G__inference_softmax_13_layer_call_and_return_conditional_losses_10548512
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
G__inference_softmax_13_layer_call_and_return_conditional_losses_1055256

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
?
?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1054955
x
conv2d_26_1054935
conv2d_26_1054937
conv2d_27_1054942
conv2d_27_1054944
dense_13_1054948
dense_13_1054950
identity??!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall? dense_13/StatefulPartitionedCall
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
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCallPad:output:0conv2d_26_1054935conv2d_26_1054937*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_10547612#
!conv2d_26/StatefulPartitionedCall?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1?
Pad_1Pad*conv2d_26/StatefulPartitionedCall:output:0Const_1:output:0*
T0*/
_output_shapes
:?????????""2
Pad_1?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallPad_1:output:0conv2d_27_1054942conv2d_27_1054944*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_27_layer_call_and_return_conditional_losses_10547902#
!conv2d_27/StatefulPartitionedCall?
flatten_13/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_13_layer_call_and_return_conditional_losses_10548122
flatten_13/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_13_1054948dense_13_1054950*
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
E__inference_dense_13_layer_call_and_return_conditional_losses_10548302"
 dense_13/StatefulPartitionedCall?
softmax_13/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
G__inference_softmax_13_layer_call_and_return_conditional_losses_10548512
softmax_13/PartitionedCall?
IdentityIdentity#softmax_13/PartitionedCall:output:0"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?%
?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055116
x,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity?? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp
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
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2DPad:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_26/Relu?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1
Pad_1Padconv2d_26/Relu:activations:0Const_1:output:0*
T0*/
_output_shapes
:?????????""2
Pad_1?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2DPad_1:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/Reluu
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_13/Const?
flatten_13/ReshapeReshapeconv2d_27/Relu:activations:0flatten_13/Const:output:0*
T0*(
_output_shapes
:??????????@2
flatten_13/Reshape?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@
*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMulflatten_13/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_13/BiasAdd?
softmax_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_13/Softmax?
IdentityIdentitysoftmax_13/Softmax:softmax:0!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
c
G__inference_flatten_13_layer_call_and_return_conditional_losses_1055246

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?%
?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055051
input_1,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identity?? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp
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
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2DPad:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_26/Relu?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1
Pad_1Padconv2d_26/Relu:activations:0Const_1:output:0*
T0*/
_output_shapes
:?????????""2
Pad_1?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2DPad_1:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_27/Reluu
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
flatten_13/Const?
flatten_13/ReshapeReshapeconv2d_27/Relu:activations:0flatten_13/Const:output:0*
T0*(
_output_shapes
:??????????@2
flatten_13/Reshape?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@
*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMulflatten_13/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_13/BiasAdd?
softmax_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_13/Softmax?
IdentityIdentitysoftmax_13/Softmax:softmax:0!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?	
?
E__inference_dense_13_layer_call_and_return_conditional_losses_1055231

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@
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
identityIdentity:output:0*/
_input_shapes
:??????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?

?
F__inference_conv2d_27_layer_call_and_return_conditional_losses_1054790

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????""::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????""
 
_user_specified_nameinputs
?

?
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1054761

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????$$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?
?
+__inference_conv2d_27_layer_call_fn_1055221

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
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_27_layer_call_and_return_conditional_losses_10547902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????""::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????""
 
_user_specified_nameinputs
?
?
+__inference_conv2d_26_layer_call_fn_1055201

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
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_10547612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????$$::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?

*__inference_dense_13_layer_call_fn_1055240

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
E__inference_dense_13_layer_call_and_return_conditional_losses_10548302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????@::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
9__inference_three_layer_conv_net_13_layer_call_fn_1055181
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_10549552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
c
G__inference_softmax_13_layer_call_and_return_conditional_losses_1054851

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
?
?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1054913
x
conv2d_26_1054893
conv2d_26_1054895
conv2d_27_1054900
conv2d_27_1054902
dense_13_1054906
dense_13_1054908
identity??!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall? dense_13/StatefulPartitionedCall
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
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCallPad:output:0conv2d_26_1054893conv2d_26_1054895*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_10547612#
!conv2d_26/StatefulPartitionedCall?
Const_1Const*
_output_shapes

:*
dtype0*9
value0B."                             2	
Const_1?
Pad_1Pad*conv2d_26/StatefulPartitionedCall:output:0Const_1:output:0*
T0*/
_output_shapes
:?????????""2
Pad_1?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallPad_1:output:0conv2d_27_1054900conv2d_27_1054902*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_27_layer_call_and_return_conditional_losses_10547902#
!conv2d_27/StatefulPartitionedCall?
flatten_13/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_13_layer_call_and_return_conditional_losses_10548122
flatten_13/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_13_1054906dense_13_1054908*
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
E__inference_dense_13_layer_call_and_return_conditional_losses_10548302"
 dense_13/StatefulPartitionedCall?
softmax_13/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
G__inference_softmax_13_layer_call_and_return_conditional_losses_10548512
softmax_13/PartitionedCall?
IdentityIdentity#softmax_13/PartitionedCall:output:0"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
?
9__inference_three_layer_conv_net_13_layer_call_fn_1055068
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_10549132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
9__inference_three_layer_conv_net_13_layer_call_fn_1055085
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_10549552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????  ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
#__inference__traced_restore_1055330
file_prefix=
9assignvariableop_three_layer_conv_net_13_conv2d_26_kernel=
9assignvariableop_1_three_layer_conv_net_13_conv2d_26_bias?
;assignvariableop_2_three_layer_conv_net_13_conv2d_27_kernel=
9assignvariableop_3_three_layer_conv_net_13_conv2d_27_bias>
:assignvariableop_4_three_layer_conv_net_13_dense_13_kernel<
8assignvariableop_5_three_layer_conv_net_13_dense_13_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB$fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB"fc/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp9assignvariableop_three_layer_conv_net_13_conv2d_26_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp9assignvariableop_1_three_layer_conv_net_13_conv2d_26_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp;assignvariableop_2_three_layer_conv_net_13_conv2d_27_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp9assignvariableop_3_three_layer_conv_net_13_conv2d_27_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp:assignvariableop_4_three_layer_conv_net_13_dense_13_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_three_layer_conv_net_13_dense_13_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1055192

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????$$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????$$
 
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
fc
flatten
softmax
regularization_losses
	variables
trainable_variables
		keras_api


signatures
C_default_save_signature
D__call__
*E&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ThreeLayerConvNet", "name": "three_layer_conv_net_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ThreeLayerConvNet"}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 36, 36, 3]}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 34, 34, 12]}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 8192]}}
?
regularization_losses
	variables
trainable_variables
 	keras_api
L__call__
*M&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
!regularization_losses
"	variables
#trainable_variables
$	keras_api
N__call__
*O&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax_13", "trainable": true, "dtype": "float32", "axis": -1}}
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
regularization_losses

%layers
	variables
&layer_metrics
'metrics
trainable_variables
(non_trainable_variables
)layer_regularization_losses
D__call__
C_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Pserving_default"
signature_map
B:@2(three_layer_conv_net_13/conv2d_26/kernel
4:22&three_layer_conv_net_13/conv2d_26/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

*layers
+layer_metrics
	variables
,metrics
trainable_variables
-non_trainable_variables
.layer_regularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
B:@2(three_layer_conv_net_13/conv2d_27/kernel
4:22&three_layer_conv_net_13/conv2d_27/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

/layers
0layer_metrics
	variables
1metrics
trainable_variables
2non_trainable_variables
3layer_regularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
::8	?@
2'three_layer_conv_net_13/dense_13/kernel
3:1
2%three_layer_conv_net_13/dense_13/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

4layers
5layer_metrics
	variables
6metrics
trainable_variables
7non_trainable_variables
8layer_regularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

9layers
:layer_metrics
	variables
;metrics
trainable_variables
<non_trainable_variables
=layer_regularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
!regularization_losses

>layers
?layer_metrics
"	variables
@metrics
#trainable_variables
Anon_trainable_variables
Blayer_regularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
C
0
1
2
3
4"
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
"__inference__wrapped_model_1054744?
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
9__inference_three_layer_conv_net_13_layer_call_fn_1055164
9__inference_three_layer_conv_net_13_layer_call_fn_1055181
9__inference_three_layer_conv_net_13_layer_call_fn_1055068
9__inference_three_layer_conv_net_13_layer_call_fn_1055085?
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
?2?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055051
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055116
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055147
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055020?
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
+__inference_conv2d_26_layer_call_fn_1055201?
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
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1055192?
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
+__inference_conv2d_27_layer_call_fn_1055221?
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
F__inference_conv2d_27_layer_call_and_return_conditional_losses_1055212?
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
*__inference_dense_13_layer_call_fn_1055240?
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
E__inference_dense_13_layer_call_and_return_conditional_losses_1055231?
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
,__inference_flatten_13_layer_call_fn_1055251?
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
G__inference_flatten_13_layer_call_and_return_conditional_losses_1055246?
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
,__inference_softmax_13_layer_call_fn_1055261?
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
G__inference_softmax_13_layer_call_and_return_conditional_losses_1055256?
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
?B?
%__inference_signature_wrapper_1054989input_1"?
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
"__inference__wrapped_model_1054744w8?5
.?+
)?&
input_1?????????  
? "3?0
.
output_1"?
output_1?????????
?
F__inference_conv2d_26_layer_call_and_return_conditional_losses_1055192l7?4
-?*
(?%
inputs?????????$$
? "-?*
#? 
0?????????  
? ?
+__inference_conv2d_26_layer_call_fn_1055201_7?4
-?*
(?%
inputs?????????$$
? " ??????????  ?
F__inference_conv2d_27_layer_call_and_return_conditional_losses_1055212l7?4
-?*
(?%
inputs?????????""
? "-?*
#? 
0?????????  
? ?
+__inference_conv2d_27_layer_call_fn_1055221_7?4
-?*
(?%
inputs?????????""
? " ??????????  ?
E__inference_dense_13_layer_call_and_return_conditional_losses_1055231]0?-
&?#
!?
inputs??????????@
? "%?"
?
0?????????

? ~
*__inference_dense_13_layer_call_fn_1055240P0?-
&?#
!?
inputs??????????@
? "??????????
?
G__inference_flatten_13_layer_call_and_return_conditional_losses_1055246a7?4
-?*
(?%
inputs?????????  
? "&?#
?
0??????????@
? ?
,__inference_flatten_13_layer_call_fn_1055251T7?4
-?*
(?%
inputs?????????  
? "???????????@?
%__inference_signature_wrapper_1054989?C?@
? 
9?6
4
input_1)?&
input_1?????????  "3?0
.
output_1"?
output_1?????????
?
G__inference_softmax_13_layer_call_and_return_conditional_losses_1055256\3?0
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
,__inference_softmax_13_layer_call_fn_1055261O3?0
)?&
 ?
inputs?????????


 
? "??????????
?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055020m<?9
2?/
)?&
input_1?????????  
p
? "%?"
?
0?????????

? ?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055051m<?9
2?/
)?&
input_1?????????  
p 
? "%?"
?
0?????????

? ?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055116g6?3
,?)
#? 
x?????????  
p
? "%?"
?
0?????????

? ?
T__inference_three_layer_conv_net_13_layer_call_and_return_conditional_losses_1055147g6?3
,?)
#? 
x?????????  
p 
? "%?"
?
0?????????

? ?
9__inference_three_layer_conv_net_13_layer_call_fn_1055068`<?9
2?/
)?&
input_1?????????  
p
? "??????????
?
9__inference_three_layer_conv_net_13_layer_call_fn_1055085`<?9
2?/
)?&
input_1?????????  
p 
? "??????????
?
9__inference_three_layer_conv_net_13_layer_call_fn_1055164Z6?3
,?)
#? 
x?????????  
p
? "??????????
?
9__inference_three_layer_conv_net_13_layer_call_fn_1055181Z6?3
,?)
#? 
x?????????  
p 
? "??????????
