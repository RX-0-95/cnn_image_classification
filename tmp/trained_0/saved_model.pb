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
lennet5_2/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelennet5_2/conv2d_26/kernel
?
.lennet5_2/conv2d_26/kernel/Read/ReadVariableOpReadVariableOplennet5_2/conv2d_26/kernel*&
_output_shapes
:*
dtype0
?
lennet5_2/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namelennet5_2/conv2d_26/bias
?
,lennet5_2/conv2d_26/bias/Read/ReadVariableOpReadVariableOplennet5_2/conv2d_26/bias*
_output_shapes
:*
dtype0
?
lennet5_2/conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelennet5_2/conv2d_27/kernel
?
.lennet5_2/conv2d_27/kernel/Read/ReadVariableOpReadVariableOplennet5_2/conv2d_27/kernel*&
_output_shapes
: *
dtype0
?
lennet5_2/conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namelennet5_2/conv2d_27/bias
?
,lennet5_2/conv2d_27/bias/Read/ReadVariableOpReadVariableOplennet5_2/conv2d_27/bias*
_output_shapes
: *
dtype0
?
lennet5_2/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?x*)
shared_namelennet5_2/dense_9/kernel
?
,lennet5_2/dense_9/kernel/Read/ReadVariableOpReadVariableOplennet5_2/dense_9/kernel*
_output_shapes
:	?x*
dtype0
?
lennet5_2/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*'
shared_namelennet5_2/dense_9/bias
}
*lennet5_2/dense_9/bias/Read/ReadVariableOpReadVariableOplennet5_2/dense_9/bias*
_output_shapes
:x*
dtype0
?
lennet5_2/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xP**
shared_namelennet5_2/dense_10/kernel
?
-lennet5_2/dense_10/kernel/Read/ReadVariableOpReadVariableOplennet5_2/dense_10/kernel*
_output_shapes

:xP*
dtype0
?
lennet5_2/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*(
shared_namelennet5_2/dense_10/bias

+lennet5_2/dense_10/bias/Read/ReadVariableOpReadVariableOplennet5_2/dense_10/bias*
_output_shapes
:P*
dtype0
?
lennet5_2/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P
**
shared_namelennet5_2/dense_11/kernel
?
-lennet5_2/dense_11/kernel/Read/ReadVariableOpReadVariableOplennet5_2/dense_11/kernel*
_output_shapes

:P
*
dtype0
?
lennet5_2/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namelennet5_2/dense_11/bias

+lennet5_2/dense_11/bias/Read/ReadVariableOpReadVariableOplennet5_2/dense_11/bias*
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
=non_trainable_variables
>layer_regularization_losses

regularization_losses
?layer_metrics
@metrics
trainable_variables
	variables

Alayers
 
WU
VARIABLE_VALUElennet5_2/conv2d_26/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElennet5_2/conv2d_26/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Bnon_trainable_variables
Clayer_regularization_losses
regularization_losses
Dlayer_metrics
Emetrics
trainable_variables
	variables

Flayers
WU
VARIABLE_VALUElennet5_2/conv2d_27/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElennet5_2/conv2d_27/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Gnon_trainable_variables
Hlayer_regularization_losses
regularization_losses
Ilayer_metrics
Jmetrics
trainable_variables
	variables

Klayers
 
 
 
?
Lnon_trainable_variables
Mlayer_regularization_losses
regularization_losses
Nlayer_metrics
Ometrics
trainable_variables
	variables

Players
 
 
 
?
Qnon_trainable_variables
Rlayer_regularization_losses
regularization_losses
Slayer_metrics
Tmetrics
 trainable_variables
!	variables

Ulayers
SQ
VARIABLE_VALUElennet5_2/dense_9/kernel%fc0/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElennet5_2/dense_9/bias#fc0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
?
Vnon_trainable_variables
Wlayer_regularization_losses
%regularization_losses
Xlayer_metrics
Ymetrics
&trainable_variables
'	variables

Zlayers
TR
VARIABLE_VALUElennet5_2/dense_10/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUElennet5_2/dense_10/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
[non_trainable_variables
\layer_regularization_losses
+regularization_losses
]layer_metrics
^metrics
,trainable_variables
-	variables

_layers
WU
VARIABLE_VALUElennet5_2/dense_11/kernel(fc_out/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElennet5_2/dense_11/bias&fc_out/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
?
`non_trainable_variables
alayer_regularization_losses
1regularization_losses
blayer_metrics
cmetrics
2trainable_variables
3	variables

dlayers
 
 
 
?
enon_trainable_variables
flayer_regularization_losses
5regularization_losses
glayer_metrics
hmetrics
6trainable_variables
7	variables

ilayers
 
 
 
?
jnon_trainable_variables
klayer_regularization_losses
9regularization_losses
llayer_metrics
mmetrics
:trainable_variables
;	variables

nlayers
 
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
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lennet5_2/conv2d_26/kernellennet5_2/conv2d_26/biaslennet5_2/conv2d_27/kernellennet5_2/conv2d_27/biaslennet5_2/dense_9/kernellennet5_2/dense_9/biaslennet5_2/dense_10/kernellennet5_2/dense_10/biaslennet5_2/dense_11/kernellennet5_2/dense_11/bias*
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
GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_363864
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.lennet5_2/conv2d_26/kernel/Read/ReadVariableOp,lennet5_2/conv2d_26/bias/Read/ReadVariableOp.lennet5_2/conv2d_27/kernel/Read/ReadVariableOp,lennet5_2/conv2d_27/bias/Read/ReadVariableOp,lennet5_2/dense_9/kernel/Read/ReadVariableOp*lennet5_2/dense_9/bias/Read/ReadVariableOp-lennet5_2/dense_10/kernel/Read/ReadVariableOp+lennet5_2/dense_10/bias/Read/ReadVariableOp-lennet5_2/dense_11/kernel/Read/ReadVariableOp+lennet5_2/dense_11/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_364449
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelennet5_2/conv2d_26/kernellennet5_2/conv2d_26/biaslennet5_2/conv2d_27/kernellennet5_2/conv2d_27/biaslennet5_2/dense_9/kernellennet5_2/dense_9/biaslennet5_2/dense_10/kernellennet5_2/dense_10/biaslennet5_2/dense_11/kernellennet5_2/dense_11/bias*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_364489??
?
L
0__inference_max_pooling2d_4_layer_call_fn_363391

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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_3633852
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
?
?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_364215

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
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
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_26/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2/
-lennet5_2/conv2d_26/kernel/Regularizer/Square?
,lennet5_2/conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_26/kernel/Regularizer/Const?
*lennet5_2/conv2d_26/kernel/Regularizer/SumSum1lennet5_2/conv2d_26/kernel/Regularizer/Square:y:05lennet5_2/conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/Sum?
,lennet5_2/conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_26/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_26/kernel/Regularizer/mulMul5lennet5_2/conv2d_26/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
`
D__inference_re_lu_23_layer_call_and_return_conditional_losses_363480

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
?
?
*__inference_lennet5_2_layer_call_fn_364192
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
GPU2*0J 8? *N
fIRG
E__inference_lennet5_2_layer_call_and_return_conditional_losses_3638022
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
?
?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_364247

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
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
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02>
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_27/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2/
-lennet5_2/conv2d_27/kernel/Regularizer/Square?
,lennet5_2/conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_27/kernel/Regularizer/Const?
*lennet5_2/conv2d_27/kernel/Regularizer/SumSum1lennet5_2/conv2d_27/kernel/Regularizer/Square:y:05lennet5_2/conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/Sum?
,lennet5_2/conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_27/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_27/kernel/Regularizer/mulMul5lennet5_2/conv2d_27/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
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
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_softmax_4_layer_call_fn_364334

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
GPU2*0J 8? *N
fIRG
E__inference_softmax_4_layer_call_and_return_conditional_losses_3636062
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
?	
?
C__inference_dense_9_layer_call_and_return_conditional_losses_364277

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
?
a
E__inference_softmax_4_layer_call_and_return_conditional_losses_363606

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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364349

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
`
D__inference_re_lu_23_layer_call_and_return_conditional_losses_363568

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
?
}
(__inference_dense_9_layer_call_fn_364286

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
GPU2*0J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3635112
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_363434

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
?
~
)__inference_dense_10_layer_call_fn_364305

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
D__inference_dense_10_layer_call_and_return_conditional_losses_3635482
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
E
)__inference_re_lu_23_layer_call_fn_364364

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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3635682
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
E__inference_lennet5_2_layer_call_and_return_conditional_losses_363802
x
conv2d_26_363756
conv2d_26_363758
conv2d_27_363763
conv2d_27_363765
dense_9_363771
dense_9_363773
dense_10_363777
dense_10_363779
dense_11_363783
dense_11_363785
identity??!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCallxconv2d_26_363756conv2d_26_363758*
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
E__inference_conv2d_26_layer_call_and_return_conditional_losses_3634122#
!conv2d_26/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_3633852!
max_pooling2d_4/PartitionedCall?
re_lu_23/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3634342
re_lu_23/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall!re_lu_23/PartitionedCall:output:0conv2d_27_363763conv2d_27_363765*
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
E__inference_conv2d_27_layer_call_and_return_conditional_losses_3634592#
!conv2d_27/StatefulPartitionedCall?
!max_pooling2d_4/PartitionedCall_1PartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_3633852#
!max_pooling2d_4/PartitionedCall_1?
re_lu_23/PartitionedCall_1PartitionedCall*max_pooling2d_4/PartitionedCall_1:output:0*
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3634802
re_lu_23/PartitionedCall_1?
flatten_4/PartitionedCallPartitionedCall#re_lu_23/PartitionedCall_1:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_3634932
flatten_4/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_9_363771dense_9_363773*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3635112!
dense_9/StatefulPartitionedCall?
re_lu_23/PartitionedCall_2PartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3635312
re_lu_23/PartitionedCall_2?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#re_lu_23/PartitionedCall_2:output:0dense_10_363777dense_10_363779*
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
D__inference_dense_10_layer_call_and_return_conditional_losses_3635482"
 dense_10/StatefulPartitionedCall?
re_lu_23/PartitionedCall_3PartitionedCall)dense_10/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3635682
re_lu_23/PartitionedCall_3?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#re_lu_23/PartitionedCall_3:output:0dense_11_363783dense_11_363785*
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
D__inference_dense_11_layer_call_and_return_conditional_losses_3635852"
 dense_11/StatefulPartitionedCall?
softmax_4/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_softmax_4_layer_call_and_return_conditional_losses_3636062
softmax_4/PartitionedCall?
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_363756*&
_output_shapes
:*
dtype02>
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_26/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2/
-lennet5_2/conv2d_26/kernel/Regularizer/Square?
,lennet5_2/conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_26/kernel/Regularizer/Const?
*lennet5_2/conv2d_26/kernel/Regularizer/SumSum1lennet5_2/conv2d_26/kernel/Regularizer/Square:y:05lennet5_2/conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/Sum?
,lennet5_2/conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_26/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_26/kernel/Regularizer/mulMul5lennet5_2/conv2d_26/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/mul?
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_363763*&
_output_shapes
: *
dtype02>
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_27/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2/
-lennet5_2/conv2d_27/kernel/Regularizer/Square?
,lennet5_2/conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_27/kernel/Regularizer/Const?
*lennet5_2/conv2d_27/kernel/Regularizer/SumSum1lennet5_2/conv2d_27/kernel/Regularizer/Square:y:05lennet5_2/conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/Sum?
,lennet5_2/conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_27/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_27/kernel/Regularizer/mulMul5lennet5_2/conv2d_27/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/mul?
IdentityIdentity"softmax_4/PartitionedCall:output:0"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall=^lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp=^lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2|
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp2|
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?	
?
D__inference_dense_10_layer_call_and_return_conditional_losses_363548

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
?Q
?
E__inference_lennet5_2_layer_call_and_return_conditional_losses_364085
x,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dx'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_26/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_26/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
re_lu_23/ReluRelu max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
re_lu_23/Relu?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2Dre_lu_23/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2d_27/Relu?
max_pooling2d_4/MaxPool_1MaxPoolconv2d_27/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool_1?
re_lu_23/Relu_1Relu"max_pooling2d_4/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_23/Relu_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapere_lu_23/Relu_1:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulflatten_4/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_9/BiasAddv
re_lu_23/Relu_2Reludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
re_lu_23/Relu_2?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulre_lu_23/Relu_2:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_10/BiasAddw
re_lu_23/Relu_3Reludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
re_lu_23/Relu_3?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulre_lu_23/Relu_3:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/BiasAdd~
softmax_4/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_4/Softmax?
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_26/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2/
-lennet5_2/conv2d_26/kernel/Regularizer/Square?
,lennet5_2/conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_26/kernel/Regularizer/Const?
*lennet5_2/conv2d_26/kernel/Regularizer/SumSum1lennet5_2/conv2d_26/kernel/Regularizer/Square:y:05lennet5_2/conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/Sum?
,lennet5_2/conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_26/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_26/kernel/Regularizer/mulMul5lennet5_2/conv2d_26/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/mul?
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02>
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_27/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2/
-lennet5_2/conv2d_27/kernel/Regularizer/Square?
,lennet5_2/conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_27/kernel/Regularizer/Const?
*lennet5_2/conv2d_27/kernel/Regularizer/SumSum1lennet5_2/conv2d_27/kernel/Regularizer/Square:y:05lennet5_2/conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/Sum?
,lennet5_2/conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_27/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_27/kernel/Regularizer/mulMul5lennet5_2/conv2d_27/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/mul?
IdentityIdentitysoftmax_4/Softmax:softmax:0!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp=^lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp=^lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2|
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp2|
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
E
)__inference_re_lu_23_layer_call_fn_364344

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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3634802
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
?
?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_363459

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
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
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02>
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_27/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2/
-lennet5_2/conv2d_27/kernel/Regularizer/Square?
,lennet5_2/conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_27/kernel/Regularizer/Const?
*lennet5_2/conv2d_27/kernel/Regularizer/SumSum1lennet5_2/conv2d_27/kernel/Regularizer/Square:y:05lennet5_2/conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/Sum?
,lennet5_2/conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_27/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_27/kernel/Regularizer/mulMul5lennet5_2/conv2d_27/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
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
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_softmax_4_layer_call_and_return_conditional_losses_364329

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
?
?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_363412

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
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
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_26/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2/
-lennet5_2/conv2d_26/kernel/Regularizer/Square?
,lennet5_2/conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_26/kernel/Regularizer/Const?
*lennet5_2/conv2d_26/kernel/Regularizer/SumSum1lennet5_2/conv2d_26/kernel/Regularizer/Square:y:05lennet5_2/conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/Sum?
,lennet5_2/conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_26/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_26/kernel/Regularizer/mulMul5lennet5_2/conv2d_26/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp=^lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2|
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

*__inference_conv2d_27_layer_call_fn_364256

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
E__inference_conv2d_27_layer_call_and_return_conditional_losses_3634592
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
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_363493

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
?D
?
!__inference__wrapped_model_363379
input_16
2lennet5_2_conv2d_26_conv2d_readvariableop_resource7
3lennet5_2_conv2d_26_biasadd_readvariableop_resource6
2lennet5_2_conv2d_27_conv2d_readvariableop_resource7
3lennet5_2_conv2d_27_biasadd_readvariableop_resource4
0lennet5_2_dense_9_matmul_readvariableop_resource5
1lennet5_2_dense_9_biasadd_readvariableop_resource5
1lennet5_2_dense_10_matmul_readvariableop_resource6
2lennet5_2_dense_10_biasadd_readvariableop_resource5
1lennet5_2_dense_11_matmul_readvariableop_resource6
2lennet5_2_dense_11_biasadd_readvariableop_resource
identity??*lennet5_2/conv2d_26/BiasAdd/ReadVariableOp?)lennet5_2/conv2d_26/Conv2D/ReadVariableOp?*lennet5_2/conv2d_27/BiasAdd/ReadVariableOp?)lennet5_2/conv2d_27/Conv2D/ReadVariableOp?)lennet5_2/dense_10/BiasAdd/ReadVariableOp?(lennet5_2/dense_10/MatMul/ReadVariableOp?)lennet5_2/dense_11/BiasAdd/ReadVariableOp?(lennet5_2/dense_11/MatMul/ReadVariableOp?(lennet5_2/dense_9/BiasAdd/ReadVariableOp?'lennet5_2/dense_9/MatMul/ReadVariableOp?
)lennet5_2/conv2d_26/Conv2D/ReadVariableOpReadVariableOp2lennet5_2_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)lennet5_2/conv2d_26/Conv2D/ReadVariableOp?
lennet5_2/conv2d_26/Conv2DConv2Dinput_11lennet5_2/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
lennet5_2/conv2d_26/Conv2D?
*lennet5_2/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp3lennet5_2_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*lennet5_2/conv2d_26/BiasAdd/ReadVariableOp?
lennet5_2/conv2d_26/BiasAddBiasAdd#lennet5_2/conv2d_26/Conv2D:output:02lennet5_2/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
lennet5_2/conv2d_26/BiasAdd?
lennet5_2/conv2d_26/ReluRelu$lennet5_2/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
lennet5_2/conv2d_26/Relu?
!lennet5_2/max_pooling2d_4/MaxPoolMaxPool&lennet5_2/conv2d_26/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2#
!lennet5_2/max_pooling2d_4/MaxPool?
lennet5_2/re_lu_23/ReluRelu*lennet5_2/max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
lennet5_2/re_lu_23/Relu?
)lennet5_2/conv2d_27/Conv2D/ReadVariableOpReadVariableOp2lennet5_2_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)lennet5_2/conv2d_27/Conv2D/ReadVariableOp?
lennet5_2/conv2d_27/Conv2DConv2D%lennet5_2/re_lu_23/Relu:activations:01lennet5_2/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
lennet5_2/conv2d_27/Conv2D?
*lennet5_2/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp3lennet5_2_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*lennet5_2/conv2d_27/BiasAdd/ReadVariableOp?
lennet5_2/conv2d_27/BiasAddBiasAdd#lennet5_2/conv2d_27/Conv2D:output:02lennet5_2/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
lennet5_2/conv2d_27/BiasAdd?
lennet5_2/conv2d_27/ReluRelu$lennet5_2/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
lennet5_2/conv2d_27/Relu?
#lennet5_2/max_pooling2d_4/MaxPool_1MaxPool&lennet5_2/conv2d_27/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2%
#lennet5_2/max_pooling2d_4/MaxPool_1?
lennet5_2/re_lu_23/Relu_1Relu,lennet5_2/max_pooling2d_4/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
lennet5_2/re_lu_23/Relu_1?
lennet5_2/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
lennet5_2/flatten_4/Const?
lennet5_2/flatten_4/ReshapeReshape'lennet5_2/re_lu_23/Relu_1:activations:0"lennet5_2/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lennet5_2/flatten_4/Reshape?
'lennet5_2/dense_9/MatMul/ReadVariableOpReadVariableOp0lennet5_2_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02)
'lennet5_2/dense_9/MatMul/ReadVariableOp?
lennet5_2/dense_9/MatMulMatMul$lennet5_2/flatten_4/Reshape:output:0/lennet5_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
lennet5_2/dense_9/MatMul?
(lennet5_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp1lennet5_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02*
(lennet5_2/dense_9/BiasAdd/ReadVariableOp?
lennet5_2/dense_9/BiasAddBiasAdd"lennet5_2/dense_9/MatMul:product:00lennet5_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
lennet5_2/dense_9/BiasAdd?
lennet5_2/re_lu_23/Relu_2Relu"lennet5_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
lennet5_2/re_lu_23/Relu_2?
(lennet5_2/dense_10/MatMul/ReadVariableOpReadVariableOp1lennet5_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02*
(lennet5_2/dense_10/MatMul/ReadVariableOp?
lennet5_2/dense_10/MatMulMatMul'lennet5_2/re_lu_23/Relu_2:activations:00lennet5_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
lennet5_2/dense_10/MatMul?
)lennet5_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp2lennet5_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02+
)lennet5_2/dense_10/BiasAdd/ReadVariableOp?
lennet5_2/dense_10/BiasAddBiasAdd#lennet5_2/dense_10/MatMul:product:01lennet5_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
lennet5_2/dense_10/BiasAdd?
lennet5_2/re_lu_23/Relu_3Relu#lennet5_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
lennet5_2/re_lu_23/Relu_3?
(lennet5_2/dense_11/MatMul/ReadVariableOpReadVariableOp1lennet5_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02*
(lennet5_2/dense_11/MatMul/ReadVariableOp?
lennet5_2/dense_11/MatMulMatMul'lennet5_2/re_lu_23/Relu_3:activations:00lennet5_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
lennet5_2/dense_11/MatMul?
)lennet5_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp2lennet5_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)lennet5_2/dense_11/BiasAdd/ReadVariableOp?
lennet5_2/dense_11/BiasAddBiasAdd#lennet5_2/dense_11/MatMul:product:01lennet5_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
lennet5_2/dense_11/BiasAdd?
lennet5_2/softmax_4/SoftmaxSoftmax#lennet5_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
lennet5_2/softmax_4/Softmax?
IdentityIdentity%lennet5_2/softmax_4/Softmax:softmax:0+^lennet5_2/conv2d_26/BiasAdd/ReadVariableOp*^lennet5_2/conv2d_26/Conv2D/ReadVariableOp+^lennet5_2/conv2d_27/BiasAdd/ReadVariableOp*^lennet5_2/conv2d_27/Conv2D/ReadVariableOp*^lennet5_2/dense_10/BiasAdd/ReadVariableOp)^lennet5_2/dense_10/MatMul/ReadVariableOp*^lennet5_2/dense_11/BiasAdd/ReadVariableOp)^lennet5_2/dense_11/MatMul/ReadVariableOp)^lennet5_2/dense_9/BiasAdd/ReadVariableOp(^lennet5_2/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2X
*lennet5_2/conv2d_26/BiasAdd/ReadVariableOp*lennet5_2/conv2d_26/BiasAdd/ReadVariableOp2V
)lennet5_2/conv2d_26/Conv2D/ReadVariableOp)lennet5_2/conv2d_26/Conv2D/ReadVariableOp2X
*lennet5_2/conv2d_27/BiasAdd/ReadVariableOp*lennet5_2/conv2d_27/BiasAdd/ReadVariableOp2V
)lennet5_2/conv2d_27/Conv2D/ReadVariableOp)lennet5_2/conv2d_27/Conv2D/ReadVariableOp2V
)lennet5_2/dense_10/BiasAdd/ReadVariableOp)lennet5_2/dense_10/BiasAdd/ReadVariableOp2T
(lennet5_2/dense_10/MatMul/ReadVariableOp(lennet5_2/dense_10/MatMul/ReadVariableOp2V
)lennet5_2/dense_11/BiasAdd/ReadVariableOp)lennet5_2/dense_11/BiasAdd/ReadVariableOp2T
(lennet5_2/dense_11/MatMul/ReadVariableOp(lennet5_2/dense_11/MatMul/ReadVariableOp2T
(lennet5_2/dense_9/BiasAdd/ReadVariableOp(lennet5_2/dense_9/BiasAdd/ReadVariableOp2R
'lennet5_2/dense_9/MatMul/ReadVariableOp'lennet5_2/dense_9/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?Q
?
E__inference_lennet5_2_layer_call_and_return_conditional_losses_364142
x,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dx'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_26/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_26/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
re_lu_23/ReluRelu max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
re_lu_23/Relu?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2Dre_lu_23/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2d_27/Relu?
max_pooling2d_4/MaxPool_1MaxPoolconv2d_27/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool_1?
re_lu_23/Relu_1Relu"max_pooling2d_4/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_23/Relu_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapere_lu_23/Relu_1:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulflatten_4/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_9/BiasAddv
re_lu_23/Relu_2Reludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
re_lu_23/Relu_2?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulre_lu_23/Relu_2:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_10/BiasAddw
re_lu_23/Relu_3Reludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
re_lu_23/Relu_3?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulre_lu_23/Relu_3:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/BiasAdd~
softmax_4/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_4/Softmax?
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_26/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2/
-lennet5_2/conv2d_26/kernel/Regularizer/Square?
,lennet5_2/conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_26/kernel/Regularizer/Const?
*lennet5_2/conv2d_26/kernel/Regularizer/SumSum1lennet5_2/conv2d_26/kernel/Regularizer/Square:y:05lennet5_2/conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/Sum?
,lennet5_2/conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_26/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_26/kernel/Regularizer/mulMul5lennet5_2/conv2d_26/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/mul?
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02>
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_27/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2/
-lennet5_2/conv2d_27/kernel/Regularizer/Square?
,lennet5_2/conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_27/kernel/Regularizer/Const?
*lennet5_2/conv2d_27/kernel/Regularizer/SumSum1lennet5_2/conv2d_27/kernel/Regularizer/Square:y:05lennet5_2/conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/Sum?
,lennet5_2/conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_27/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_27/kernel/Regularizer/mulMul5lennet5_2/conv2d_27/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/mul?
IdentityIdentitysoftmax_4/Softmax:softmax:0!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp=^lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp=^lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2|
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp2|
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
`
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364369

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
?	
?
D__inference_dense_10_layer_call_and_return_conditional_losses_364296

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
D__inference_dense_11_layer_call_and_return_conditional_losses_364315

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
?
`
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364359

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
?
g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_363385

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
?
`
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364339

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
?
E
)__inference_re_lu_23_layer_call_fn_364354

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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3634342
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
?

*__inference_conv2d_26_layer_call_fn_364224

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
E__inference_conv2d_26_layer_call_and_return_conditional_losses_3634122
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
?
?
__inference_loss_fn_0_364385I
Elennet5_2_conv2d_26_kernel_regularizer_square_readvariableop_resource
identity??<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpElennet5_2_conv2d_26_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02>
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_26/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2/
-lennet5_2/conv2d_26/kernel/Regularizer/Square?
,lennet5_2/conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_26/kernel/Regularizer/Const?
*lennet5_2/conv2d_26/kernel/Regularizer/SumSum1lennet5_2/conv2d_26/kernel/Regularizer/Square:y:05lennet5_2/conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/Sum?
,lennet5_2/conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_26/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_26/kernel/Regularizer/mulMul5lennet5_2/conv2d_26/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/mul?
IdentityIdentity.lennet5_2/conv2d_26/kernel/Regularizer/mul:z:0=^lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp
?
?
*__inference_lennet5_2_layer_call_fn_364028
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
GPU2*0J 8? *N
fIRG
E__inference_lennet5_2_layer_call_and_return_conditional_losses_3638022
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
?
~
)__inference_dense_11_layer_call_fn_364324

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
D__inference_dense_11_layer_call_and_return_conditional_losses_3635852
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
?
?
__inference_loss_fn_1_364396I
Elennet5_2_conv2d_27_kernel_regularizer_square_readvariableop_resource
identity??<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpElennet5_2_conv2d_27_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype02>
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_27/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2/
-lennet5_2/conv2d_27/kernel/Regularizer/Square?
,lennet5_2/conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_27/kernel/Regularizer/Const?
*lennet5_2/conv2d_27/kernel/Regularizer/SumSum1lennet5_2/conv2d_27/kernel/Regularizer/Square:y:05lennet5_2/conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/Sum?
,lennet5_2/conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_27/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_27/kernel/Regularizer/mulMul5lennet5_2/conv2d_27/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/mul?
IdentityIdentity.lennet5_2/conv2d_27/kernel/Regularizer/mul:z:0=^lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp
?Q
?
E__inference_lennet5_2_layer_call_and_return_conditional_losses_363978
input_1,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dinput_1'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_26/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_26/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
re_lu_23/ReluRelu max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
re_lu_23/Relu?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2Dre_lu_23/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2d_27/Relu?
max_pooling2d_4/MaxPool_1MaxPoolconv2d_27/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool_1?
re_lu_23/Relu_1Relu"max_pooling2d_4/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_23/Relu_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapere_lu_23/Relu_1:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulflatten_4/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_9/BiasAddv
re_lu_23/Relu_2Reludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
re_lu_23/Relu_2?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulre_lu_23/Relu_2:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_10/BiasAddw
re_lu_23/Relu_3Reludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
re_lu_23/Relu_3?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulre_lu_23/Relu_3:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/BiasAdd~
softmax_4/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_4/Softmax?
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_26/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2/
-lennet5_2/conv2d_26/kernel/Regularizer/Square?
,lennet5_2/conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_26/kernel/Regularizer/Const?
*lennet5_2/conv2d_26/kernel/Regularizer/SumSum1lennet5_2/conv2d_26/kernel/Regularizer/Square:y:05lennet5_2/conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/Sum?
,lennet5_2/conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_26/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_26/kernel/Regularizer/mulMul5lennet5_2/conv2d_26/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/mul?
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02>
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_27/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2/
-lennet5_2/conv2d_27/kernel/Regularizer/Square?
,lennet5_2/conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_27/kernel/Regularizer/Const?
*lennet5_2/conv2d_27/kernel/Regularizer/SumSum1lennet5_2/conv2d_27/kernel/Regularizer/Square:y:05lennet5_2/conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/Sum?
,lennet5_2/conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_27/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_27/kernel/Regularizer/mulMul5lennet5_2/conv2d_27/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/mul?
IdentityIdentitysoftmax_4/Softmax:softmax:0!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp=^lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp=^lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2|
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp2|
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
?
*__inference_lennet5_2_layer_call_fn_364003
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
GPU2*0J 8? *N
fIRG
E__inference_lennet5_2_layer_call_and_return_conditional_losses_3637282
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
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_364262

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
$__inference_signature_wrapper_363864
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
GPU2*0J 8? **
f%R#
!__inference__wrapped_model_3633792
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
?-
?
"__inference__traced_restore_364489
file_prefix/
+assignvariableop_lennet5_2_conv2d_26_kernel/
+assignvariableop_1_lennet5_2_conv2d_26_bias1
-assignvariableop_2_lennet5_2_conv2d_27_kernel/
+assignvariableop_3_lennet5_2_conv2d_27_bias/
+assignvariableop_4_lennet5_2_dense_9_kernel-
)assignvariableop_5_lennet5_2_dense_9_bias0
,assignvariableop_6_lennet5_2_dense_10_kernel.
*assignvariableop_7_lennet5_2_dense_10_bias0
,assignvariableop_8_lennet5_2_dense_11_kernel.
*assignvariableop_9_lennet5_2_dense_11_bias
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
AssignVariableOpAssignVariableOp+assignvariableop_lennet5_2_conv2d_26_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_lennet5_2_conv2d_26_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_lennet5_2_conv2d_27_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_lennet5_2_conv2d_27_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp+assignvariableop_4_lennet5_2_dense_9_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp)assignvariableop_5_lennet5_2_dense_9_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_lennet5_2_dense_10_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_lennet5_2_dense_10_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_lennet5_2_dense_11_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_lennet5_2_dense_11_biasIdentity_9:output:0"/device:CPU:0*
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
?
E
)__inference_re_lu_23_layer_call_fn_364374

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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3635312
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
__inference__traced_save_364449
file_prefix9
5savev2_lennet5_2_conv2d_26_kernel_read_readvariableop7
3savev2_lennet5_2_conv2d_26_bias_read_readvariableop9
5savev2_lennet5_2_conv2d_27_kernel_read_readvariableop7
3savev2_lennet5_2_conv2d_27_bias_read_readvariableop7
3savev2_lennet5_2_dense_9_kernel_read_readvariableop5
1savev2_lennet5_2_dense_9_bias_read_readvariableop8
4savev2_lennet5_2_dense_10_kernel_read_readvariableop6
2savev2_lennet5_2_dense_10_bias_read_readvariableop8
4savev2_lennet5_2_dense_11_kernel_read_readvariableop6
2savev2_lennet5_2_dense_11_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_lennet5_2_conv2d_26_kernel_read_readvariableop3savev2_lennet5_2_conv2d_26_bias_read_readvariableop5savev2_lennet5_2_conv2d_27_kernel_read_readvariableop3savev2_lennet5_2_conv2d_27_bias_read_readvariableop3savev2_lennet5_2_dense_9_kernel_read_readvariableop1savev2_lennet5_2_dense_9_bias_read_readvariableop4savev2_lennet5_2_dense_10_kernel_read_readvariableop2savev2_lennet5_2_dense_10_bias_read_readvariableop4savev2_lennet5_2_dense_11_kernel_read_readvariableop2savev2_lennet5_2_dense_11_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?K
?
E__inference_lennet5_2_layer_call_and_return_conditional_losses_363728
x
conv2d_26_363682
conv2d_26_363684
conv2d_27_363689
conv2d_27_363691
dense_9_363697
dense_9_363699
dense_10_363703
dense_10_363705
dense_11_363709
dense_11_363711
identity??!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCallxconv2d_26_363682conv2d_26_363684*
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
E__inference_conv2d_26_layer_call_and_return_conditional_losses_3634122#
!conv2d_26/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_3633852!
max_pooling2d_4/PartitionedCall?
re_lu_23/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3634342
re_lu_23/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall!re_lu_23/PartitionedCall:output:0conv2d_27_363689conv2d_27_363691*
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
E__inference_conv2d_27_layer_call_and_return_conditional_losses_3634592#
!conv2d_27/StatefulPartitionedCall?
!max_pooling2d_4/PartitionedCall_1PartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_3633852#
!max_pooling2d_4/PartitionedCall_1?
re_lu_23/PartitionedCall_1PartitionedCall*max_pooling2d_4/PartitionedCall_1:output:0*
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3634802
re_lu_23/PartitionedCall_1?
flatten_4/PartitionedCallPartitionedCall#re_lu_23/PartitionedCall_1:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_3634932
flatten_4/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_9_363697dense_9_363699*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3635112!
dense_9/StatefulPartitionedCall?
re_lu_23/PartitionedCall_2PartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3635312
re_lu_23/PartitionedCall_2?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#re_lu_23/PartitionedCall_2:output:0dense_10_363703dense_10_363705*
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
D__inference_dense_10_layer_call_and_return_conditional_losses_3635482"
 dense_10/StatefulPartitionedCall?
re_lu_23/PartitionedCall_3PartitionedCall)dense_10/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_3635682
re_lu_23/PartitionedCall_3?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#re_lu_23/PartitionedCall_3:output:0dense_11_363709dense_11_363711*
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
D__inference_dense_11_layer_call_and_return_conditional_losses_3635852"
 dense_11/StatefulPartitionedCall?
softmax_4/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_softmax_4_layer_call_and_return_conditional_losses_3636062
softmax_4/PartitionedCall?
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_363682*&
_output_shapes
:*
dtype02>
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_26/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2/
-lennet5_2/conv2d_26/kernel/Regularizer/Square?
,lennet5_2/conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_26/kernel/Regularizer/Const?
*lennet5_2/conv2d_26/kernel/Regularizer/SumSum1lennet5_2/conv2d_26/kernel/Regularizer/Square:y:05lennet5_2/conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/Sum?
,lennet5_2/conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_26/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_26/kernel/Regularizer/mulMul5lennet5_2/conv2d_26/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/mul?
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_363689*&
_output_shapes
: *
dtype02>
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_27/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2/
-lennet5_2/conv2d_27/kernel/Regularizer/Square?
,lennet5_2/conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_27/kernel/Regularizer/Const?
*lennet5_2/conv2d_27/kernel/Regularizer/SumSum1lennet5_2/conv2d_27/kernel/Regularizer/Square:y:05lennet5_2/conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/Sum?
,lennet5_2/conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_27/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_27/kernel/Regularizer/mulMul5lennet5_2/conv2d_27/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/mul?
IdentityIdentity"softmax_4/PartitionedCall:output:0"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall=^lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp=^lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2|
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp2|
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:R N
/
_output_shapes
:?????????  

_user_specified_namex
?
?
*__inference_lennet5_2_layer_call_fn_364167
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
GPU2*0J 8? *N
fIRG
E__inference_lennet5_2_layer_call_and_return_conditional_losses_3637282
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
?
F
*__inference_flatten_4_layer_call_fn_364267

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
GPU2*0J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_3634932
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
?	
?
C__inference_dense_9_layer_call_and_return_conditional_losses_363511

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
D__inference_dense_11_layer_call_and_return_conditional_losses_363585

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
?
`
D__inference_re_lu_23_layer_call_and_return_conditional_losses_363531

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
?Q
?
E__inference_lennet5_2_layer_call_and_return_conditional_losses_363921
input_1,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dinput_1'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_26/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_26/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
re_lu_23/ReluRelu max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
re_lu_23/Relu?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2Dre_lu_23/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2d_27/Relu?
max_pooling2d_4/MaxPool_1MaxPoolconv2d_27/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool_1?
re_lu_23/Relu_1Relu"max_pooling2d_4/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
re_lu_23/Relu_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapere_lu_23/Relu_1:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_4/Reshape?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?x*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMulflatten_4/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x2
dense_9/BiasAddv
re_lu_23/Relu_2Reludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x2
re_lu_23/Relu_2?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:xP*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulre_lu_23/Relu_2:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense_10/BiasAddw
re_lu_23/Relu_3Reludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P2
re_lu_23/Relu_3?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:P
*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMulre_lu_23/Relu_3:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/BiasAdd~
softmax_4/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
softmax_4/Softmax?
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_26/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2/
-lennet5_2/conv2d_26/kernel/Regularizer/Square?
,lennet5_2/conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_26/kernel/Regularizer/Const?
*lennet5_2/conv2d_26/kernel/Regularizer/SumSum1lennet5_2/conv2d_26/kernel/Regularizer/Square:y:05lennet5_2/conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/Sum?
,lennet5_2/conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_26/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_26/kernel/Regularizer/mulMul5lennet5_2/conv2d_26/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_26/kernel/Regularizer/mul?
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02>
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp?
-lennet5_2/conv2d_27/kernel/Regularizer/SquareSquareDlennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2/
-lennet5_2/conv2d_27/kernel/Regularizer/Square?
,lennet5_2/conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,lennet5_2/conv2d_27/kernel/Regularizer/Const?
*lennet5_2/conv2d_27/kernel/Regularizer/SumSum1lennet5_2/conv2d_27/kernel/Regularizer/Square:y:05lennet5_2/conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/Sum?
,lennet5_2/conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,lennet5_2/conv2d_27/kernel/Regularizer/mul/x?
*lennet5_2/conv2d_27/kernel/Regularizer/mulMul5lennet5_2/conv2d_27/kernel/Regularizer/mul/x:output:03lennet5_2/conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*lennet5_2/conv2d_27/kernel/Regularizer/mul?
IdentityIdentitysoftmax_4/Softmax:softmax:0!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp=^lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp=^lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????  ::::::::::2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2|
<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_26/kernel/Regularizer/Square/ReadVariableOp2|
<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp<lennet5_2/conv2d_27/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1"?L
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
o_default_save_signature
p__call__
*q&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "Lennet5", "name": "lennet5_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Lennet5"}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 18, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 32, 32, 3]}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 18}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 14, 14, 18]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
regularization_losses
 trainable_variables
!	variables
"	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
z__call__
*{&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 120, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 800}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 800]}}
?

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
|__call__
*}&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 120]}}
?

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
~__call__
*&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.002, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 80]}}
?
5regularization_losses
6trainable_variables
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax_4", "trainable": true, "dtype": "float32", "axis": -1}}
?
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_23", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
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
=non_trainable_variables
>layer_regularization_losses

regularization_losses
?layer_metrics
@metrics
trainable_variables
	variables

Alayers
p__call__
o_default_save_signature
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
4:22lennet5_2/conv2d_26/kernel
&:$2lennet5_2/conv2d_26/bias
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
Bnon_trainable_variables
Clayer_regularization_losses
regularization_losses
Dlayer_metrics
Emetrics
trainable_variables
	variables

Flayers
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
4:2 2lennet5_2/conv2d_27/kernel
&:$ 2lennet5_2/conv2d_27/bias
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
Gnon_trainable_variables
Hlayer_regularization_losses
regularization_losses
Ilayer_metrics
Jmetrics
trainable_variables
	variables

Klayers
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
Lnon_trainable_variables
Mlayer_regularization_losses
regularization_losses
Nlayer_metrics
Ometrics
trainable_variables
	variables

Players
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
Qnon_trainable_variables
Rlayer_regularization_losses
regularization_losses
Slayer_metrics
Tmetrics
 trainable_variables
!	variables

Ulayers
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
+:)	?x2lennet5_2/dense_9/kernel
$:"x2lennet5_2/dense_9/bias
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
Vnon_trainable_variables
Wlayer_regularization_losses
%regularization_losses
Xlayer_metrics
Ymetrics
&trainable_variables
'	variables

Zlayers
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
+:)xP2lennet5_2/dense_10/kernel
%:#P2lennet5_2/dense_10/bias
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
[non_trainable_variables
\layer_regularization_losses
+regularization_losses
]layer_metrics
^metrics
,trainable_variables
-	variables

_layers
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
+:)P
2lennet5_2/dense_11/kernel
%:#
2lennet5_2/dense_11/bias
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
`non_trainable_variables
alayer_regularization_losses
1regularization_losses
blayer_metrics
cmetrics
2trainable_variables
3	variables

dlayers
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
enon_trainable_variables
flayer_regularization_losses
5regularization_losses
glayer_metrics
hmetrics
6trainable_variables
7	variables

ilayers
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
jnon_trainable_variables
klayer_regularization_losses
9regularization_losses
llayer_metrics
mmetrics
:trainable_variables
;	variables

nlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
!__inference__wrapped_model_363379?
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
?2?
*__inference_lennet5_2_layer_call_fn_364003
*__inference_lennet5_2_layer_call_fn_364192
*__inference_lennet5_2_layer_call_fn_364028
*__inference_lennet5_2_layer_call_fn_364167?
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
?2?
E__inference_lennet5_2_layer_call_and_return_conditional_losses_363978
E__inference_lennet5_2_layer_call_and_return_conditional_losses_364142
E__inference_lennet5_2_layer_call_and_return_conditional_losses_364085
E__inference_lennet5_2_layer_call_and_return_conditional_losses_363921?
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
*__inference_conv2d_26_layer_call_fn_364224?
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
E__inference_conv2d_26_layer_call_and_return_conditional_losses_364215?
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
*__inference_conv2d_27_layer_call_fn_364256?
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
E__inference_conv2d_27_layer_call_and_return_conditional_losses_364247?
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
0__inference_max_pooling2d_4_layer_call_fn_363391?
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
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_363385?
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
*__inference_flatten_4_layer_call_fn_364267?
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
E__inference_flatten_4_layer_call_and_return_conditional_losses_364262?
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
(__inference_dense_9_layer_call_fn_364286?
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
C__inference_dense_9_layer_call_and_return_conditional_losses_364277?
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
)__inference_dense_10_layer_call_fn_364305?
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
D__inference_dense_10_layer_call_and_return_conditional_losses_364296?
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
)__inference_dense_11_layer_call_fn_364324?
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
D__inference_dense_11_layer_call_and_return_conditional_losses_364315?
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
*__inference_softmax_4_layer_call_fn_364334?
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
E__inference_softmax_4_layer_call_and_return_conditional_losses_364329?
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
)__inference_re_lu_23_layer_call_fn_364364
)__inference_re_lu_23_layer_call_fn_364344
)__inference_re_lu_23_layer_call_fn_364354
)__inference_re_lu_23_layer_call_fn_364374?
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
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364369
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364359
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364339
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364349?
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
__inference_loss_fn_0_364385?
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
__inference_loss_fn_1_364396?
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
$__inference_signature_wrapper_363864input_1"?
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
!__inference__wrapped_model_363379{
#$)*/08?5
.?+
)?&
input_1?????????  
? "3?0
.
output_1"?
output_1?????????
?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_364215l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????
? ?
*__inference_conv2d_26_layer_call_fn_364224_7?4
-?*
(?%
inputs?????????  
? " ???????????
E__inference_conv2d_27_layer_call_and_return_conditional_losses_364247l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????

 
? ?
*__inference_conv2d_27_layer_call_fn_364256_7?4
-?*
(?%
inputs?????????
? " ??????????

 ?
D__inference_dense_10_layer_call_and_return_conditional_losses_364296\)*/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????P
? |
)__inference_dense_10_layer_call_fn_364305O)*/?,
%?"
 ?
inputs?????????x
? "??????????P?
D__inference_dense_11_layer_call_and_return_conditional_losses_364315\/0/?,
%?"
 ?
inputs?????????P
? "%?"
?
0?????????

? |
)__inference_dense_11_layer_call_fn_364324O/0/?,
%?"
 ?
inputs?????????P
? "??????????
?
C__inference_dense_9_layer_call_and_return_conditional_losses_364277]#$0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????x
? |
(__inference_dense_9_layer_call_fn_364286P#$0?-
&?#
!?
inputs??????????
? "??????????x?
E__inference_flatten_4_layer_call_and_return_conditional_losses_364262a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
*__inference_flatten_4_layer_call_fn_364267T7?4
-?*
(?%
inputs????????? 
? "????????????
E__inference_lennet5_2_layer_call_and_return_conditional_losses_363921q
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
E__inference_lennet5_2_layer_call_and_return_conditional_losses_363978q
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
E__inference_lennet5_2_layer_call_and_return_conditional_losses_364085k
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
E__inference_lennet5_2_layer_call_and_return_conditional_losses_364142k
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
*__inference_lennet5_2_layer_call_fn_364003d
#$)*/0<?9
2?/
)?&
input_1?????????  
p
? "??????????
?
*__inference_lennet5_2_layer_call_fn_364028d
#$)*/0<?9
2?/
)?&
input_1?????????  
p 
? "??????????
?
*__inference_lennet5_2_layer_call_fn_364167^
#$)*/06?3
,?)
#? 
x?????????  
p
? "??????????
?
*__inference_lennet5_2_layer_call_fn_364192^
#$)*/06?3
,?)
#? 
x?????????  
p 
? "??????????
;
__inference_loss_fn_0_364385?

? 
? "? ;
__inference_loss_fn_1_364396?

? 
? "? ?
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_363385?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_4_layer_call_fn_363391?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364339h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364349h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364359X/?,
%?"
 ?
inputs?????????P
? "%?"
?
0?????????P
? ?
D__inference_re_lu_23_layer_call_and_return_conditional_losses_364369X/?,
%?"
 ?
inputs?????????x
? "%?"
?
0?????????x
? ?
)__inference_re_lu_23_layer_call_fn_364344[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
)__inference_re_lu_23_layer_call_fn_364354[7?4
-?*
(?%
inputs?????????
? " ??????????x
)__inference_re_lu_23_layer_call_fn_364364K/?,
%?"
 ?
inputs?????????P
? "??????????Px
)__inference_re_lu_23_layer_call_fn_364374K/?,
%?"
 ?
inputs?????????x
? "??????????x?
$__inference_signature_wrapper_363864?
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
E__inference_softmax_4_layer_call_and_return_conditional_losses_364329\3?0
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
? }
*__inference_softmax_4_layer_call_fn_364334O3?0
)?&
 ?
inputs?????????


 
? "??????????
