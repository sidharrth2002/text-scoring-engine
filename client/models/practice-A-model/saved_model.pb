??3
?%?%
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
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
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.6.02v2.6.0-0-g919f693420e8??1
?
embedding_9/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameembedding_9/embeddings
?
*embedding_9/embeddings/Read/ReadVariableOpReadVariableOpembedding_9/embeddings* 
_output_shapes
:
??*
dtype0

conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?2* 
shared_nameconv1d_5/kernel
x
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*#
_output_shapes
:?2*
dtype0
r
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameconv1d_5/bias
k
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes
:2*
dtype0
{
attention_3/att_vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameattention_3/att_v
t
%attention_3/att_v/Read/ReadVariableOpReadVariableOpattention_3/att_v*
_output_shapes	
:?*
dtype0
?
attention_3/att_WVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*"
shared_nameattention_3/att_W
y
%attention_3/att_W/Read/ReadVariableOpReadVariableOpattention_3/att_W* 
_output_shapes
:
??*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	?@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P *
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:P *
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
: *
dtype0
t
score/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namescore/kernel
m
 score/kernel/Read/ReadVariableOpReadVariableOpscore/kernel*
_output_shapes

: *
dtype0
l

score/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
score/bias
e
score/bias/Read/ReadVariableOpReadVariableOp
score/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_3/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?	**
shared_namelstm_3/lstm_cell_3/kernel
?
-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/kernel*
_output_shapes
:	2?	*
dtype0
?
#lstm_3/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??	*4
shared_name%#lstm_3/lstm_cell_3/recurrent_kernel
?
7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_3/recurrent_kernel* 
_output_shapes
:
??	*
dtype0
?
lstm_3/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?	*(
shared_namelstm_3/lstm_cell_3/bias
?
+lstm_3/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/bias*
_output_shapes	
:?	*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv1d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?2*'
shared_nameAdam/conv1d_5/kernel/m
?
*Adam/conv1d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/kernel/m*#
_output_shapes
:?2*
dtype0
?
Adam/conv1d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/conv1d_5/bias/m
y
(Adam/conv1d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/bias/m*
_output_shapes
:2*
dtype0
?
Adam/attention_3/att_v/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/attention_3/att_v/m
?
,Adam/attention_3/att_v/m/Read/ReadVariableOpReadVariableOpAdam/attention_3/att_v/m*
_output_shapes	
:?*
dtype0
?
Adam/attention_3/att_W/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameAdam/attention_3/att_W/m
?
,Adam/attention_3/att_W/m/Read/ReadVariableOpReadVariableOpAdam/attention_3/att_W/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/dense_5/kernel/m
?
)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	?@*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P *&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:P *
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
: *
dtype0
?
Adam/score/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/score/kernel/m
{
'Adam/score/kernel/m/Read/ReadVariableOpReadVariableOpAdam/score/kernel/m*
_output_shapes

: *
dtype0
z
Adam/score/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/score/bias/m
s
%Adam/score/bias/m/Read/ReadVariableOpReadVariableOpAdam/score/bias/m*
_output_shapes
:*
dtype0
?
 Adam/lstm_3/lstm_cell_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?	*1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/m
?
4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/m*
_output_shapes
:	2?	*
dtype0
?
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??	*;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
?
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m* 
_output_shapes
:
??	*
dtype0
?
Adam/lstm_3/lstm_cell_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?	*/
shared_name Adam/lstm_3/lstm_cell_3/bias/m
?
2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/m*
_output_shapes	
:?	*
dtype0
?
Adam/conv1d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?2*'
shared_nameAdam/conv1d_5/kernel/v
?
*Adam/conv1d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/kernel/v*#
_output_shapes
:?2*
dtype0
?
Adam/conv1d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/conv1d_5/bias/v
y
(Adam/conv1d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/bias/v*
_output_shapes
:2*
dtype0
?
Adam/attention_3/att_v/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/attention_3/att_v/v
?
,Adam/attention_3/att_v/v/Read/ReadVariableOpReadVariableOpAdam/attention_3/att_v/v*
_output_shapes	
:?*
dtype0
?
Adam/attention_3/att_W/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameAdam/attention_3/att_W/v
?
,Adam/attention_3/att_W/v/Read/ReadVariableOpReadVariableOpAdam/attention_3/att_W/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/dense_5/kernel/v
?
)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	?@*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P *&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:P *
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
: *
dtype0
?
Adam/score/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/score/kernel/v
{
'Adam/score/kernel/v/Read/ReadVariableOpReadVariableOpAdam/score/kernel/v*
_output_shapes

: *
dtype0
z
Adam/score/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/score/bias/v
s
%Adam/score/bias/v/Read/ReadVariableOpReadVariableOpAdam/score/bias/v*
_output_shapes
:*
dtype0
?
 Adam/lstm_3/lstm_cell_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?	*1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/v
?
4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/v*
_output_shapes
:	2?	*
dtype0
?
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??	*;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
?
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v* 
_output_shapes
:
??	*
dtype0
?
Adam/lstm_3/lstm_cell_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?	*/
shared_name Adam/lstm_3/lstm_cell_3/bias/v
?
2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/v*
_output_shapes	
:?	*
dtype0

NoOpNoOp
?\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?[
value?[B?[ B?[
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
	optimizer
loss
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
b

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
l
%cell
&
state_spec
'	variables
(regularization_losses
)trainable_variables
*	keras_api
R
+	variables
,regularization_losses
-trainable_variables
.	keras_api
h
	/att_v
	0att_W
1	variables
2regularization_losses
3trainable_variables
4	keras_api
 
h

5kernel
6bias
7	variables
8regularization_losses
9trainable_variables
:	keras_api
h

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
R
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
R
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
h

Ikernel
Jbias
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
h

Okernel
Pbias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
?
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem? m?/m?0m?5m?6m?;m?<m?Im?Jm?Om?Pm?Zm?[m?\m?v? v?/v?0v?5v?6v?;v?<v?Iv?Jv?Ov?Pv?Zv?[v?\v?
 
v
0
1
 2
Z3
[4
\5
/6
07
58
69
;10
<11
I12
J13
O14
P15
 
n
0
 1
Z2
[3
\4
/5
06
57
68
;9
<10
I11
J12
O13
P14
?
]non_trainable_variables
	variables
^layer_metrics
_metrics
`layer_regularization_losses
regularization_losses
trainable_variables

alayers
 
fd
VARIABLE_VALUEembedding_9/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 
 
?
bnon_trainable_variables
	variables
clayer_metrics
dmetrics
elayer_regularization_losses
regularization_losses
trainable_variables

flayers
 
 
 
?
gnon_trainable_variables
	variables
hlayer_metrics
imetrics
jlayer_regularization_losses
regularization_losses
trainable_variables

klayers
[Y
VARIABLE_VALUEconv1d_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
?
lnon_trainable_variables
!	variables
mlayer_metrics
nmetrics
olayer_regularization_losses
"regularization_losses
#trainable_variables

players
?
q
state_size

Zkernel
[recurrent_kernel
\bias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
 

Z0
[1
\2
 

Z0
[1
\2
?
vnon_trainable_variables
'	variables
wlayer_metrics
xmetrics
ylayer_regularization_losses
(regularization_losses
)trainable_variables

zstates

{layers
 
 
 
?
|non_trainable_variables
+	variables
}layer_metrics
~metrics
layer_regularization_losses
,regularization_losses
-trainable_variables
?layers
\Z
VARIABLE_VALUEattention_3/att_v5layer_with_weights-3/att_v/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEattention_3/att_W5layer_with_weights-3/att_W/.ATTRIBUTES/VARIABLE_VALUE

/0
01
 

/0
01
?
?non_trainable_variables
1	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
2regularization_losses
3trainable_variables
?layers
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61
 

50
61
?
?non_trainable_variables
7	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
8regularization_losses
9trainable_variables
?layers
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
?
?non_trainable_variables
=	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
>regularization_losses
?trainable_variables
?layers
 
 
 
?
?non_trainable_variables
A	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
Bregularization_losses
Ctrainable_variables
?layers
 
 
 
?
?non_trainable_variables
E	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
Fregularization_losses
Gtrainable_variables
?layers
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 

I0
J1
?
?non_trainable_variables
K	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
Lregularization_losses
Mtrainable_variables
?layers
XV
VARIABLE_VALUEscore/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
score/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1
 

O0
P1
?
?non_trainable_variables
Q	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
Rregularization_losses
Strainable_variables
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_3/lstm_cell_3/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_3/lstm_cell_3/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_3/lstm_cell_3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE

0
 

?0
?1
 
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
9
10
11
12
13

0
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

Z0
[1
\2
 

Z0
[1
\2
?
?non_trainable_variables
r	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
sregularization_losses
ttrainable_variables
?layers
 
 
 
 
 

%0
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/conv1d_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/attention_3/att_v/mQlayer_with_weights-3/att_v/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/attention_3/att_W/mQlayer_with_weights-3/att_W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/score/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/score/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/attention_3/att_v/vQlayer_with_weights-3/att_v/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/attention_3/att_W/vQlayer_with_weights-3/att_W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/score/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/score/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_appliedPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_responsePlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_appliedserving_default_responseembedding_9/embeddingsconv1d_5/kernelconv1d_5/biaslstm_3/lstm_cell_3/kernellstm_3/lstm_cell_3/bias#lstm_3/lstm_cell_3/recurrent_kernelattention_3/att_Wattention_3/att_vdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasscore/kernel
score/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_75966
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_9/embeddings/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOp%attention_3/att_v/Read/ReadVariableOp%attention_3/att_W/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp score/kernel/Read/ReadVariableOpscore/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv1d_5/kernel/m/Read/ReadVariableOp(Adam/conv1d_5/bias/m/Read/ReadVariableOp,Adam/attention_3/att_v/m/Read/ReadVariableOp,Adam/attention_3/att_W/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp'Adam/score/kernel/m/Read/ReadVariableOp%Adam/score/bias/m/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOp*Adam/conv1d_5/kernel/v/Read/ReadVariableOp(Adam/conv1d_5/bias/v/Read/ReadVariableOp,Adam/attention_3/att_v/v/Read/ReadVariableOp,Adam/attention_3/att_W/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp'Adam/score/kernel/v/Read/ReadVariableOp%Adam/score/bias/v/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_78943
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_9/embeddingsconv1d_5/kernelconv1d_5/biasattention_3/att_vattention_3/att_Wdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasscore/kernel
score/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biastotalcounttotal_1count_1Adam/conv1d_5/kernel/mAdam/conv1d_5/bias/mAdam/attention_3/att_v/mAdam/attention_3/att_W/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/score/kernel/mAdam/score/bias/m Adam/lstm_3/lstm_cell_3/kernel/m*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mAdam/lstm_3/lstm_cell_3/bias/mAdam/conv1d_5/kernel/vAdam/conv1d_5/bias/vAdam/attention_3/att_v/vAdam/attention_3/att_W/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/vAdam/score/kernel/vAdam/score/bias/v Adam/lstm_3/lstm_cell_3/kernel/v*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vAdam/lstm_3/lstm_cell_3/bias/v*C
Tin<
:28*
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_79118Ѥ/
?

k
J__inference_pos_x_maskedout_layer_call_and_return_conditional_losses_74652
x
mask

identity\
CastCastmask*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
Castb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2
stackq
TileTileExpandDims:output:0stack:output:0*
T0*-
_output_shapes
:???????????2
Tileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	TransposeTile:output:0transpose/perm:output:0*
T0*-
_output_shapes
:???????????2
	transpose[
mulMulxtranspose:y:0*
T0*-
_output_shapes
:???????????2
mula
IdentityIdentitymul:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????:??????????:P L
-
_output_shapes
:???????????

_user_specified_namex:NJ
(
_output_shapes
:??????????

_user_specified_namemask
?o
?
__inference__traced_save_78943
file_prefix5
1savev2_embedding_9_embeddings_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop0
,savev2_attention_3_att_v_read_readvariableop0
,savev2_attention_3_att_w_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop+
'savev2_score_kernel_read_readvariableop)
%savev2_score_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv1d_5_kernel_m_read_readvariableop3
/savev2_adam_conv1d_5_bias_m_read_readvariableop7
3savev2_adam_attention_3_att_v_m_read_readvariableop7
3savev2_adam_attention_3_att_w_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop2
.savev2_adam_score_kernel_m_read_readvariableop0
,savev2_adam_score_bias_m_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop5
1savev2_adam_conv1d_5_kernel_v_read_readvariableop3
/savev2_adam_conv1d_5_bias_v_read_readvariableop7
3savev2_adam_attention_3_att_v_v_read_readvariableop7
3savev2_adam_attention_3_att_w_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop2
.savev2_adam_score_kernel_v_read_readvariableop0
,savev2_adam_score_bias_v_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/att_v/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/att_W/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/att_v/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/att_W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/att_v/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/att_W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_9_embeddings_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop,savev2_attention_3_att_v_read_readvariableop,savev2_attention_3_att_w_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop'savev2_score_kernel_read_readvariableop%savev2_score_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv1d_5_kernel_m_read_readvariableop/savev2_adam_conv1d_5_bias_m_read_readvariableop3savev2_adam_attention_3_att_v_m_read_readvariableop3savev2_adam_attention_3_att_w_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop.savev2_adam_score_kernel_m_read_readvariableop,savev2_adam_score_bias_m_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop1savev2_adam_conv1d_5_kernel_v_read_readvariableop/savev2_adam_conv1d_5_bias_v_read_readvariableop3savev2_adam_attention_3_att_v_v_read_readvariableop3savev2_adam_attention_3_att_w_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop.savev2_adam_score_kernel_v_read_readvariableop,savev2_adam_score_bias_v_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
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

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?2:2:?:
??:	?@:@:::P : : :: : : : : :	2?	:
??	:?	: : : : :?2:2:?:
??:	?@:@:::P : : ::	2?	:
??	:?	:?2:2:?:
??:	?@:@:::P : : ::	2?	:
??	:?	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:)%
#
_output_shapes
:?2: 

_output_shapes
:2:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:: 	

_output_shapes
::$
 

_output_shapes

:P : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2?	:&"
 
_output_shapes
:
??	:!

_output_shapes	
:?	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :)%
#
_output_shapes
:?2: 

_output_shapes
:2:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?@: 

_output_shapes
:@:$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:P : #

_output_shapes
: :$$ 

_output_shapes

: : %

_output_shapes
::%&!

_output_shapes
:	2?	:&'"
 
_output_shapes
:
??	:!(

_output_shapes	
:?	:))%
#
_output_shapes
:?2: *

_output_shapes
:2:!+

_output_shapes	
:?:&,"
 
_output_shapes
:
??:%-!

_output_shapes
:	?@: .

_output_shapes
:@:$/ 

_output_shapes

:: 0

_output_shapes
::$1 

_output_shapes

:P : 2

_output_shapes
: :$3 

_output_shapes

: : 4

_output_shapes
::%5!

_output_shapes
:	2?	:&6"
 
_output_shapes
:
??	:!7

_output_shapes	
:?	:8

_output_shapes
: 
?
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_78443

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?;
?
D__inference_Taghipour_layer_call_and_return_conditional_losses_75869
response
applied%
embedding_9_75822:
??%
conv1d_5_75828:?2
conv1d_5_75830:2
lstm_3_75833:	2?	
lstm_3_75835:	?	 
lstm_3_75837:
??	%
attention_3_75841:
?? 
attention_3_75843:	? 
dense_5_75846:	?@
dense_5_75848:@
dense_6_75851:
dense_6_75853:
dense_7_75858:P 
dense_7_75860: 
score_75863: 
score_75865:
identity??#attention_3/StatefulPartitionedCall? conv1d_5/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?score/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallresponseembedding_9_75822*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_9_layer_call_and_return_conditional_losses_746332%
#embedding_9/StatefulPartitionedCallu
embedding_9/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_9/NotEqual/y?
embedding_9/NotEqualNotEqualresponseembedding_9/NotEqual/y:output:0*
T0*(
_output_shapes
:??????????2
embedding_9/NotEqual?
pos_x_maskedout/PartitionedCallPartitionedCall,embedding_9/StatefulPartitionedCall:output:0embedding_9/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_pos_x_maskedout_layer_call_and_return_conditional_losses_746522!
pos_x_maskedout/PartitionedCall?
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall(pos_x_maskedout/PartitionedCall:output:0conv1d_5_75828conv1d_5_75830*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_746692"
 conv1d_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0lstm_3_75833lstm_3_75835lstm_3_75837*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_749252 
lstm_3/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_749382
dropout_4/PartitionedCall?
#attention_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0attention_3_75841attention_3_75843*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_attention_3_layer_call_and_return_conditional_losses_750012%
#attention_3/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall,attention_3/StatefulPartitionedCall:output:0dense_5_75846dense_5_75848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_750172!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallapplieddense_6_75851dense_6_75853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_750332!
dense_6/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_750462
concatenate_1/PartitionedCall?
dropout_5/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_750532
dropout_5/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_7_75858dense_7_75860*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_750652!
dense_7/StatefulPartitionedCall?
score/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0score_75863score_75865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_score_layer_call_and_return_conditional_losses_750822
score/StatefulPartitionedCall?
IdentityIdentity&score/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp$^attention_3/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^score/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 2J
#attention_3/StatefulPartitionedCall#attention_3/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2>
score/StatefulPartitionedCallscore/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
response:PL
'
_output_shapes
:?????????
!
_user_specified_name	applied
?
r
H__inference_concatenate_1_layer_call_and_return_conditional_losses_75046

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????P2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????@:?????????:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_5_layer_call_and_return_conditional_losses_74669

inputsB
+conv1d_expanddims_1_readvariableop_resource:?2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?2*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?22
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????2*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????2*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????22	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????22

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
while_cond_75420
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_75420___redundant_placeholder03
/while_while_cond_75420___redundant_placeholder13
/while_while_cond_75420___redundant_placeholder23
/while_while_cond_75420___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_74790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_74790___redundant_placeholder03
/while_while_cond_74790___redundant_placeholder13
/while_while_cond_74790___redundant_placeholder23
/while_while_cond_74790___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?>
?
D__inference_Taghipour_layer_call_and_return_conditional_losses_75745

inputs
inputs_1%
embedding_9_75698:
??%
conv1d_5_75704:?2
conv1d_5_75706:2
lstm_3_75709:	2?	
lstm_3_75711:	?	 
lstm_3_75713:
??	%
attention_3_75717:
?? 
attention_3_75719:	? 
dense_5_75722:	?@
dense_5_75724:@
dense_6_75727:
dense_6_75729:
dense_7_75734:P 
dense_7_75736: 
score_75739: 
score_75741:
identity??#attention_3/StatefulPartitionedCall? conv1d_5/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?score/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_9_75698*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_9_layer_call_and_return_conditional_losses_746332%
#embedding_9/StatefulPartitionedCallu
embedding_9/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_9/NotEqual/y?
embedding_9/NotEqualNotEqualinputsembedding_9/NotEqual/y:output:0*
T0*(
_output_shapes
:??????????2
embedding_9/NotEqual?
pos_x_maskedout/PartitionedCallPartitionedCall,embedding_9/StatefulPartitionedCall:output:0embedding_9/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_pos_x_maskedout_layer_call_and_return_conditional_losses_746522!
pos_x_maskedout/PartitionedCall?
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall(pos_x_maskedout/PartitionedCall:output:0conv1d_5_75704conv1d_5_75706*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_746692"
 conv1d_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0lstm_3_75709lstm_3_75711lstm_3_75713*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_756192 
lstm_3/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_752242#
!dropout_4/StatefulPartitionedCall?
#attention_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0attention_3_75717attention_3_75719*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_attention_3_layer_call_and_return_conditional_losses_750012%
#attention_3/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall,attention_3/StatefulPartitionedCall:output:0dense_5_75722dense_5_75724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_750172!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_6_75727dense_6_75729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_750332!
dense_6/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_750462
concatenate_1/PartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_751642#
!dropout_5/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_7_75734dense_7_75736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_750652!
dense_7/StatefulPartitionedCall?
score/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0score_75739score_75741*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_score_layer_call_and_return_conditional_losses_750822
score/StatefulPartitionedCall?
IdentityIdentity&score/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp$^attention_3/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^score/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 2J
#attention_3/StatefulPartitionedCall#attention_3/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2>
score/StatefulPartitionedCallscore/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_Taghipour_layer_call_fn_76914
inputs_response#
inputs_whether_criteria_applied
unknown:
?? 
	unknown_0:?2
	unknown_1:2
	unknown_2:	2?	
	unknown_3:	?	
	unknown_4:
??	
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:

unknown_10:

unknown_11:P 

unknown_12: 

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_responseinputs_whether_criteria_appliedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Taghipour_layer_call_and_return_conditional_losses_757452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:??????????
)
_user_specified_nameinputs/response:hd
'
_output_shapes
:?????????
9
_user_specified_name!inputs/whether_criteria_applied
??
?"
!__inference__traced_restore_79118
file_prefix;
'assignvariableop_embedding_9_embeddings:
??9
"assignvariableop_1_conv1d_5_kernel:?2.
 assignvariableop_2_conv1d_5_bias:23
$assignvariableop_3_attention_3_att_v:	?8
$assignvariableop_4_attention_3_att_w:
??4
!assignvariableop_5_dense_5_kernel:	?@-
assignvariableop_6_dense_5_bias:@3
!assignvariableop_7_dense_6_kernel:-
assignvariableop_8_dense_6_bias:3
!assignvariableop_9_dense_7_kernel:P .
 assignvariableop_10_dense_7_bias: 2
 assignvariableop_11_score_kernel: ,
assignvariableop_12_score_bias:'
assignvariableop_13_adam_iter:	 )
assignvariableop_14_adam_beta_1: )
assignvariableop_15_adam_beta_2: (
assignvariableop_16_adam_decay: 0
&assignvariableop_17_adam_learning_rate: @
-assignvariableop_18_lstm_3_lstm_cell_3_kernel:	2?	K
7assignvariableop_19_lstm_3_lstm_cell_3_recurrent_kernel:
??	:
+assignvariableop_20_lstm_3_lstm_cell_3_bias:	?	#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: A
*assignvariableop_25_adam_conv1d_5_kernel_m:?26
(assignvariableop_26_adam_conv1d_5_bias_m:2;
,assignvariableop_27_adam_attention_3_att_v_m:	?@
,assignvariableop_28_adam_attention_3_att_w_m:
??<
)assignvariableop_29_adam_dense_5_kernel_m:	?@5
'assignvariableop_30_adam_dense_5_bias_m:@;
)assignvariableop_31_adam_dense_6_kernel_m:5
'assignvariableop_32_adam_dense_6_bias_m:;
)assignvariableop_33_adam_dense_7_kernel_m:P 5
'assignvariableop_34_adam_dense_7_bias_m: 9
'assignvariableop_35_adam_score_kernel_m: 3
%assignvariableop_36_adam_score_bias_m:G
4assignvariableop_37_adam_lstm_3_lstm_cell_3_kernel_m:	2?	R
>assignvariableop_38_adam_lstm_3_lstm_cell_3_recurrent_kernel_m:
??	A
2assignvariableop_39_adam_lstm_3_lstm_cell_3_bias_m:	?	A
*assignvariableop_40_adam_conv1d_5_kernel_v:?26
(assignvariableop_41_adam_conv1d_5_bias_v:2;
,assignvariableop_42_adam_attention_3_att_v_v:	?@
,assignvariableop_43_adam_attention_3_att_w_v:
??<
)assignvariableop_44_adam_dense_5_kernel_v:	?@5
'assignvariableop_45_adam_dense_5_bias_v:@;
)assignvariableop_46_adam_dense_6_kernel_v:5
'assignvariableop_47_adam_dense_6_bias_v:;
)assignvariableop_48_adam_dense_7_kernel_v:P 5
'assignvariableop_49_adam_dense_7_bias_v: 9
'assignvariableop_50_adam_score_kernel_v: 3
%assignvariableop_51_adam_score_bias_v:G
4assignvariableop_52_adam_lstm_3_lstm_cell_3_kernel_v:	2?	R
>assignvariableop_53_adam_lstm_3_lstm_cell_3_recurrent_kernel_v:
??	A
2assignvariableop_54_adam_lstm_3_lstm_cell_3_bias_v:	?	
identity_56??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/att_v/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/att_W/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/att_v/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/att_W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/att_v/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/att_W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_9_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv1d_5_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv1d_5_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_attention_3_att_vIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_attention_3_att_wIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_5_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_5_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_6_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_6_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_7_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_7_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_score_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_score_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_iterIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_decayIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp&assignvariableop_17_adam_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp-assignvariableop_18_lstm_3_lstm_cell_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp7assignvariableop_19_lstm_3_lstm_cell_3_recurrent_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_lstm_3_lstm_cell_3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv1d_5_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv1d_5_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_attention_3_att_v_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_attention_3_att_w_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_5_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_6_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_6_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_7_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_7_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_score_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_score_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_lstm_3_lstm_cell_3_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp>assignvariableop_38_adam_lstm_3_lstm_cell_3_recurrent_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adam_lstm_3_lstm_cell_3_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv1d_5_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_conv1d_5_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_attention_3_att_v_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_attention_3_att_w_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_5_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_5_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_6_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_dense_6_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_7_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_dense_7_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_score_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp%assignvariableop_51_adam_score_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp4assignvariableop_52_adam_lstm_3_lstm_cell_3_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp>assignvariableop_53_adam_lstm_3_lstm_cell_3_recurrent_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_lstm_3_lstm_cell_3_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55f
Identity_56IdentityIdentity_55:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_56?

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_56Identity_56:output:0*?
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
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
?
while_cond_77720
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_77720___redundant_placeholder03
/while_while_cond_77720___redundant_placeholder13
/while_while_cond_77720___redundant_placeholder23
/while_while_cond_77720___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_77090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_77090___redundant_placeholder03
/while_while_cond_77090___redundant_placeholder13
/while_while_cond_77090___redundant_placeholder23
/while_while_cond_77090___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_75224

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consty
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????2
dropout/Mul_1k
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
@__inference_score_layer_call_and_return_conditional_losses_78483

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_78234

inputs<
)lstm_cell_3_split_readvariableop_resource:	2?	:
+lstm_cell_3_split_1_readvariableop_resource:	?	7
#lstm_cell_3_readvariableop_resource:
??	
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like/Shape
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like/Const?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/ones_like{
lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout/Const?
lstm_cell_3/dropout/MulMullstm_cell_3/ones_like:output:0"lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout/Mul?
lstm_cell_3/dropout/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout/Shape?
0lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??`22
0lstm_cell_3/dropout/random_uniform/RandomUniform?
"lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2$
"lstm_cell_3/dropout/GreaterEqual/y?
 lstm_cell_3/dropout/GreaterEqualGreaterEqual9lstm_cell_3/dropout/random_uniform/RandomUniform:output:0+lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_cell_3/dropout/GreaterEqual?
lstm_cell_3/dropout/CastCast$lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout/Cast?
lstm_cell_3/dropout/Mul_1Mullstm_cell_3/dropout/Mul:z:0lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout/Mul_1
lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_1/Const?
lstm_cell_3/dropout_1/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_1/Mul?
lstm_cell_3/dropout_1/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_1/Shape?
2lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_1/random_uniform/RandomUniform?
$lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_1/GreaterEqual/y?
"lstm_cell_3/dropout_1/GreaterEqualGreaterEqual;lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_cell_3/dropout_1/GreaterEqual?
lstm_cell_3/dropout_1/CastCast&lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout_1/Cast?
lstm_cell_3/dropout_1/Mul_1Mullstm_cell_3/dropout_1/Mul:z:0lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_1/Mul_1
lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_2/Const?
lstm_cell_3/dropout_2/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_2/Mul?
lstm_cell_3/dropout_2/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_2/Shape?
2lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_2/random_uniform/RandomUniform?
$lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_2/GreaterEqual/y?
"lstm_cell_3/dropout_2/GreaterEqualGreaterEqual;lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_cell_3/dropout_2/GreaterEqual?
lstm_cell_3/dropout_2/CastCast&lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout_2/Cast?
lstm_cell_3/dropout_2/Mul_1Mullstm_cell_3/dropout_2/Mul:z:0lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_2/Mul_1
lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_3/Const?
lstm_cell_3/dropout_3/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_3/Mul?
lstm_cell_3/dropout_3/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_3/Shape?
2lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_3/random_uniform/RandomUniform?
$lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_3/GreaterEqual/y?
"lstm_cell_3/dropout_3/GreaterEqualGreaterEqual;lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_cell_3/dropout_3/GreaterEqual?
lstm_cell_3/dropout_3/CastCast&lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout_3/Cast?
lstm_cell_3/dropout_3/Mul_1Mullstm_cell_3/dropout_3/Mul:z:0lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_3/Mul_1|
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like_1/Shape?
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like_1/Const?
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/ones_like_1
lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_4/Const?
lstm_cell_3/dropout_4/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_4/Mul?
lstm_cell_3/dropout_4/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_4/Shape?
2lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_4/random_uniform/RandomUniform?
$lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_4/GreaterEqual/y?
"lstm_cell_3/dropout_4/GreaterEqualGreaterEqual;lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_4/GreaterEqual?
lstm_cell_3/dropout_4/CastCast&lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_4/Cast?
lstm_cell_3/dropout_4/Mul_1Mullstm_cell_3/dropout_4/Mul:z:0lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_4/Mul_1
lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_5/Const?
lstm_cell_3/dropout_5/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_5/Mul?
lstm_cell_3/dropout_5/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_5/Shape?
2lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_5/random_uniform/RandomUniform?
$lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_5/GreaterEqual/y?
"lstm_cell_3/dropout_5/GreaterEqualGreaterEqual;lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_5/GreaterEqual?
lstm_cell_3/dropout_5/CastCast&lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_5/Cast?
lstm_cell_3/dropout_5/Mul_1Mullstm_cell_3/dropout_5/Mul:z:0lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_5/Mul_1
lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_6/Const?
lstm_cell_3/dropout_6/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_6/Mul?
lstm_cell_3/dropout_6/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_6/Shape?
2lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_6/random_uniform/RandomUniform?
$lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_6/GreaterEqual/y?
"lstm_cell_3/dropout_6/GreaterEqualGreaterEqual;lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_6/GreaterEqual?
lstm_cell_3/dropout_6/CastCast&lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_6/Cast?
lstm_cell_3/dropout_6/Mul_1Mullstm_cell_3/dropout_6/Mul:z:0lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_6/Mul_1
lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_7/Const?
lstm_cell_3/dropout_7/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_7/Mul?
lstm_cell_3/dropout_7/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_7/Shape?
2lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_7/random_uniform/RandomUniform?
$lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_7/GreaterEqual/y?
"lstm_cell_3/dropout_7/GreaterEqualGreaterEqual;lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_7/GreaterEqual?
lstm_cell_3/dropout_7/CastCast&lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_7/Cast?
lstm_cell_3/dropout_7/Mul_1Mullstm_cell_3/dropout_7/Mul:z:0lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_7/Mul_1?
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul?
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_1?
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_2?
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_3|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	2?	*
dtype02"
 lstm_cell_3/split/ReadVariableOp?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
lstm_cell_3/split?
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_2?
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_3?
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_3/split_1/split_dim?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02$
"lstm_cell_3/split_1/ReadVariableOp?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_3/split_1?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd?
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_1?
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_2?
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_3?
lstm_cell_3/mul_4Mulzeros:output:0lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_4?
lstm_cell_3/mul_5Mulzeros:output:0lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_5?
lstm_cell_3/mul_6Mulzeros:output:0lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_6?
lstm_cell_3/mul_7Mulzeros:output:0lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_7?
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp?
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_3/strided_slice/stack?
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice/stack_1?
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_3/strided_slice/stack_2?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice?
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_4?
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add}
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid?
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_1?
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice_1/stack?
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2%
#lstm_cell_3/strided_slice_1/stack_1?
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_1/stack_2?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_1?
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_5?
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_8?
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_2?
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell_3/strided_slice_2/stack?
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#lstm_cell_3/strided_slice_2/stack_1?
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_2/stack_2?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_2?
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_6?
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_2v
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh?
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_9?
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_3?
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_3?
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell_3/strided_slice_3/stack?
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_3/strided_slice_3/stack_1?
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_3/stack_2?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_3?
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_7?
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_4?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_2z
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh_1?
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_78036*
condR
while_cond_78035*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimep
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????2: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
??
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_78720

inputs
states_0
states_10
split_readvariableop_resource:	2?	.
split_1_readvariableop_resource:	?	+
readvariableop_resource:
??	
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22
dropout_3/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ڗ2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_6/Const?
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??+2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_7/Const?
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ѣ?2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
mul_3d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	2?	*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3g
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4g
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5g
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6g
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10f
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????2:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
D__inference_Taghipour_layer_call_and_return_conditional_losses_76331
inputs_response#
inputs_whether_criteria_applied6
"embedding_9_embedding_lookup_75971:
??K
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:?26
(conv1d_5_biasadd_readvariableop_resource:2C
0lstm_3_lstm_cell_3_split_readvariableop_resource:	2?	A
2lstm_3_lstm_cell_3_split_1_readvariableop_resource:	?	>
*lstm_3_lstm_cell_3_readvariableop_resource:
??	?
+attention_3_shape_1_readvariableop_resource:
??<
-attention_3_tensordot_readvariableop_resource:	?9
&dense_5_matmul_readvariableop_resource:	?@5
'dense_5_biasadd_readvariableop_resource:@8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:P 5
'dense_7_biasadd_readvariableop_resource: 6
$score_matmul_readvariableop_resource: 3
%score_biasadd_readvariableop_resource:
identity??$attention_3/Tensordot/ReadVariableOp?$attention_3/transpose/ReadVariableOp?conv1d_5/BiasAdd/ReadVariableOp?+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?embedding_9/embedding_lookup?!lstm_3/lstm_cell_3/ReadVariableOp?#lstm_3/lstm_cell_3/ReadVariableOp_1?#lstm_3/lstm_cell_3/ReadVariableOp_2?#lstm_3/lstm_cell_3/ReadVariableOp_3?'lstm_3/lstm_cell_3/split/ReadVariableOp?)lstm_3/lstm_cell_3/split_1/ReadVariableOp?lstm_3/while?score/BiasAdd/ReadVariableOp?score/MatMul/ReadVariableOp
embedding_9/CastCastinputs_response*

DstT0*

SrcT0*(
_output_shapes
:??????????2
embedding_9/Cast?
embedding_9/embedding_lookupResourceGather"embedding_9_embedding_lookup_75971embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*5
_class+
)'loc:@embedding_9/embedding_lookup/75971*-
_output_shapes
:???????????*
dtype02
embedding_9/embedding_lookup?
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*5
_class+
)'loc:@embedding_9/embedding_lookup/75971*-
_output_shapes
:???????????2'
%embedding_9/embedding_lookup/Identity?
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????2)
'embedding_9/embedding_lookup/Identity_1u
embedding_9/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_9/NotEqual/y?
embedding_9/NotEqualNotEqualinputs_responseembedding_9/NotEqual/y:output:0*
T0*(
_output_shapes
:??????????2
embedding_9/NotEqual?
pos_x_maskedout/CastCastembedding_9/NotEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
pos_x_maskedout/Cast?
pos_x_maskedout/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
pos_x_maskedout/ExpandDims/dim?
pos_x_maskedout/ExpandDims
ExpandDimspos_x_maskedout/Cast:y:0'pos_x_maskedout/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
pos_x_maskedout/ExpandDims?
pos_x_maskedout/stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2
pos_x_maskedout/stack?
pos_x_maskedout/TileTile#pos_x_maskedout/ExpandDims:output:0pos_x_maskedout/stack:output:0*
T0*-
_output_shapes
:???????????2
pos_x_maskedout/Tile?
pos_x_maskedout/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
pos_x_maskedout/transpose/perm?
pos_x_maskedout/transpose	Transposepos_x_maskedout/Tile:output:0'pos_x_maskedout/transpose/perm:output:0*
T0*-
_output_shapes
:???????????2
pos_x_maskedout/transpose?
pos_x_maskedout/mulMul0embedding_9/embedding_lookup/Identity_1:output:0pos_x_maskedout/transpose:y:0*
T0*-
_output_shapes
:???????????2
pos_x_maskedout/mul?
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_5/conv1d/ExpandDims/dim?
conv1d_5/conv1d/ExpandDims
ExpandDimspos_x_maskedout/mul:z:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_5/conv1d/ExpandDims?
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?2*
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim?
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?22
conv1d_5/conv1d/ExpandDims_1?
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????2*
paddingVALID*
strides
2
conv1d_5/conv1d?
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*,
_output_shapes
:??????????2*
squeeze_dims

?????????2
conv1d_5/conv1d/Squeeze?
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp?
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????22
conv1d_5/BiasAdde
lstm_3/ShapeShapeconv1d_5/BiasAdd:output:0*
T0*
_output_shapes
:2
lstm_3/Shape?
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack?
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1?
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2?
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicek
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/mul/y?
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/Less/y?
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessq
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/packed/1?
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const?
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/zeroso
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/mul/y?
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/Less/y?
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lessu
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/packed/1?
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const?
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/zeros_1?
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/perm?
lstm_3/transpose	Transposeconv1d_5/BiasAdd:output:0lstm_3/transpose/perm:output:0*
T0*,
_output_shapes
:??????????22
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1?
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stack?
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1?
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2?
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1?
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_3/TensorArrayV2/element_shape?
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2?
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensor?
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stack?
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1?
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2?
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_3/strided_slice_2?
"lstm_3/lstm_cell_3/ones_like/ShapeShapelstm_3/strided_slice_2:output:0*
T0*
_output_shapes
:2$
"lstm_3/lstm_cell_3/ones_like/Shape?
"lstm_3/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"lstm_3/lstm_cell_3/ones_like/Const?
lstm_3/lstm_cell_3/ones_likeFill+lstm_3/lstm_cell_3/ones_like/Shape:output:0+lstm_3/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/ones_like?
$lstm_3/lstm_cell_3/ones_like_1/ShapeShapelstm_3/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_3/lstm_cell_3/ones_like_1/Shape?
$lstm_3/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$lstm_3/lstm_cell_3/ones_like_1/Const?
lstm_3/lstm_cell_3/ones_like_1Fill-lstm_3/lstm_cell_3/ones_like_1/Shape:output:0-lstm_3/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
lstm_3/lstm_cell_3/ones_like_1?
lstm_3/lstm_cell_3/mulMullstm_3/strided_slice_2:output:0%lstm_3/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/mul?
lstm_3/lstm_cell_3/mul_1Mullstm_3/strided_slice_2:output:0%lstm_3/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/mul_1?
lstm_3/lstm_cell_3/mul_2Mullstm_3/strided_slice_2:output:0%lstm_3/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/mul_2?
lstm_3/lstm_cell_3/mul_3Mullstm_3/strided_slice_2:output:0%lstm_3/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/mul_3?
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_3/split/split_dim?
'lstm_3/lstm_cell_3/split/ReadVariableOpReadVariableOp0lstm_3_lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	2?	*
dtype02)
'lstm_3/lstm_cell_3/split/ReadVariableOp?
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0/lstm_3/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
lstm_3/lstm_cell_3/split?
lstm_3/lstm_cell_3/MatMulMatMullstm_3/lstm_cell_3/mul:z:0!lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul?
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/lstm_cell_3/mul_1:z:0!lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_1?
lstm_3/lstm_cell_3/MatMul_2MatMullstm_3/lstm_cell_3/mul_2:z:0!lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_2?
lstm_3/lstm_cell_3/MatMul_3MatMullstm_3/lstm_cell_3/mul_3:z:0!lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_3?
$lstm_3/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_3/lstm_cell_3/split_1/split_dim?
)lstm_3/lstm_cell_3/split_1/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02+
)lstm_3/lstm_cell_3/split_1/ReadVariableOp?
lstm_3/lstm_cell_3/split_1Split-lstm_3/lstm_cell_3/split_1/split_dim:output:01lstm_3/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_3/lstm_cell_3/split_1?
lstm_3/lstm_cell_3/BiasAddBiasAdd#lstm_3/lstm_cell_3/MatMul:product:0#lstm_3/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAdd?
lstm_3/lstm_cell_3/BiasAdd_1BiasAdd%lstm_3/lstm_cell_3/MatMul_1:product:0#lstm_3/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAdd_1?
lstm_3/lstm_cell_3/BiasAdd_2BiasAdd%lstm_3/lstm_cell_3/MatMul_2:product:0#lstm_3/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAdd_2?
lstm_3/lstm_cell_3/BiasAdd_3BiasAdd%lstm_3/lstm_cell_3/MatMul_3:product:0#lstm_3/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAdd_3?
lstm_3/lstm_cell_3/mul_4Mullstm_3/zeros:output:0'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_4?
lstm_3/lstm_cell_3/mul_5Mullstm_3/zeros:output:0'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_5?
lstm_3/lstm_cell_3/mul_6Mullstm_3/zeros:output:0'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_6?
lstm_3/lstm_cell_3/mul_7Mullstm_3/zeros:output:0'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_7?
!lstm_3/lstm_cell_3/ReadVariableOpReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02#
!lstm_3/lstm_cell_3/ReadVariableOp?
&lstm_3/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_3/lstm_cell_3/strided_slice/stack?
(lstm_3/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2*
(lstm_3/lstm_cell_3/strided_slice/stack_1?
(lstm_3/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_3/lstm_cell_3/strided_slice/stack_2?
 lstm_3/lstm_cell_3/strided_sliceStridedSlice)lstm_3/lstm_cell_3/ReadVariableOp:value:0/lstm_3/lstm_cell_3/strided_slice/stack:output:01lstm_3/lstm_cell_3/strided_slice/stack_1:output:01lstm_3/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 lstm_3/lstm_cell_3/strided_slice?
lstm_3/lstm_cell_3/MatMul_4MatMullstm_3/lstm_cell_3/mul_4:z:0)lstm_3/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_4?
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/BiasAdd:output:0%lstm_3/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add?
lstm_3/lstm_cell_3/SigmoidSigmoidlstm_3/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Sigmoid?
#lstm_3/lstm_cell_3/ReadVariableOp_1ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02%
#lstm_3/lstm_cell_3/ReadVariableOp_1?
(lstm_3/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2*
(lstm_3/lstm_cell_3/strided_slice_1/stack?
*lstm_3/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2,
*lstm_3/lstm_cell_3/strided_slice_1/stack_1?
*lstm_3/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_3/lstm_cell_3/strided_slice_1/stack_2?
"lstm_3/lstm_cell_3/strided_slice_1StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_1:value:01lstm_3/lstm_cell_3/strided_slice_1/stack:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"lstm_3/lstm_cell_3/strided_slice_1?
lstm_3/lstm_cell_3/MatMul_5MatMullstm_3/lstm_cell_3/mul_5:z:0+lstm_3/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_5?
lstm_3/lstm_cell_3/add_1AddV2%lstm_3/lstm_cell_3/BiasAdd_1:output:0%lstm_3/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add_1?
lstm_3/lstm_cell_3/Sigmoid_1Sigmoidlstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Sigmoid_1?
lstm_3/lstm_cell_3/mul_8Mul lstm_3/lstm_cell_3/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_8?
#lstm_3/lstm_cell_3/ReadVariableOp_2ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02%
#lstm_3/lstm_cell_3/ReadVariableOp_2?
(lstm_3/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2*
(lstm_3/lstm_cell_3/strided_slice_2/stack?
*lstm_3/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*lstm_3/lstm_cell_3/strided_slice_2/stack_1?
*lstm_3/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_3/lstm_cell_3/strided_slice_2/stack_2?
"lstm_3/lstm_cell_3/strided_slice_2StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_2:value:01lstm_3/lstm_cell_3/strided_slice_2/stack:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"lstm_3/lstm_cell_3/strided_slice_2?
lstm_3/lstm_cell_3/MatMul_6MatMullstm_3/lstm_cell_3/mul_6:z:0+lstm_3/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_6?
lstm_3/lstm_cell_3/add_2AddV2%lstm_3/lstm_cell_3/BiasAdd_2:output:0%lstm_3/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add_2?
lstm_3/lstm_cell_3/TanhTanhlstm_3/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Tanh?
lstm_3/lstm_cell_3/mul_9Mullstm_3/lstm_cell_3/Sigmoid:y:0lstm_3/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_9?
lstm_3/lstm_cell_3/add_3AddV2lstm_3/lstm_cell_3/mul_8:z:0lstm_3/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add_3?
#lstm_3/lstm_cell_3/ReadVariableOp_3ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02%
#lstm_3/lstm_cell_3/ReadVariableOp_3?
(lstm_3/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(lstm_3/lstm_cell_3/strided_slice_3/stack?
*lstm_3/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_3/lstm_cell_3/strided_slice_3/stack_1?
*lstm_3/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_3/lstm_cell_3/strided_slice_3/stack_2?
"lstm_3/lstm_cell_3/strided_slice_3StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_3:value:01lstm_3/lstm_cell_3/strided_slice_3/stack:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"lstm_3/lstm_cell_3/strided_slice_3?
lstm_3/lstm_cell_3/MatMul_7MatMullstm_3/lstm_cell_3/mul_7:z:0+lstm_3/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_7?
lstm_3/lstm_cell_3/add_4AddV2%lstm_3/lstm_cell_3/BiasAdd_3:output:0%lstm_3/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add_4?
lstm_3/lstm_cell_3/Sigmoid_2Sigmoidlstm_3/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Sigmoid_2?
lstm_3/lstm_cell_3/Tanh_1Tanhlstm_3/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Tanh_1?
lstm_3/lstm_cell_3/mul_10Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0lstm_3/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_10?
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2&
$lstm_3/TensorArrayV2_1/element_shape?
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/time?
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counter?
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_3_lstm_cell_3_split_readvariableop_resource2lstm_3_lstm_cell_3_split_1_readvariableop_resource*lstm_3_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_76111*#
condR
lstm_3_while_cond_76110*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_3/while?
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStack?
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_3/strided_slice_3/stack?
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1?
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2?
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_3/strided_slice_3?
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/perm?
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtime?
dropout_4/IdentityIdentitylstm_3/transpose_1:y:0*
T0*-
_output_shapes
:???????????2
dropout_4/Identityq
attention_3/ShapeShapedropout_4/Identity:output:0*
T0*
_output_shapes
:2
attention_3/Shape?
attention_3/unstackUnpackattention_3/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
attention_3/unstack?
"attention_3/Shape_1/ReadVariableOpReadVariableOp+attention_3_shape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"attention_3/Shape_1/ReadVariableOp{
attention_3/Shape_1Const*
_output_shapes
:*
dtype0*
valueB",  ,  2
attention_3/Shape_1?
attention_3/unstack_1Unpackattention_3/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
attention_3/unstack_1?
attention_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
attention_3/Reshape/shape?
attention_3/ReshapeReshapedropout_4/Identity:output:0"attention_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
attention_3/Reshape?
$attention_3/transpose/ReadVariableOpReadVariableOp+attention_3_shape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$attention_3/transpose/ReadVariableOp?
attention_3/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_3/transpose/perm?
attention_3/transpose	Transpose,attention_3/transpose/ReadVariableOp:value:0#attention_3/transpose/perm:output:0*
T0* 
_output_shapes
:
??2
attention_3/transpose?
attention_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB",  ????2
attention_3/Reshape_1/shape?
attention_3/Reshape_1Reshapeattention_3/transpose:y:0$attention_3/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
??2
attention_3/Reshape_1?
attention_3/MatMulMatMulattention_3/Reshape:output:0attention_3/Reshape_1:output:0*
T0*(
_output_shapes
:??????????2
attention_3/MatMul?
attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
attention_3/Reshape_2/shape/1?
attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
attention_3/Reshape_2/shape/2?
attention_3/Reshape_2/shapePackattention_3/unstack:output:0&attention_3/Reshape_2/shape/1:output:0&attention_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
attention_3/Reshape_2/shape?
attention_3/Reshape_2Reshapeattention_3/MatMul:product:0$attention_3/Reshape_2/shape:output:0*
T0*-
_output_shapes
:???????????2
attention_3/Reshape_2?
attention_3/TanhTanhattention_3/Reshape_2:output:0*
T0*-
_output_shapes
:???????????2
attention_3/Tanh?
$attention_3/Tensordot/ReadVariableOpReadVariableOp-attention_3_tensordot_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$attention_3/Tensordot/ReadVariableOp?
#attention_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  2%
#attention_3/Tensordot/Reshape/shape?
attention_3/Tensordot/ReshapeReshape,attention_3/Tensordot/ReadVariableOp:value:0,attention_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
attention_3/Tensordot/Reshape?
attention_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
attention_3/Tensordot/axes?
attention_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_3/Tensordot/free~
attention_3/Tensordot/ShapeShapeattention_3/Tanh:y:0*
T0*
_output_shapes
:2
attention_3/Tensordot/Shape?
#attention_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#attention_3/Tensordot/GatherV2/axis?
attention_3/Tensordot/GatherV2GatherV2$attention_3/Tensordot/Shape:output:0#attention_3/Tensordot/free:output:0,attention_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
attention_3/Tensordot/GatherV2?
%attention_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%attention_3/Tensordot/GatherV2_1/axis?
 attention_3/Tensordot/GatherV2_1GatherV2$attention_3/Tensordot/Shape:output:0#attention_3/Tensordot/axes:output:0.attention_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 attention_3/Tensordot/GatherV2_1?
attention_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
attention_3/Tensordot/Const?
attention_3/Tensordot/ProdProd'attention_3/Tensordot/GatherV2:output:0$attention_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
attention_3/Tensordot/Prod?
attention_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
attention_3/Tensordot/Const_1?
attention_3/Tensordot/Prod_1Prod)attention_3/Tensordot/GatherV2_1:output:0&attention_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
attention_3/Tensordot/Prod_1?
!attention_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!attention_3/Tensordot/concat/axis?
attention_3/Tensordot/concatConcatV2#attention_3/Tensordot/axes:output:0#attention_3/Tensordot/free:output:0*attention_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
attention_3/Tensordot/concat?
attention_3/Tensordot/stackPack%attention_3/Tensordot/Prod_1:output:0#attention_3/Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2
attention_3/Tensordot/stack?
attention_3/Tensordot/transpose	Transposeattention_3/Tanh:y:0%attention_3/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2!
attention_3/Tensordot/transpose?
attention_3/Tensordot/Reshape_1Reshape#attention_3/Tensordot/transpose:y:0$attention_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2!
attention_3/Tensordot/Reshape_1?
attention_3/Tensordot/MatMulMatMul&attention_3/Tensordot/Reshape:output:0(attention_3/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2
attention_3/Tensordot/MatMul?
attention_3/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
attention_3/Tensordot/Const_2?
#attention_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#attention_3/Tensordot/concat_1/axis?
attention_3/Tensordot/concat_1ConcatV2&attention_3/Tensordot/Const_2:output:0'attention_3/Tensordot/GatherV2:output:0,attention_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2 
attention_3/Tensordot/concat_1?
attention_3/TensordotReshape&attention_3/Tensordot/MatMul:product:0'attention_3/Tensordot/concat_1:output:0*
T0*(
_output_shapes
:??????????2
attention_3/Tensordot?
attention_3/SoftmaxSoftmaxattention_3/Tensordot:output:0*
T0*(
_output_shapes
:??????????2
attention_3/Softmaxz
attention_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
attention_3/ExpandDims/dim?
attention_3/ExpandDims
ExpandDimsattention_3/Softmax:softmax:0#attention_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
attention_3/ExpandDims{
attention_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2
attention_3/stack?
attention_3/TileTileattention_3/ExpandDims:output:0attention_3/stack:output:0*
T0*-
_output_shapes
:???????????2
attention_3/Tile?
attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
attention_3/transpose_1/perm?
attention_3/transpose_1	Transposeattention_3/Tile:output:0%attention_3/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
attention_3/transpose_1?
attention_3/mulMuldropout_4/Identity:output:0attention_3/transpose_1:y:0*
T0*-
_output_shapes
:???????????2
attention_3/mul?
!attention_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2#
!attention_3/Sum/reduction_indices?
attention_3/SumSumattention_3/mul:z:0*attention_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
attention_3/Sum?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulattention_3/Sum:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs_whether_criteria_applied%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2dense_5/BiasAdd:output:0dense_6/BiasAdd:output:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????P2
concatenate_1/concat?
dropout_5/IdentityIdentityconcatenate_1/concat:output:0*
T0*'
_output_shapes
:?????????P2
dropout_5/Identity?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:P *
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldropout_5/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_7/BiasAdd?
score/MatMul/ReadVariableOpReadVariableOp$score_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
score/MatMul/ReadVariableOp?
score/MatMulMatMuldense_7/BiasAdd:output:0#score/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
score/MatMul?
score/BiasAdd/ReadVariableOpReadVariableOp%score_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
score/BiasAdd/ReadVariableOp?
score/BiasAddBiasAddscore/MatMul:product:0$score/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
score/BiasAdds
score/SoftmaxSoftmaxscore/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
score/Softmaxr
IdentityIdentityscore/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp%^attention_3/Tensordot/ReadVariableOp%^attention_3/transpose/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/conv1d/ExpandDims_1/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^embedding_9/embedding_lookup"^lstm_3/lstm_cell_3/ReadVariableOp$^lstm_3/lstm_cell_3/ReadVariableOp_1$^lstm_3/lstm_cell_3/ReadVariableOp_2$^lstm_3/lstm_cell_3/ReadVariableOp_3(^lstm_3/lstm_cell_3/split/ReadVariableOp*^lstm_3/lstm_cell_3/split_1/ReadVariableOp^lstm_3/while^score/BiasAdd/ReadVariableOp^score/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 2L
$attention_3/Tensordot/ReadVariableOp$attention_3/Tensordot/ReadVariableOp2L
$attention_3/transpose/ReadVariableOp$attention_3/transpose/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2<
embedding_9/embedding_lookupembedding_9/embedding_lookup2F
!lstm_3/lstm_cell_3/ReadVariableOp!lstm_3/lstm_cell_3/ReadVariableOp2J
#lstm_3/lstm_cell_3/ReadVariableOp_1#lstm_3/lstm_cell_3/ReadVariableOp_12J
#lstm_3/lstm_cell_3/ReadVariableOp_2#lstm_3/lstm_cell_3/ReadVariableOp_22J
#lstm_3/lstm_cell_3/ReadVariableOp_3#lstm_3/lstm_cell_3/ReadVariableOp_32R
'lstm_3/lstm_cell_3/split/ReadVariableOp'lstm_3/lstm_cell_3/split/ReadVariableOp2V
)lstm_3/lstm_cell_3/split_1/ReadVariableOp)lstm_3/lstm_cell_3/split_1/ReadVariableOp2
lstm_3/whilelstm_3/while2<
score/BiasAdd/ReadVariableOpscore/BiasAdd/ReadVariableOp2:
score/MatMul/ReadVariableOpscore/MatMul/ReadVariableOp:Y U
(
_output_shapes
:??????????
)
_user_specified_nameinputs/response:hd
'
_output_shapes
:?????????
9
_user_specified_name!inputs/whether_criteria_applied
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_77225
inputs_0<
)lstm_cell_3_split_readvariableop_resource:	2?	:
+lstm_cell_3_split_1_readvariableop_resource:	?	7
#lstm_cell_3_readvariableop_resource:
??	
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like/Shape
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like/Const?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/ones_like|
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like_1/Shape?
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like_1/Const?
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/ones_like_1?
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul?
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_1?
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_2?
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_3|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	2?	*
dtype02"
 lstm_cell_3/split/ReadVariableOp?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
lstm_cell_3/split?
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_2?
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_3?
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_3/split_1/split_dim?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02$
"lstm_cell_3/split_1/ReadVariableOp?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_3/split_1?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd?
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_1?
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_2?
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_3?
lstm_cell_3/mul_4Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_4?
lstm_cell_3/mul_5Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_5?
lstm_cell_3/mul_6Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_6?
lstm_cell_3/mul_7Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_7?
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp?
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_3/strided_slice/stack?
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice/stack_1?
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_3/strided_slice/stack_2?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice?
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_4?
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add}
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid?
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_1?
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice_1/stack?
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2%
#lstm_cell_3/strided_slice_1/stack_1?
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_1/stack_2?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_1?
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_5?
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_8?
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_2?
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell_3/strided_slice_2/stack?
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#lstm_cell_3/strided_slice_2/stack_1?
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_2/stack_2?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_2?
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_6?
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_2v
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh?
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_9?
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_3?
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_3?
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell_3/strided_slice_3/stack?
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_3/strided_slice_3/stack_1?
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_3/stack_2?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_3?
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_7?
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_4?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_2z
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh_1?
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_77091*
condR
while_cond_77090*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????2: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?
?
while_cond_73958
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_73958___redundant_placeholder03
/while_while_cond_73958___redundant_placeholder13
/while_while_cond_73958___redundant_placeholder23
/while_while_cond_73958___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?%
?
while_body_74283
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_3_74307_0:	2?	(
while_lstm_cell_3_74309_0:	?	-
while_lstm_cell_3_74311_0:
??	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_3_74307:	2?	&
while_lstm_cell_3_74309:	?	+
while_lstm_cell_3_74311:
??	??)while/lstm_cell_3/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_74307_0while_lstm_cell_3_74309_0while_lstm_cell_3_74311_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_742052+
)while/lstm_cell_3/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_74307while_lstm_cell_3_74307_0"4
while_lstm_cell_3_74309while_lstm_cell_3_74309_0"4
while_lstm_cell_3_74311while_lstm_cell_3_74311_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_dense_5_layer_call_fn_78394

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_750172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_78035
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_78035___redundant_placeholder03
/while_while_cond_78035___redundant_placeholder13
/while_while_cond_78035___redundant_placeholder23
/while_while_cond_78035___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
B__inference_dense_6_layer_call_and_return_conditional_losses_78404

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_5_layer_call_fn_78448

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
D__inference_dropout_5_layer_call_and_return_conditional_losses_750532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
&__inference_lstm_3_layer_call_fn_78278

inputs
unknown:	2?	
	unknown_0:	?	
	unknown_1:
??	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_756192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?
?
+__inference_lstm_cell_3_layer_call_fn_78754

inputs
states_0
states_1
unknown:	2?	
	unknown_0:	?	
	unknown_1:
??	
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_742052
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????2:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?:
?
F__inference_attention_3_layer_call_and_return_conditional_losses_78366
x3
shape_1_readvariableop_resource:
??0
!tensordot_readvariableop_resource:	?
identity??Tensordot/ReadVariableOp?transpose/ReadVariableOp?
ShapeShapex*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB",  ,  2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Reshape/shapek
ReshapeReshapexReshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0* 
_output_shapes
:
??2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB",  ????2
Reshape_1/shapeu
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
??2
	Reshape_1s
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:??????????2
MatMuli
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape_2/shape/1i
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*-
_output_shapes
:???????????2
	Reshape_2`
TanhTanhReshape_2:output:0*
T0*-
_output_shapes
:???????????2
Tanh?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes	
:?*
dtype02
Tensordot/ReadVariableOp?
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  2
Tensordot/Reshape/shape?
Tensordot/ReshapeReshape Tensordot/ReadVariableOp:value:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
Tensordot/Reshapej
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeZ
Tensordot/ShapeShapeTanh:y:0*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/axes:output:0Tensordot/free:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod_1:output:0Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeTanh:y:0Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2
Tensordot/transpose?
Tensordot/Reshape_1ReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape_1?
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMuli
Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/Const_2:output:0Tensordot/GatherV2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*(
_output_shapes
:??????????2
	Tensordotd
SoftmaxSoftmaxTensordot:output:0*
T0*(
_output_shapes
:??????????2	
Softmaxb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsSoftmax:softmax:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2
stackq
TileTileExpandDims:output:0stack:output:0*
T0*-
_output_shapes
:???????????2
Tiley
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeTile:output:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1]
mulMulxtranspose_1:y:0*
T0*-
_output_shapes
:???????????2
mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesm
SumSummul:z:0Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
Sumh
IdentityIdentitySum:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:P L
-
_output_shapes
:???????????

_user_specified_namex
ψ
?	
while_body_74791
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	2?	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	?	?
+while_lstm_cell_3_readvariableop_resource_0:
??	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	2?	@
1while_lstm_cell_3_split_1_readvariableop_resource:	?	=
)while_lstm_cell_3_readvariableop_resource:
??	?? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/ones_like/Shape?
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_3/ones_like/Const?
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/ones_like?
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_3/ones_like_1/Shape?
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_3/ones_like_1/Const?
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/ones_like_1?
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul?
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_1?
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_2?
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_3?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	2?	*
dtype02(
&while/lstm_cell_3/split/ReadVariableOp?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_2?
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_3?
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_3/split_1/split_dim?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?	*
dtype02*
(while/lstm_cell_3/split_1/ReadVariableOp?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_3/split_1?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd?
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_1?
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_2?
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_3?
while/lstm_cell_3/mul_4Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_4?
while/lstm_cell_3/mul_5Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_5?
while/lstm_cell_3/mul_6Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_6?
while/lstm_cell_3/mul_7Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_7?
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02"
 while/lstm_cell_3/ReadVariableOp?
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_3/strided_slice/stack?
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice/stack_1?
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_3/strided_slice/stack_2?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_3/strided_slice?
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_4?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid?
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_1?
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice_1/stack?
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2+
)while/lstm_cell_3/strided_slice_1/stack_1?
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_1/stack_2?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_1?
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_5?
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_8?
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_2?
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell_3/strided_slice_2/stack?
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2+
)while/lstm_cell_3/strided_slice_2/stack_1?
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_2/stack_2?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_2?
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_6?
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_2?
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh?
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_9?
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_3?
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_3?
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell_3/strided_slice_3/stack?
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_3/strided_slice_3/stack_1?
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_3/stack_2?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_3?
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_7?
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_4?
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh_1?
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_74205

inputs

states
states_10
split_readvariableop_resource:	2?	.
split_1_readvariableop_resource:	?	+
readvariableop_resource:
??	
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??%2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_3/Const?
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2˨?2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22
dropout_3/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2갥2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?[2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_6/Const?
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/Shape?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_6/GreaterEqual/y?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_6/GreaterEqual?
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_6/Cast?
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_7/Const?
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/Shape?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??O2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout_7/GreaterEqual/y?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_7/GreaterEqual?
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_7/Cast?
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_7/Mul_1^
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
muld
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
mul_1d
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
mul_2d
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
mul_3d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	2?	*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3e
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_4e
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_5e
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_6e
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10f
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????2:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?

?
lstm_3_while_cond_76539*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_76539___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_76539___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_76539___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_76539___redundant_placeholder3
lstm_3_while_identity
?
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_78385

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_78283

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_Taghipour_layer_call_fn_76876
inputs_response#
inputs_whether_criteria_applied
unknown:
?? 
	unknown_0:?2
	unknown_1:2
	unknown_2:	2?	
	unknown_3:	?	
	unknown_4:
??	
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:

unknown_10:

unknown_11:P 

unknown_12: 

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_responseinputs_whether_criteria_appliedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Taghipour_layer_call_and_return_conditional_losses_750892
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:??????????
)
_user_specified_nameinputs/response:hd
'
_output_shapes
:?????????
9
_user_specified_name!inputs/whether_criteria_applied
??
?	
while_body_78036
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	2?	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	?	?
+while_lstm_cell_3_readvariableop_resource_0:
??	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	2?	@
1while_lstm_cell_3_split_1_readvariableop_resource:	?	=
)while_lstm_cell_3_readvariableop_resource:
??	?? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/ones_like/Shape?
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_3/ones_like/Const?
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/ones_like?
while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2!
while/lstm_cell_3/dropout/Const?
while/lstm_cell_3/dropout/MulMul$while/lstm_cell_3/ones_like:output:0(while/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/dropout/Mul?
while/lstm_cell_3/dropout/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_3/dropout/Shape?
6while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??28
6while/lstm_cell_3/dropout/random_uniform/RandomUniform?
(while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(while/lstm_cell_3/dropout/GreaterEqual/y?
&while/lstm_cell_3/dropout/GreaterEqualGreaterEqual?while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22(
&while/lstm_cell_3/dropout/GreaterEqual?
while/lstm_cell_3/dropout/CastCast*while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22 
while/lstm_cell_3/dropout/Cast?
while/lstm_cell_3/dropout/Mul_1Mul!while/lstm_cell_3/dropout/Mul:z:0"while/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout/Mul_1?
!while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_1/Const?
while/lstm_cell_3/dropout_1/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout_1/Mul?
!while/lstm_cell_3/dropout_1/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_1/Shape?
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_1/GreaterEqual/y?
(while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22*
(while/lstm_cell_3/dropout_1/GreaterEqual?
 while/lstm_cell_3/dropout_1/CastCast,while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22"
 while/lstm_cell_3/dropout_1/Cast?
!while/lstm_cell_3/dropout_1/Mul_1Mul#while/lstm_cell_3/dropout_1/Mul:z:0$while/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22#
!while/lstm_cell_3/dropout_1/Mul_1?
!while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_2/Const?
while/lstm_cell_3/dropout_2/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout_2/Mul?
!while/lstm_cell_3/dropout_2/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_2/Shape?
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2?ͬ2:
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_2/GreaterEqual/y?
(while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22*
(while/lstm_cell_3/dropout_2/GreaterEqual?
 while/lstm_cell_3/dropout_2/CastCast,while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22"
 while/lstm_cell_3/dropout_2/Cast?
!while/lstm_cell_3/dropout_2/Mul_1Mul#while/lstm_cell_3/dropout_2/Mul:z:0$while/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22#
!while/lstm_cell_3/dropout_2/Mul_1?
!while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_3/Const?
while/lstm_cell_3/dropout_3/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout_3/Mul?
!while/lstm_cell_3/dropout_3/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_3/Shape?
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_3/GreaterEqual/y?
(while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22*
(while/lstm_cell_3/dropout_3/GreaterEqual?
 while/lstm_cell_3/dropout_3/CastCast,while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22"
 while/lstm_cell_3/dropout_3/Cast?
!while/lstm_cell_3/dropout_3/Mul_1Mul#while/lstm_cell_3/dropout_3/Mul:z:0$while/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22#
!while/lstm_cell_3/dropout_3/Mul_1?
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_3/ones_like_1/Shape?
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_3/ones_like_1/Const?
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/ones_like_1?
!while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_4/Const?
while/lstm_cell_3/dropout_4/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_4/Mul?
!while/lstm_cell_3/dropout_4/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_4/Shape?
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ۖ2:
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_4/GreaterEqual/y?
(while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_4/GreaterEqual?
 while/lstm_cell_3/dropout_4/CastCast,while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_4/Cast?
!while/lstm_cell_3/dropout_4/Mul_1Mul#while/lstm_cell_3/dropout_4/Mul:z:0$while/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_4/Mul_1?
!while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_5/Const?
while/lstm_cell_3/dropout_5/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_5/Mul?
!while/lstm_cell_3/dropout_5/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_5/Shape?
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??a2:
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_5/GreaterEqual/y?
(while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_5/GreaterEqual?
 while/lstm_cell_3/dropout_5/CastCast,while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_5/Cast?
!while/lstm_cell_3/dropout_5/Mul_1Mul#while/lstm_cell_3/dropout_5/Mul:z:0$while/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_5/Mul_1?
!while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_6/Const?
while/lstm_cell_3/dropout_6/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_6/Mul?
!while/lstm_cell_3/dropout_6/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_6/Shape?
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??F2:
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_6/GreaterEqual/y?
(while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_6/GreaterEqual?
 while/lstm_cell_3/dropout_6/CastCast,while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_6/Cast?
!while/lstm_cell_3/dropout_6/Mul_1Mul#while/lstm_cell_3/dropout_6/Mul:z:0$while/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_6/Mul_1?
!while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_7/Const?
while/lstm_cell_3/dropout_7/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_7/Mul?
!while/lstm_cell_3/dropout_7/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_7/Shape?
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_7/GreaterEqual/y?
(while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_7/GreaterEqual?
 while/lstm_cell_3/dropout_7/CastCast,while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_7/Cast?
!while/lstm_cell_3/dropout_7/Mul_1Mul#while/lstm_cell_3/dropout_7/Mul:z:0$while/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_7/Mul_1?
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul?
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_1?
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_2?
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_3?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	2?	*
dtype02(
&while/lstm_cell_3/split/ReadVariableOp?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_2?
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_3?
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_3/split_1/split_dim?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?	*
dtype02*
(while/lstm_cell_3/split_1/ReadVariableOp?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_3/split_1?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd?
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_1?
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_2?
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_3?
while/lstm_cell_3/mul_4Mulwhile_placeholder_2%while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_4?
while/lstm_cell_3/mul_5Mulwhile_placeholder_2%while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_5?
while/lstm_cell_3/mul_6Mulwhile_placeholder_2%while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_6?
while/lstm_cell_3/mul_7Mulwhile_placeholder_2%while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_7?
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02"
 while/lstm_cell_3/ReadVariableOp?
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_3/strided_slice/stack?
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice/stack_1?
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_3/strided_slice/stack_2?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_3/strided_slice?
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_4?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid?
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_1?
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice_1/stack?
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2+
)while/lstm_cell_3/strided_slice_1/stack_1?
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_1/stack_2?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_1?
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_5?
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_8?
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_2?
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell_3/strided_slice_2/stack?
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2+
)while/lstm_cell_3/strided_slice_2/stack_1?
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_2/stack_2?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_2?
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_6?
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_2?
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh?
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_9?
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_3?
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_3?
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell_3/strided_slice_3/stack?
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_3/strided_slice_3/stack_1?
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_3/stack_2?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_3?
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_7?
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_4?
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh_1?
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?F
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_74028

inputs$
lstm_cell_3_73946:	2?	 
lstm_cell_3_73948:	?	%
lstm_cell_3_73950:
??	
identity??#lstm_cell_3/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_73946lstm_cell_3_73948lstm_cell_3_73950*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_739452%
#lstm_cell_3/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_73946lstm_cell_3_73948lstm_cell_3_73950*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_73959*
condR
while_cond_73958*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity|
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????2: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?
?
while_cond_74282
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_74282___redundant_placeholder03
/while_while_cond_74282___redundant_placeholder13
/while_while_cond_74282___redundant_placeholder23
/while_while_cond_74282___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?;
?
D__inference_Taghipour_layer_call_and_return_conditional_losses_75089

inputs
inputs_1%
embedding_9_74634:
??%
conv1d_5_74670:?2
conv1d_5_74672:2
lstm_3_74926:	2?	
lstm_3_74928:	?	 
lstm_3_74930:
??	%
attention_3_75002:
?? 
attention_3_75004:	? 
dense_5_75018:	?@
dense_5_75020:@
dense_6_75034:
dense_6_75036:
dense_7_75066:P 
dense_7_75068: 
score_75083: 
score_75085:
identity??#attention_3/StatefulPartitionedCall? conv1d_5/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?score/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_9_74634*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_9_layer_call_and_return_conditional_losses_746332%
#embedding_9/StatefulPartitionedCallu
embedding_9/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_9/NotEqual/y?
embedding_9/NotEqualNotEqualinputsembedding_9/NotEqual/y:output:0*
T0*(
_output_shapes
:??????????2
embedding_9/NotEqual?
pos_x_maskedout/PartitionedCallPartitionedCall,embedding_9/StatefulPartitionedCall:output:0embedding_9/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_pos_x_maskedout_layer_call_and_return_conditional_losses_746522!
pos_x_maskedout/PartitionedCall?
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall(pos_x_maskedout/PartitionedCall:output:0conv1d_5_74670conv1d_5_74672*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_746692"
 conv1d_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0lstm_3_74926lstm_3_74928lstm_3_74930*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_749252 
lstm_3/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_749382
dropout_4/PartitionedCall?
#attention_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0attention_3_75002attention_3_75004*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_attention_3_layer_call_and_return_conditional_losses_750012%
#attention_3/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall,attention_3/StatefulPartitionedCall:output:0dense_5_75018dense_5_75020*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_750172!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_6_75034dense_6_75036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_750332!
dense_6/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_750462
concatenate_1/PartitionedCall?
dropout_5/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_750532
dropout_5/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_7_75066dense_7_75068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_750652!
dense_7/StatefulPartitionedCall?
score/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0score_75083score_75085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_score_layer_call_and_return_conditional_losses_750822
score/StatefulPartitionedCall?
IdentityIdentity&score/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp$^attention_3/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^score/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 2J
#attention_3/StatefulPartitionedCall#attention_3/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2>
score/StatefulPartitionedCallscore/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

k
J__inference_pos_x_maskedout_layer_call_and_return_conditional_losses_76944
x
mask

identity\
CastCastmask*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
Castb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2
stackq
TileTileExpandDims:output:0stack:output:0*
T0*-
_output_shapes
:???????????2
Tileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	TransposeTile:output:0transpose/perm:output:0*
T0*-
_output_shapes
:???????????2
	transpose[
mulMulxtranspose:y:0*
T0*-
_output_shapes
:???????????2
mula
IdentityIdentitymul:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????:??????????:P L
-
_output_shapes
:???????????

_user_specified_namex:NJ
(
_output_shapes
:??????????

_user_specified_namemask
?
?
(__inference_conv1d_5_layer_call_fn_76974

inputs
unknown:?2
	unknown_0:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_746692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_75966
applied
response
unknown:
?? 
	unknown_0:?2
	unknown_1:2
	unknown_2:	2?	
	unknown_3:	?	
	unknown_4:
??	
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:

unknown_10:

unknown_11:P 

unknown_12: 

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallresponseappliedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_738202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????:??????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	applied:RN
(
_output_shapes
:??????????
"
_user_specified_name
response
?
?
@__inference_score_layer_call_and_return_conditional_losses_75082

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
ψ
?	
while_body_77091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	2?	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	?	?
+while_lstm_cell_3_readvariableop_resource_0:
??	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	2?	@
1while_lstm_cell_3_split_1_readvariableop_resource:	?	=
)while_lstm_cell_3_readvariableop_resource:
??	?? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/ones_like/Shape?
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_3/ones_like/Const?
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/ones_like?
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_3/ones_like_1/Shape?
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_3/ones_like_1/Const?
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/ones_like_1?
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul?
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_1?
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_2?
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_3?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	2?	*
dtype02(
&while/lstm_cell_3/split/ReadVariableOp?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_2?
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_3?
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_3/split_1/split_dim?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?	*
dtype02*
(while/lstm_cell_3/split_1/ReadVariableOp?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_3/split_1?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd?
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_1?
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_2?
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_3?
while/lstm_cell_3/mul_4Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_4?
while/lstm_cell_3/mul_5Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_5?
while/lstm_cell_3/mul_6Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_6?
while/lstm_cell_3/mul_7Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_7?
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02"
 while/lstm_cell_3/ReadVariableOp?
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_3/strided_slice/stack?
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice/stack_1?
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_3/strided_slice/stack_2?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_3/strided_slice?
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_4?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid?
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_1?
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice_1/stack?
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2+
)while/lstm_cell_3/strided_slice_1/stack_1?
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_1/stack_2?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_1?
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_5?
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_8?
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_2?
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell_3/strided_slice_2/stack?
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2+
)while/lstm_cell_3/strided_slice_2/stack_1?
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_2/stack_2?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_2?
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_6?
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_2?
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh?
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_9?
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_3?
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_3?
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell_3/strided_slice_3/stack?
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_3/strided_slice_3/stack_1?
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_3/stack_2?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_3?
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_7?
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_4?
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh_1?
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
%__inference_score_layer_call_fn_78492

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_score_layer_call_and_return_conditional_losses_750822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
+__inference_attention_3_layer_call_fn_78375
x
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_attention_3_layer_call_and_return_conditional_losses_750012
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
-
_output_shapes
:???????????

_user_specified_namex
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_74938

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_77855

inputs<
)lstm_cell_3_split_readvariableop_resource:	2?	:
+lstm_cell_3_split_1_readvariableop_resource:	?	7
#lstm_cell_3_readvariableop_resource:
??	
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like/Shape
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like/Const?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/ones_like|
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like_1/Shape?
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like_1/Const?
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/ones_like_1?
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul?
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_1?
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_2?
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_3|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	2?	*
dtype02"
 lstm_cell_3/split/ReadVariableOp?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
lstm_cell_3/split?
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_2?
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_3?
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_3/split_1/split_dim?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02$
"lstm_cell_3/split_1/ReadVariableOp?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_3/split_1?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd?
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_1?
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_2?
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_3?
lstm_cell_3/mul_4Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_4?
lstm_cell_3/mul_5Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_5?
lstm_cell_3/mul_6Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_6?
lstm_cell_3/mul_7Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_7?
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp?
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_3/strided_slice/stack?
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice/stack_1?
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_3/strided_slice/stack_2?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice?
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_4?
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add}
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid?
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_1?
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice_1/stack?
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2%
#lstm_cell_3/strided_slice_1/stack_1?
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_1/stack_2?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_1?
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_5?
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_8?
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_2?
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell_3/strided_slice_2/stack?
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#lstm_cell_3/strided_slice_2/stack_1?
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_2/stack_2?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_2?
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_6?
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_2v
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh?
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_9?
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_3?
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_3?
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell_3/strided_slice_3/stack?
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_3/strided_slice_3/stack_1?
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_3/stack_2?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_3?
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_7?
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_4?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_2z
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh_1?
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_77721*
condR
while_cond_77720*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimep
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????2: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?F
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_74352

inputs$
lstm_cell_3_74270:	2?	 
lstm_cell_3_74272:	?	%
lstm_cell_3_74274:
??	
identity??#lstm_cell_3/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_74270lstm_cell_3_74272lstm_cell_3_74274*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_742052%
#lstm_cell_3/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_74270lstm_cell_3_74272lstm_cell_3_74274*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_74283*
condR
while_cond_74282*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity|
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????2: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_78431

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
)__inference_Taghipour_layer_call_fn_75124
response
applied
unknown:
?? 
	unknown_0:?2
	unknown_1:2
	unknown_2:	2?	
	unknown_3:	?	
	unknown_4:
??	
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:

unknown_10:

unknown_11:P 

unknown_12: 

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallresponseappliedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Taghipour_layer_call_and_return_conditional_losses_750892
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
response:PL
'
_output_shapes
:?????????
!
_user_specified_name	applied
?
E
)__inference_dropout_4_layer_call_fn_78300

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_749382
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_74925

inputs<
)lstm_cell_3_split_readvariableop_resource:	2?	:
+lstm_cell_3_split_1_readvariableop_resource:	?	7
#lstm_cell_3_readvariableop_resource:
??	
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like/Shape
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like/Const?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/ones_like|
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like_1/Shape?
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like_1/Const?
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/ones_like_1?
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul?
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_1?
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_2?
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_3|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	2?	*
dtype02"
 lstm_cell_3/split/ReadVariableOp?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
lstm_cell_3/split?
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_2?
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_3?
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_3/split_1/split_dim?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02$
"lstm_cell_3/split_1/ReadVariableOp?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_3/split_1?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd?
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_1?
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_2?
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_3?
lstm_cell_3/mul_4Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_4?
lstm_cell_3/mul_5Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_5?
lstm_cell_3/mul_6Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_6?
lstm_cell_3/mul_7Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_7?
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp?
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_3/strided_slice/stack?
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice/stack_1?
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_3/strided_slice/stack_2?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice?
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_4?
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add}
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid?
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_1?
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice_1/stack?
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2%
#lstm_cell_3/strided_slice_1/stack_1?
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_1/stack_2?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_1?
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_5?
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_8?
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_2?
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell_3/strided_slice_2/stack?
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#lstm_cell_3/strided_slice_2/stack_1?
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_2/stack_2?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_2?
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_6?
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_2v
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh?
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_9?
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_3?
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_3?
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell_3/strided_slice_3/stack?
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_3/strided_slice_3/stack_1?
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_3/stack_2?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_3?
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_7?
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_4?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_2z
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh_1?
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_74791*
condR
while_cond_74790*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimep
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????2: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?:
?
F__inference_attention_3_layer_call_and_return_conditional_losses_75001
x3
shape_1_readvariableop_resource:
??0
!tensordot_readvariableop_resource:	?
identity??Tensordot/ReadVariableOp?transpose/ReadVariableOp?
ShapeShapex*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB",  ,  2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Reshape/shapek
ReshapeReshapexReshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0* 
_output_shapes
:
??2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB",  ????2
Reshape_1/shapeu
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
??2
	Reshape_1s
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:??????????2
MatMuli
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape_2/shape/1i
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*-
_output_shapes
:???????????2
	Reshape_2`
TanhTanhReshape_2:output:0*
T0*-
_output_shapes
:???????????2
Tanh?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes	
:?*
dtype02
Tensordot/ReadVariableOp?
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  2
Tensordot/Reshape/shape?
Tensordot/ReshapeReshape Tensordot/ReadVariableOp:value:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
Tensordot/Reshapej
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeZ
Tensordot/ShapeShapeTanh:y:0*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/axes:output:0Tensordot/free:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod_1:output:0Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeTanh:y:0Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2
Tensordot/transpose?
Tensordot/Reshape_1ReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape_1?
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMuli
Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/Const_2:output:0Tensordot/GatherV2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*(
_output_shapes
:??????????2
	Tensordotd
SoftmaxSoftmaxTensordot:output:0*
T0*(
_output_shapes
:??????????2	
Softmaxb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsSoftmax:softmax:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2
stackq
TileTileExpandDims:output:0stack:output:0*
T0*-
_output_shapes
:???????????2
Tiley
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	TransposeTile:output:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1]
mulMulxtranspose_1:y:0*
T0*-
_output_shapes
:???????????2
mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesm
SumSummul:z:0Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
Sumh
IdentityIdentitySum:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:P L
-
_output_shapes
:???????????

_user_specified_namex
?
?
&__inference_lstm_3_layer_call_fn_78267

inputs
unknown:	2?	
	unknown_0:	?	
	unknown_1:
??	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_749252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
??
?	
while_body_75421
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	2?	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	?	?
+while_lstm_cell_3_readvariableop_resource_0:
??	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	2?	@
1while_lstm_cell_3_split_1_readvariableop_resource:	?	=
)while_lstm_cell_3_readvariableop_resource:
??	?? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/ones_like/Shape?
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_3/ones_like/Const?
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/ones_like?
while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2!
while/lstm_cell_3/dropout/Const?
while/lstm_cell_3/dropout/MulMul$while/lstm_cell_3/ones_like:output:0(while/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/dropout/Mul?
while/lstm_cell_3/dropout/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_3/dropout/Shape?
6while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???28
6while/lstm_cell_3/dropout/random_uniform/RandomUniform?
(while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(while/lstm_cell_3/dropout/GreaterEqual/y?
&while/lstm_cell_3/dropout/GreaterEqualGreaterEqual?while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22(
&while/lstm_cell_3/dropout/GreaterEqual?
while/lstm_cell_3/dropout/CastCast*while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22 
while/lstm_cell_3/dropout/Cast?
while/lstm_cell_3/dropout/Mul_1Mul!while/lstm_cell_3/dropout/Mul:z:0"while/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout/Mul_1?
!while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_1/Const?
while/lstm_cell_3/dropout_1/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout_1/Mul?
!while/lstm_cell_3/dropout_1/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_1/Shape?
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_1/GreaterEqual/y?
(while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22*
(while/lstm_cell_3/dropout_1/GreaterEqual?
 while/lstm_cell_3/dropout_1/CastCast,while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22"
 while/lstm_cell_3/dropout_1/Cast?
!while/lstm_cell_3/dropout_1/Mul_1Mul#while/lstm_cell_3/dropout_1/Mul:z:0$while/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22#
!while/lstm_cell_3/dropout_1/Mul_1?
!while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_2/Const?
while/lstm_cell_3/dropout_2/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout_2/Mul?
!while/lstm_cell_3/dropout_2/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_2/Shape?
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_2/GreaterEqual/y?
(while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22*
(while/lstm_cell_3/dropout_2/GreaterEqual?
 while/lstm_cell_3/dropout_2/CastCast,while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22"
 while/lstm_cell_3/dropout_2/Cast?
!while/lstm_cell_3/dropout_2/Mul_1Mul#while/lstm_cell_3/dropout_2/Mul:z:0$while/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22#
!while/lstm_cell_3/dropout_2/Mul_1?
!while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_3/Const?
while/lstm_cell_3/dropout_3/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout_3/Mul?
!while/lstm_cell_3/dropout_3/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_3/Shape?
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_3/GreaterEqual/y?
(while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22*
(while/lstm_cell_3/dropout_3/GreaterEqual?
 while/lstm_cell_3/dropout_3/CastCast,while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22"
 while/lstm_cell_3/dropout_3/Cast?
!while/lstm_cell_3/dropout_3/Mul_1Mul#while/lstm_cell_3/dropout_3/Mul:z:0$while/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22#
!while/lstm_cell_3/dropout_3/Mul_1?
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_3/ones_like_1/Shape?
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_3/ones_like_1/Const?
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/ones_like_1?
!while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_4/Const?
while/lstm_cell_3/dropout_4/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_4/Mul?
!while/lstm_cell_3/dropout_4/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_4/Shape?
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??C2:
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_4/GreaterEqual/y?
(while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_4/GreaterEqual?
 while/lstm_cell_3/dropout_4/CastCast,while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_4/Cast?
!while/lstm_cell_3/dropout_4/Mul_1Mul#while/lstm_cell_3/dropout_4/Mul:z:0$while/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_4/Mul_1?
!while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_5/Const?
while/lstm_cell_3/dropout_5/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_5/Mul?
!while/lstm_cell_3/dropout_5/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_5/Shape?
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_5/GreaterEqual/y?
(while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_5/GreaterEqual?
 while/lstm_cell_3/dropout_5/CastCast,while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_5/Cast?
!while/lstm_cell_3/dropout_5/Mul_1Mul#while/lstm_cell_3/dropout_5/Mul:z:0$while/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_5/Mul_1?
!while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_6/Const?
while/lstm_cell_3/dropout_6/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_6/Mul?
!while/lstm_cell_3/dropout_6/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_6/Shape?
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?̎2:
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_6/GreaterEqual/y?
(while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_6/GreaterEqual?
 while/lstm_cell_3/dropout_6/CastCast,while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_6/Cast?
!while/lstm_cell_3/dropout_6/Mul_1Mul#while/lstm_cell_3/dropout_6/Mul:z:0$while/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_6/Mul_1?
!while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_7/Const?
while/lstm_cell_3/dropout_7/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_7/Mul?
!while/lstm_cell_3/dropout_7/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_7/Shape?
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_7/GreaterEqual/y?
(while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_7/GreaterEqual?
 while/lstm_cell_3/dropout_7/CastCast,while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_7/Cast?
!while/lstm_cell_3/dropout_7/Mul_1Mul#while/lstm_cell_3/dropout_7/Mul:z:0$while/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_7/Mul_1?
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul?
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_1?
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_2?
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_3?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	2?	*
dtype02(
&while/lstm_cell_3/split/ReadVariableOp?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_2?
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_3?
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_3/split_1/split_dim?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?	*
dtype02*
(while/lstm_cell_3/split_1/ReadVariableOp?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_3/split_1?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd?
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_1?
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_2?
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_3?
while/lstm_cell_3/mul_4Mulwhile_placeholder_2%while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_4?
while/lstm_cell_3/mul_5Mulwhile_placeholder_2%while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_5?
while/lstm_cell_3/mul_6Mulwhile_placeholder_2%while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_6?
while/lstm_cell_3/mul_7Mulwhile_placeholder_2%while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_7?
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02"
 while/lstm_cell_3/ReadVariableOp?
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_3/strided_slice/stack?
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice/stack_1?
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_3/strided_slice/stack_2?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_3/strided_slice?
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_4?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid?
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_1?
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice_1/stack?
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2+
)while/lstm_cell_3/strided_slice_1/stack_1?
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_1/stack_2?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_1?
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_5?
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_8?
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_2?
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell_3/strided_slice_2/stack?
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2+
)while/lstm_cell_3/strided_slice_2/stack_1?
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_2/stack_2?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_2?
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_6?
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_2?
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh?
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_9?
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_3?
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_3?
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell_3/strided_slice_3/stack?
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_3/strided_slice_3/stack_1?
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_3/stack_2?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_3?
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_7?
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_4?
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh_1?
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_lstm_3_layer_call_fn_78256
inputs_0
unknown:	2?	
	unknown_0:	?	
	unknown_1:
??	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_743522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?
?
&__inference_lstm_3_layer_call_fn_78245
inputs_0
unknown:	2?	
	unknown_0:	?	
	unknown_1:
??	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_740282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?
b
)__inference_dropout_4_layer_call_fn_78305

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_752242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_75053

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????P2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????P2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?

?
F__inference_embedding_9_layer_call_and_return_conditional_losses_74633

inputs*
embedding_lookup_74627:
??
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_74627Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/74627*-
_output_shapes
:???????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/74627*-
_output_shapes
:???????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_1_layer_call_fn_78426
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
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
GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_750462
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????@:?????????:Q M
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_75017

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
!Taghipour_lstm_3_while_body_73600>
:taghipour_lstm_3_while_taghipour_lstm_3_while_loop_counterD
@taghipour_lstm_3_while_taghipour_lstm_3_while_maximum_iterations&
"taghipour_lstm_3_while_placeholder(
$taghipour_lstm_3_while_placeholder_1(
$taghipour_lstm_3_while_placeholder_2(
$taghipour_lstm_3_while_placeholder_3=
9taghipour_lstm_3_while_taghipour_lstm_3_strided_slice_1_0y
utaghipour_lstm_3_while_tensorarrayv2read_tensorlistgetitem_taghipour_lstm_3_tensorarrayunstack_tensorlistfromtensor_0U
Btaghipour_lstm_3_while_lstm_cell_3_split_readvariableop_resource_0:	2?	S
Dtaghipour_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0:	?	P
<taghipour_lstm_3_while_lstm_cell_3_readvariableop_resource_0:
??	#
taghipour_lstm_3_while_identity%
!taghipour_lstm_3_while_identity_1%
!taghipour_lstm_3_while_identity_2%
!taghipour_lstm_3_while_identity_3%
!taghipour_lstm_3_while_identity_4%
!taghipour_lstm_3_while_identity_5;
7taghipour_lstm_3_while_taghipour_lstm_3_strided_slice_1w
staghipour_lstm_3_while_tensorarrayv2read_tensorlistgetitem_taghipour_lstm_3_tensorarrayunstack_tensorlistfromtensorS
@taghipour_lstm_3_while_lstm_cell_3_split_readvariableop_resource:	2?	Q
Btaghipour_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:	?	N
:taghipour_lstm_3_while_lstm_cell_3_readvariableop_resource:
??	??1Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp?3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_1?3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_2?3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_3?7Taghipour/lstm_3/while/lstm_cell_3/split/ReadVariableOp?9Taghipour/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp?
HTaghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2J
HTaghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape?
:Taghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemutaghipour_lstm_3_while_tensorarrayv2read_tensorlistgetitem_taghipour_lstm_3_tensorarrayunstack_tensorlistfromtensor_0"taghipour_lstm_3_while_placeholderQTaghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02<
:Taghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItem?
2Taghipour/lstm_3/while/lstm_cell_3/ones_like/ShapeShapeATaghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:24
2Taghipour/lstm_3/while/lstm_cell_3/ones_like/Shape?
2Taghipour/lstm_3/while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2Taghipour/lstm_3/while/lstm_cell_3/ones_like/Const?
,Taghipour/lstm_3/while/lstm_cell_3/ones_likeFill;Taghipour/lstm_3/while/lstm_cell_3/ones_like/Shape:output:0;Taghipour/lstm_3/while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22.
,Taghipour/lstm_3/while/lstm_cell_3/ones_like?
4Taghipour/lstm_3/while/lstm_cell_3/ones_like_1/ShapeShape$taghipour_lstm_3_while_placeholder_2*
T0*
_output_shapes
:26
4Taghipour/lstm_3/while/lstm_cell_3/ones_like_1/Shape?
4Taghipour/lstm_3/while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4Taghipour/lstm_3/while/lstm_cell_3/ones_like_1/Const?
.Taghipour/lstm_3/while/lstm_cell_3/ones_like_1Fill=Taghipour/lstm_3/while/lstm_cell_3/ones_like_1/Shape:output:0=Taghipour/lstm_3/while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????20
.Taghipour/lstm_3/while/lstm_cell_3/ones_like_1?
&Taghipour/lstm_3/while/lstm_cell_3/mulMulATaghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:05Taghipour/lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22(
&Taghipour/lstm_3/while/lstm_cell_3/mul?
(Taghipour/lstm_3/while/lstm_cell_3/mul_1MulATaghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:05Taghipour/lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22*
(Taghipour/lstm_3/while/lstm_cell_3/mul_1?
(Taghipour/lstm_3/while/lstm_cell_3/mul_2MulATaghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:05Taghipour/lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22*
(Taghipour/lstm_3/while/lstm_cell_3/mul_2?
(Taghipour/lstm_3/while/lstm_cell_3/mul_3MulATaghipour/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:05Taghipour/lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22*
(Taghipour/lstm_3/while/lstm_cell_3/mul_3?
2Taghipour/lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2Taghipour/lstm_3/while/lstm_cell_3/split/split_dim?
7Taghipour/lstm_3/while/lstm_cell_3/split/ReadVariableOpReadVariableOpBtaghipour_lstm_3_while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	2?	*
dtype029
7Taghipour/lstm_3/while/lstm_cell_3/split/ReadVariableOp?
(Taghipour/lstm_3/while/lstm_cell_3/splitSplit;Taghipour/lstm_3/while/lstm_cell_3/split/split_dim:output:0?Taghipour/lstm_3/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2*
(Taghipour/lstm_3/while/lstm_cell_3/split?
)Taghipour/lstm_3/while/lstm_cell_3/MatMulMatMul*Taghipour/lstm_3/while/lstm_cell_3/mul:z:01Taghipour/lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2+
)Taghipour/lstm_3/while/lstm_cell_3/MatMul?
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_1MatMul,Taghipour/lstm_3/while/lstm_cell_3/mul_1:z:01Taghipour/lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2-
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_1?
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_2MatMul,Taghipour/lstm_3/while/lstm_cell_3/mul_2:z:01Taghipour/lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2-
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_2?
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_3MatMul,Taghipour/lstm_3/while/lstm_cell_3/mul_3:z:01Taghipour/lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2-
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_3?
4Taghipour/lstm_3/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4Taghipour/lstm_3/while/lstm_cell_3/split_1/split_dim?
9Taghipour/lstm_3/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOpDtaghipour_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?	*
dtype02;
9Taghipour/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp?
*Taghipour/lstm_3/while/lstm_cell_3/split_1Split=Taghipour/lstm_3/while/lstm_cell_3/split_1/split_dim:output:0ATaghipour/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2,
*Taghipour/lstm_3/while/lstm_cell_3/split_1?
*Taghipour/lstm_3/while/lstm_cell_3/BiasAddBiasAdd3Taghipour/lstm_3/while/lstm_cell_3/MatMul:product:03Taghipour/lstm_3/while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2,
*Taghipour/lstm_3/while/lstm_cell_3/BiasAdd?
,Taghipour/lstm_3/while/lstm_cell_3/BiasAdd_1BiasAdd5Taghipour/lstm_3/while/lstm_cell_3/MatMul_1:product:03Taghipour/lstm_3/while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2.
,Taghipour/lstm_3/while/lstm_cell_3/BiasAdd_1?
,Taghipour/lstm_3/while/lstm_cell_3/BiasAdd_2BiasAdd5Taghipour/lstm_3/while/lstm_cell_3/MatMul_2:product:03Taghipour/lstm_3/while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2.
,Taghipour/lstm_3/while/lstm_cell_3/BiasAdd_2?
,Taghipour/lstm_3/while/lstm_cell_3/BiasAdd_3BiasAdd5Taghipour/lstm_3/while/lstm_cell_3/MatMul_3:product:03Taghipour/lstm_3/while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2.
,Taghipour/lstm_3/while/lstm_cell_3/BiasAdd_3?
(Taghipour/lstm_3/while/lstm_cell_3/mul_4Mul$taghipour_lstm_3_while_placeholder_27Taghipour/lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/mul_4?
(Taghipour/lstm_3/while/lstm_cell_3/mul_5Mul$taghipour_lstm_3_while_placeholder_27Taghipour/lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/mul_5?
(Taghipour/lstm_3/while/lstm_cell_3/mul_6Mul$taghipour_lstm_3_while_placeholder_27Taghipour/lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/mul_6?
(Taghipour/lstm_3/while/lstm_cell_3/mul_7Mul$taghipour_lstm_3_while_placeholder_27Taghipour/lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/mul_7?
1Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOpReadVariableOp<taghipour_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype023
1Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp?
6Taghipour/lstm_3/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        28
6Taghipour/lstm_3/while/lstm_cell_3/strided_slice/stack?
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2:
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice/stack_1?
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice/stack_2?
0Taghipour/lstm_3/while/lstm_cell_3/strided_sliceStridedSlice9Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp:value:0?Taghipour/lstm_3/while/lstm_cell_3/strided_slice/stack:output:0ATaghipour/lstm_3/while/lstm_cell_3/strided_slice/stack_1:output:0ATaghipour/lstm_3/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask22
0Taghipour/lstm_3/while/lstm_cell_3/strided_slice?
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_4MatMul,Taghipour/lstm_3/while/lstm_cell_3/mul_4:z:09Taghipour/lstm_3/while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2-
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_4?
&Taghipour/lstm_3/while/lstm_cell_3/addAddV23Taghipour/lstm_3/while/lstm_cell_3/BiasAdd:output:05Taghipour/lstm_3/while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2(
&Taghipour/lstm_3/while/lstm_cell_3/add?
*Taghipour/lstm_3/while/lstm_cell_3/SigmoidSigmoid*Taghipour/lstm_3/while/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2,
*Taghipour/lstm_3/while/lstm_cell_3/Sigmoid?
3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp<taghipour_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype025
3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_1?
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2:
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice_1/stack?
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2<
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_1/stack_1?
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_1/stack_2?
2Taghipour/lstm_3/while/lstm_cell_3/strided_slice_1StridedSlice;Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_1:value:0ATaghipour/lstm_3/while/lstm_cell_3/strided_slice_1/stack:output:0CTaghipour/lstm_3/while/lstm_cell_3/strided_slice_1/stack_1:output:0CTaghipour/lstm_3/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask24
2Taghipour/lstm_3/while/lstm_cell_3/strided_slice_1?
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_5MatMul,Taghipour/lstm_3/while/lstm_cell_3/mul_5:z:0;Taghipour/lstm_3/while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2-
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_5?
(Taghipour/lstm_3/while/lstm_cell_3/add_1AddV25Taghipour/lstm_3/while/lstm_cell_3/BiasAdd_1:output:05Taghipour/lstm_3/while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/add_1?
,Taghipour/lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid,Taghipour/lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2.
,Taghipour/lstm_3/while/lstm_cell_3/Sigmoid_1?
(Taghipour/lstm_3/while/lstm_cell_3/mul_8Mul0Taghipour/lstm_3/while/lstm_cell_3/Sigmoid_1:y:0$taghipour_lstm_3_while_placeholder_3*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/mul_8?
3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp<taghipour_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype025
3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_2?
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2:
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice_2/stack?
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2<
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_2/stack_1?
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_2/stack_2?
2Taghipour/lstm_3/while/lstm_cell_3/strided_slice_2StridedSlice;Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_2:value:0ATaghipour/lstm_3/while/lstm_cell_3/strided_slice_2/stack:output:0CTaghipour/lstm_3/while/lstm_cell_3/strided_slice_2/stack_1:output:0CTaghipour/lstm_3/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask24
2Taghipour/lstm_3/while/lstm_cell_3/strided_slice_2?
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_6MatMul,Taghipour/lstm_3/while/lstm_cell_3/mul_6:z:0;Taghipour/lstm_3/while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2-
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_6?
(Taghipour/lstm_3/while/lstm_cell_3/add_2AddV25Taghipour/lstm_3/while/lstm_cell_3/BiasAdd_2:output:05Taghipour/lstm_3/while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/add_2?
'Taghipour/lstm_3/while/lstm_cell_3/TanhTanh,Taghipour/lstm_3/while/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2)
'Taghipour/lstm_3/while/lstm_cell_3/Tanh?
(Taghipour/lstm_3/while/lstm_cell_3/mul_9Mul.Taghipour/lstm_3/while/lstm_cell_3/Sigmoid:y:0+Taghipour/lstm_3/while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/mul_9?
(Taghipour/lstm_3/while/lstm_cell_3/add_3AddV2,Taghipour/lstm_3/while/lstm_cell_3/mul_8:z:0,Taghipour/lstm_3/while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/add_3?
3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp<taghipour_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype025
3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_3?
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2:
8Taghipour/lstm_3/while/lstm_cell_3/strided_slice_3/stack?
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_3/stack_1?
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:Taghipour/lstm_3/while/lstm_cell_3/strided_slice_3/stack_2?
2Taghipour/lstm_3/while/lstm_cell_3/strided_slice_3StridedSlice;Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_3:value:0ATaghipour/lstm_3/while/lstm_cell_3/strided_slice_3/stack:output:0CTaghipour/lstm_3/while/lstm_cell_3/strided_slice_3/stack_1:output:0CTaghipour/lstm_3/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask24
2Taghipour/lstm_3/while/lstm_cell_3/strided_slice_3?
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_7MatMul,Taghipour/lstm_3/while/lstm_cell_3/mul_7:z:0;Taghipour/lstm_3/while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2-
+Taghipour/lstm_3/while/lstm_cell_3/MatMul_7?
(Taghipour/lstm_3/while/lstm_cell_3/add_4AddV25Taghipour/lstm_3/while/lstm_cell_3/BiasAdd_3:output:05Taghipour/lstm_3/while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/while/lstm_cell_3/add_4?
,Taghipour/lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid,Taghipour/lstm_3/while/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2.
,Taghipour/lstm_3/while/lstm_cell_3/Sigmoid_2?
)Taghipour/lstm_3/while/lstm_cell_3/Tanh_1Tanh,Taghipour/lstm_3/while/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2+
)Taghipour/lstm_3/while/lstm_cell_3/Tanh_1?
)Taghipour/lstm_3/while/lstm_cell_3/mul_10Mul0Taghipour/lstm_3/while/lstm_cell_3/Sigmoid_2:y:0-Taghipour/lstm_3/while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2+
)Taghipour/lstm_3/while/lstm_cell_3/mul_10?
;Taghipour/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$taghipour_lstm_3_while_placeholder_1"taghipour_lstm_3_while_placeholder-Taghipour/lstm_3/while/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype02=
;Taghipour/lstm_3/while/TensorArrayV2Write/TensorListSetItem~
Taghipour/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
Taghipour/lstm_3/while/add/y?
Taghipour/lstm_3/while/addAddV2"taghipour_lstm_3_while_placeholder%Taghipour/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
Taghipour/lstm_3/while/add?
Taghipour/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
Taghipour/lstm_3/while/add_1/y?
Taghipour/lstm_3/while/add_1AddV2:taghipour_lstm_3_while_taghipour_lstm_3_while_loop_counter'Taghipour/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
Taghipour/lstm_3/while/add_1?
Taghipour/lstm_3/while/IdentityIdentity Taghipour/lstm_3/while/add_1:z:0^Taghipour/lstm_3/while/NoOp*
T0*
_output_shapes
: 2!
Taghipour/lstm_3/while/Identity?
!Taghipour/lstm_3/while/Identity_1Identity@taghipour_lstm_3_while_taghipour_lstm_3_while_maximum_iterations^Taghipour/lstm_3/while/NoOp*
T0*
_output_shapes
: 2#
!Taghipour/lstm_3/while/Identity_1?
!Taghipour/lstm_3/while/Identity_2IdentityTaghipour/lstm_3/while/add:z:0^Taghipour/lstm_3/while/NoOp*
T0*
_output_shapes
: 2#
!Taghipour/lstm_3/while/Identity_2?
!Taghipour/lstm_3/while/Identity_3IdentityKTaghipour/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^Taghipour/lstm_3/while/NoOp*
T0*
_output_shapes
: 2#
!Taghipour/lstm_3/while/Identity_3?
!Taghipour/lstm_3/while/Identity_4Identity-Taghipour/lstm_3/while/lstm_cell_3/mul_10:z:0^Taghipour/lstm_3/while/NoOp*
T0*(
_output_shapes
:??????????2#
!Taghipour/lstm_3/while/Identity_4?
!Taghipour/lstm_3/while/Identity_5Identity,Taghipour/lstm_3/while/lstm_cell_3/add_3:z:0^Taghipour/lstm_3/while/NoOp*
T0*(
_output_shapes
:??????????2#
!Taghipour/lstm_3/while/Identity_5?
Taghipour/lstm_3/while/NoOpNoOp2^Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp4^Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_14^Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_24^Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_38^Taghipour/lstm_3/while/lstm_cell_3/split/ReadVariableOp:^Taghipour/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
Taghipour/lstm_3/while/NoOp"K
taghipour_lstm_3_while_identity(Taghipour/lstm_3/while/Identity:output:0"O
!taghipour_lstm_3_while_identity_1*Taghipour/lstm_3/while/Identity_1:output:0"O
!taghipour_lstm_3_while_identity_2*Taghipour/lstm_3/while/Identity_2:output:0"O
!taghipour_lstm_3_while_identity_3*Taghipour/lstm_3/while/Identity_3:output:0"O
!taghipour_lstm_3_while_identity_4*Taghipour/lstm_3/while/Identity_4:output:0"O
!taghipour_lstm_3_while_identity_5*Taghipour/lstm_3/while/Identity_5:output:0"z
:taghipour_lstm_3_while_lstm_cell_3_readvariableop_resource<taghipour_lstm_3_while_lstm_cell_3_readvariableop_resource_0"?
Btaghipour_lstm_3_while_lstm_cell_3_split_1_readvariableop_resourceDtaghipour_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0"?
@taghipour_lstm_3_while_lstm_cell_3_split_readvariableop_resourceBtaghipour_lstm_3_while_lstm_cell_3_split_readvariableop_resource_0"t
7taghipour_lstm_3_while_taghipour_lstm_3_strided_slice_19taghipour_lstm_3_while_taghipour_lstm_3_strided_slice_1_0"?
staghipour_lstm_3_while_tensorarrayv2read_tensorlistgetitem_taghipour_lstm_3_tensorarrayunstack_tensorlistfromtensorutaghipour_lstm_3_while_tensorarrayv2read_tensorlistgetitem_taghipour_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp1Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp2j
3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_13Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_12j
3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_23Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_22j
3Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_33Taghipour/lstm_3/while/lstm_cell_3/ReadVariableOp_32r
7Taghipour/lstm_3/while/lstm_cell_3/split/ReadVariableOp7Taghipour/lstm_3/while/lstm_cell_3/split/ReadVariableOp2v
9Taghipour/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp9Taghipour/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
F__inference_embedding_9_layer_call_and_return_conditional_losses_76924

inputs*
embedding_lookup_76918:
??
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_76918Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/76918*-
_output_shapes
:???????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/76918*-
_output_shapes
:???????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
P
/__inference_pos_x_maskedout_layer_call_fn_76950
x
mask

identity?
PartitionedCallPartitionedCallxmask*
Tin
2
*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_pos_x_maskedout_layer_call_and_return_conditional_losses_746522
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????:??????????:P L
-
_output_shapes
:???????????

_user_specified_namex:NJ
(
_output_shapes
:??????????

_user_specified_namemask
?
?
)__inference_Taghipour_layer_call_fn_75818
response
applied
unknown:
?? 
	unknown_0:?2
	unknown_1:2
	unknown_2:	2?	
	unknown_3:	?	
	unknown_4:
??	
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?@
	unknown_8:@
	unknown_9:

unknown_10:

unknown_11:P 

unknown_12: 

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallresponseappliedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_Taghipour_layer_call_and_return_conditional_losses_757452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
response:PL
'
_output_shapes
:?????????
!
_user_specified_name	applied
?
?
while_cond_77405
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_77405___redundant_placeholder03
/while_while_cond_77405___redundant_placeholder13
/while_while_cond_77405___redundant_placeholder23
/while_while_cond_77405___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
lstm_3_while_cond_76110*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_76110___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_76110___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_76110___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_76110___redundant_placeholder3
lstm_3_while_identity
?
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
lstm_3_while_body_76111*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0:	2?	I
:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0:	?	F
2lstm_3_while_lstm_cell_3_readvariableop_resource_0:
??	
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorI
6lstm_3_while_lstm_cell_3_split_readvariableop_resource:	2?	G
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:	?	D
0lstm_3_while_lstm_cell_3_readvariableop_resource:
??	??'lstm_3/while/lstm_cell_3/ReadVariableOp?)lstm_3/while/lstm_cell_3/ReadVariableOp_1?)lstm_3/while/lstm_cell_3/ReadVariableOp_2?)lstm_3/while/lstm_cell_3/ReadVariableOp_3?-lstm_3/while/lstm_cell_3/split/ReadVariableOp?/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp?
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItem?
(lstm_3/while/lstm_cell_3/ones_like/ShapeShape7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2*
(lstm_3/while/lstm_cell_3/ones_like/Shape?
(lstm_3/while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(lstm_3/while/lstm_cell_3/ones_like/Const?
"lstm_3/while/lstm_cell_3/ones_likeFill1lstm_3/while/lstm_cell_3/ones_like/Shape:output:01lstm_3/while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_3/while/lstm_cell_3/ones_like?
*lstm_3/while/lstm_cell_3/ones_like_1/ShapeShapelstm_3_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_3/while/lstm_cell_3/ones_like_1/Shape?
*lstm_3/while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*lstm_3/while/lstm_cell_3/ones_like_1/Const?
$lstm_3/while/lstm_cell_3/ones_like_1Fill3lstm_3/while/lstm_cell_3/ones_like_1/Shape:output:03lstm_3/while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$lstm_3/while/lstm_cell_3/ones_like_1?
lstm_3/while/lstm_cell_3/mulMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
lstm_3/while/lstm_cell_3/mul?
lstm_3/while/lstm_cell_3/mul_1Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22 
lstm_3/while/lstm_cell_3/mul_1?
lstm_3/while/lstm_cell_3/mul_2Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22 
lstm_3/while/lstm_cell_3/mul_2?
lstm_3/while/lstm_cell_3/mul_3Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22 
lstm_3/while/lstm_cell_3/mul_3?
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_3/split/split_dim?
-lstm_3/while/lstm_cell_3/split/ReadVariableOpReadVariableOp8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	2?	*
dtype02/
-lstm_3/while/lstm_cell_3/split/ReadVariableOp?
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:05lstm_3/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2 
lstm_3/while/lstm_cell_3/split?
lstm_3/while/lstm_cell_3/MatMulMatMul lstm_3/while/lstm_cell_3/mul:z:0'lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2!
lstm_3/while/lstm_cell_3/MatMul?
!lstm_3/while/lstm_cell_3/MatMul_1MatMul"lstm_3/while/lstm_cell_3/mul_1:z:0'lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_1?
!lstm_3/while/lstm_cell_3/MatMul_2MatMul"lstm_3/while/lstm_cell_3/mul_2:z:0'lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_2?
!lstm_3/while/lstm_cell_3/MatMul_3MatMul"lstm_3/while/lstm_cell_3/mul_3:z:0'lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_3?
*lstm_3/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_3/while/lstm_cell_3/split_1/split_dim?
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?	*
dtype021
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp?
 lstm_3/while/lstm_cell_3/split_1Split3lstm_3/while/lstm_cell_3/split_1/split_dim:output:07lstm_3/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2"
 lstm_3/while/lstm_cell_3/split_1?
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd)lstm_3/while/lstm_cell_3/MatMul:product:0)lstm_3/while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/while/lstm_cell_3/BiasAdd?
"lstm_3/while/lstm_cell_3/BiasAdd_1BiasAdd+lstm_3/while/lstm_cell_3/MatMul_1:product:0)lstm_3/while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/BiasAdd_1?
"lstm_3/while/lstm_cell_3/BiasAdd_2BiasAdd+lstm_3/while/lstm_cell_3/MatMul_2:product:0)lstm_3/while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/BiasAdd_2?
"lstm_3/while/lstm_cell_3/BiasAdd_3BiasAdd+lstm_3/while/lstm_cell_3/MatMul_3:product:0)lstm_3/while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/BiasAdd_3?
lstm_3/while/lstm_cell_3/mul_4Mullstm_3_while_placeholder_2-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_4?
lstm_3/while/lstm_cell_3/mul_5Mullstm_3_while_placeholder_2-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_5?
lstm_3/while/lstm_cell_3/mul_6Mullstm_3_while_placeholder_2-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_6?
lstm_3/while/lstm_cell_3/mul_7Mullstm_3_while_placeholder_2-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_7?
'lstm_3/while/lstm_cell_3/ReadVariableOpReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02)
'lstm_3/while/lstm_cell_3/ReadVariableOp?
,lstm_3/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_3/while/lstm_cell_3/strided_slice/stack?
.lstm_3/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  20
.lstm_3/while/lstm_cell_3/strided_slice/stack_1?
.lstm_3/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_3/while/lstm_cell_3/strided_slice/stack_2?
&lstm_3/while/lstm_cell_3/strided_sliceStridedSlice/lstm_3/while/lstm_cell_3/ReadVariableOp:value:05lstm_3/while/lstm_cell_3/strided_slice/stack:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_1:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&lstm_3/while/lstm_cell_3/strided_slice?
!lstm_3/while/lstm_cell_3/MatMul_4MatMul"lstm_3/while/lstm_cell_3/mul_4:z:0/lstm_3/while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_4?
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/BiasAdd:output:0+lstm_3/while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/while/lstm_cell_3/add?
 lstm_3/while/lstm_cell_3/SigmoidSigmoid lstm_3/while/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/while/lstm_cell_3/Sigmoid?
)lstm_3/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02+
)lstm_3/while/lstm_cell_3/ReadVariableOp_1?
.lstm_3/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  20
.lstm_3/while/lstm_cell_3/strided_slice_1/stack?
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  22
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_1?
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_2?
(lstm_3/while/lstm_cell_3/strided_slice_1StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_1:value:07lstm_3/while/lstm_cell_3/strided_slice_1/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(lstm_3/while/lstm_cell_3/strided_slice_1?
!lstm_3/while/lstm_cell_3/MatMul_5MatMul"lstm_3/while/lstm_cell_3/mul_5:z:01lstm_3/while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_5?
lstm_3/while/lstm_cell_3/add_1AddV2+lstm_3/while/lstm_cell_3/BiasAdd_1:output:0+lstm_3/while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/add_1?
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/Sigmoid_1?
lstm_3/while/lstm_cell_3/mul_8Mul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_8?
)lstm_3/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02+
)lstm_3/while/lstm_cell_3/ReadVariableOp_2?
.lstm_3/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  20
.lstm_3/while/lstm_cell_3/strided_slice_2/stack?
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  22
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_1?
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_2?
(lstm_3/while/lstm_cell_3/strided_slice_2StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_2:value:07lstm_3/while/lstm_cell_3/strided_slice_2/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(lstm_3/while/lstm_cell_3/strided_slice_2?
!lstm_3/while/lstm_cell_3/MatMul_6MatMul"lstm_3/while/lstm_cell_3/mul_6:z:01lstm_3/while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_6?
lstm_3/while/lstm_cell_3/add_2AddV2+lstm_3/while/lstm_cell_3/BiasAdd_2:output:0+lstm_3/while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/add_2?
lstm_3/while/lstm_cell_3/TanhTanh"lstm_3/while/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/while/lstm_cell_3/Tanh?
lstm_3/while/lstm_cell_3/mul_9Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0!lstm_3/while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_9?
lstm_3/while/lstm_cell_3/add_3AddV2"lstm_3/while/lstm_cell_3/mul_8:z:0"lstm_3/while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/add_3?
)lstm_3/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02+
)lstm_3/while/lstm_cell_3/ReadVariableOp_3?
.lstm_3/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  20
.lstm_3/while/lstm_cell_3/strided_slice_3/stack?
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_1?
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_2?
(lstm_3/while/lstm_cell_3/strided_slice_3StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_3:value:07lstm_3/while/lstm_cell_3/strided_slice_3/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(lstm_3/while/lstm_cell_3/strided_slice_3?
!lstm_3/while/lstm_cell_3/MatMul_7MatMul"lstm_3/while/lstm_cell_3/mul_7:z:01lstm_3/while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_7?
lstm_3/while/lstm_cell_3/add_4AddV2+lstm_3/while/lstm_cell_3/BiasAdd_3:output:0+lstm_3/while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/add_4?
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid"lstm_3/while/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/Sigmoid_2?
lstm_3/while/lstm_cell_3/Tanh_1Tanh"lstm_3/while/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2!
lstm_3/while/lstm_cell_3/Tanh_1?
lstm_3/while/lstm_cell_3/mul_10Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0#lstm_3/while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2!
lstm_3/while/lstm_cell_3/mul_10?
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder#lstm_3/while/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/y?
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/y?
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1?
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity?
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1?
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2?
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3?
lstm_3/while/Identity_4Identity#lstm_3/while/lstm_cell_3/mul_10:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_3/while/Identity_4?
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_3:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_3/while/Identity_5?
lstm_3/while/NoOpNoOp(^lstm_3/while/lstm_cell_3/ReadVariableOp*^lstm_3/while/lstm_cell_3/ReadVariableOp_1*^lstm_3/while/lstm_cell_3/ReadVariableOp_2*^lstm_3/while/lstm_cell_3/ReadVariableOp_3.^lstm_3/while/lstm_cell_3/split/ReadVariableOp0^lstm_3/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_3/while/NoOp"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"f
0lstm_3_while_lstm_cell_3_readvariableop_resource2lstm_3_while_lstm_cell_3_readvariableop_resource_0"v
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0"r
6lstm_3_while_lstm_cell_3_split_readvariableop_resource8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0"?
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2R
'lstm_3/while/lstm_cell_3/ReadVariableOp'lstm_3/while/lstm_cell_3/ReadVariableOp2V
)lstm_3/while/lstm_cell_3/ReadVariableOp_1)lstm_3/while/lstm_cell_3/ReadVariableOp_12V
)lstm_3/while/lstm_cell_3/ReadVariableOp_2)lstm_3/while/lstm_cell_3/ReadVariableOp_22V
)lstm_3/while/lstm_cell_3/ReadVariableOp_3)lstm_3/while/lstm_cell_3/ReadVariableOp_32^
-lstm_3/while/lstm_cell_3/split/ReadVariableOp-lstm_3/while/lstm_cell_3/split/ReadVariableOp2b
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_dense_7_layer_call_and_return_conditional_losses_75065

inputs0
matmul_readvariableop_resource:P -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
C__inference_conv1d_5_layer_call_and_return_conditional_losses_76965

inputsB
+conv1d_expanddims_1_readvariableop_resource:?2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?2*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?22
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????2*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????2*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????22	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????22

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_77604
inputs_0<
)lstm_cell_3_split_readvariableop_resource:	2?	:
+lstm_cell_3_split_1_readvariableop_resource:	?	7
#lstm_cell_3_readvariableop_resource:
??	
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like/Shape
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like/Const?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/ones_like{
lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout/Const?
lstm_cell_3/dropout/MulMullstm_cell_3/ones_like:output:0"lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout/Mul?
lstm_cell_3/dropout/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout/Shape?
0lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2ï822
0lstm_cell_3/dropout/random_uniform/RandomUniform?
"lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2$
"lstm_cell_3/dropout/GreaterEqual/y?
 lstm_cell_3/dropout/GreaterEqualGreaterEqual9lstm_cell_3/dropout/random_uniform/RandomUniform:output:0+lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_cell_3/dropout/GreaterEqual?
lstm_cell_3/dropout/CastCast$lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout/Cast?
lstm_cell_3/dropout/Mul_1Mullstm_cell_3/dropout/Mul:z:0lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout/Mul_1
lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_1/Const?
lstm_cell_3/dropout_1/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_1/Mul?
lstm_cell_3/dropout_1/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_1/Shape?
2lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??c24
2lstm_cell_3/dropout_1/random_uniform/RandomUniform?
$lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_1/GreaterEqual/y?
"lstm_cell_3/dropout_1/GreaterEqualGreaterEqual;lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_cell_3/dropout_1/GreaterEqual?
lstm_cell_3/dropout_1/CastCast&lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout_1/Cast?
lstm_cell_3/dropout_1/Mul_1Mullstm_cell_3/dropout_1/Mul:z:0lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_1/Mul_1
lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_2/Const?
lstm_cell_3/dropout_2/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_2/Mul?
lstm_cell_3/dropout_2/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_2/Shape?
2lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_2/random_uniform/RandomUniform?
$lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_2/GreaterEqual/y?
"lstm_cell_3/dropout_2/GreaterEqualGreaterEqual;lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_cell_3/dropout_2/GreaterEqual?
lstm_cell_3/dropout_2/CastCast&lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout_2/Cast?
lstm_cell_3/dropout_2/Mul_1Mullstm_cell_3/dropout_2/Mul:z:0lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_2/Mul_1
lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_3/Const?
lstm_cell_3/dropout_3/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_3/Mul?
lstm_cell_3/dropout_3/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_3/Shape?
2lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2魋24
2lstm_cell_3/dropout_3/random_uniform/RandomUniform?
$lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_3/GreaterEqual/y?
"lstm_cell_3/dropout_3/GreaterEqualGreaterEqual;lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_cell_3/dropout_3/GreaterEqual?
lstm_cell_3/dropout_3/CastCast&lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout_3/Cast?
lstm_cell_3/dropout_3/Mul_1Mullstm_cell_3/dropout_3/Mul:z:0lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_3/Mul_1|
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like_1/Shape?
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like_1/Const?
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/ones_like_1
lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_4/Const?
lstm_cell_3/dropout_4/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_4/Mul?
lstm_cell_3/dropout_4/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_4/Shape?
2lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_4/random_uniform/RandomUniform?
$lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_4/GreaterEqual/y?
"lstm_cell_3/dropout_4/GreaterEqualGreaterEqual;lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_4/GreaterEqual?
lstm_cell_3/dropout_4/CastCast&lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_4/Cast?
lstm_cell_3/dropout_4/Mul_1Mullstm_cell_3/dropout_4/Mul:z:0lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_4/Mul_1
lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_5/Const?
lstm_cell_3/dropout_5/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_5/Mul?
lstm_cell_3/dropout_5/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_5/Shape?
2lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??L24
2lstm_cell_3/dropout_5/random_uniform/RandomUniform?
$lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_5/GreaterEqual/y?
"lstm_cell_3/dropout_5/GreaterEqualGreaterEqual;lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_5/GreaterEqual?
lstm_cell_3/dropout_5/CastCast&lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_5/Cast?
lstm_cell_3/dropout_5/Mul_1Mullstm_cell_3/dropout_5/Mul:z:0lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_5/Mul_1
lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_6/Const?
lstm_cell_3/dropout_6/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_6/Mul?
lstm_cell_3/dropout_6/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_6/Shape?
2lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??X24
2lstm_cell_3/dropout_6/random_uniform/RandomUniform?
$lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_6/GreaterEqual/y?
"lstm_cell_3/dropout_6/GreaterEqualGreaterEqual;lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_6/GreaterEqual?
lstm_cell_3/dropout_6/CastCast&lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_6/Cast?
lstm_cell_3/dropout_6/Mul_1Mullstm_cell_3/dropout_6/Mul:z:0lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_6/Mul_1
lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_7/Const?
lstm_cell_3/dropout_7/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_7/Mul?
lstm_cell_3/dropout_7/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_7/Shape?
2lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2톜24
2lstm_cell_3/dropout_7/random_uniform/RandomUniform?
$lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_7/GreaterEqual/y?
"lstm_cell_3/dropout_7/GreaterEqualGreaterEqual;lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_7/GreaterEqual?
lstm_cell_3/dropout_7/CastCast&lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_7/Cast?
lstm_cell_3/dropout_7/Mul_1Mullstm_cell_3/dropout_7/Mul:z:0lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_7/Mul_1?
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul?
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_1?
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_2?
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_3|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	2?	*
dtype02"
 lstm_cell_3/split/ReadVariableOp?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
lstm_cell_3/split?
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_2?
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_3?
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_3/split_1/split_dim?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02$
"lstm_cell_3/split_1/ReadVariableOp?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_3/split_1?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd?
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_1?
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_2?
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_3?
lstm_cell_3/mul_4Mulzeros:output:0lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_4?
lstm_cell_3/mul_5Mulzeros:output:0lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_5?
lstm_cell_3/mul_6Mulzeros:output:0lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_6?
lstm_cell_3/mul_7Mulzeros:output:0lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_7?
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp?
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_3/strided_slice/stack?
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice/stack_1?
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_3/strided_slice/stack_2?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice?
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_4?
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add}
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid?
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_1?
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice_1/stack?
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2%
#lstm_cell_3/strided_slice_1/stack_1?
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_1/stack_2?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_1?
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_5?
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_8?
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_2?
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell_3/strided_slice_2/stack?
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#lstm_cell_3/strided_slice_2/stack_1?
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_2/stack_2?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_2?
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_6?
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_2v
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh?
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_9?
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_3?
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_3?
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell_3/strided_slice_3/stack?
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_3/strided_slice_3/stack_1?
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_3/stack_2?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_3?
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_7?
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_4?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_2z
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh_1?
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_77406*
condR
while_cond_77405*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????2: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?M
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_78574

inputs
states_0
states_10
split_readvariableop_resource:	2?	.
split_1_readvariableop_resource:	?	+
readvariableop_resource:
??	
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????22
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????22
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????22
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????22
mul_3d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	2?	*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3h
mul_4Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_4h
mul_5Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_5h
mul_6Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_6h
mul_7Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10f
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????2:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
lstm_3_while_body_76540*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0:	2?	I
:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0:	?	F
2lstm_3_while_lstm_cell_3_readvariableop_resource_0:
??	
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorI
6lstm_3_while_lstm_cell_3_split_readvariableop_resource:	2?	G
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:	?	D
0lstm_3_while_lstm_cell_3_readvariableop_resource:
??	??'lstm_3/while/lstm_cell_3/ReadVariableOp?)lstm_3/while/lstm_cell_3/ReadVariableOp_1?)lstm_3/while/lstm_cell_3/ReadVariableOp_2?)lstm_3/while/lstm_cell_3/ReadVariableOp_3?-lstm_3/while/lstm_cell_3/split/ReadVariableOp?/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp?
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItem?
(lstm_3/while/lstm_cell_3/ones_like/ShapeShape7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2*
(lstm_3/while/lstm_cell_3/ones_like/Shape?
(lstm_3/while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(lstm_3/while/lstm_cell_3/ones_like/Const?
"lstm_3/while/lstm_cell_3/ones_likeFill1lstm_3/while/lstm_cell_3/ones_like/Shape:output:01lstm_3/while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_3/while/lstm_cell_3/ones_like?
&lstm_3/while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2(
&lstm_3/while/lstm_cell_3/dropout/Const?
$lstm_3/while/lstm_cell_3/dropout/MulMul+lstm_3/while/lstm_cell_3/ones_like:output:0/lstm_3/while/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22&
$lstm_3/while/lstm_cell_3/dropout/Mul?
&lstm_3/while/lstm_cell_3/dropout/ShapeShape+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_3/while/lstm_cell_3/dropout/Shape?
=lstm_3/while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform/lstm_3/while/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2?
=lstm_3/while/lstm_cell_3/dropout/random_uniform/RandomUniform?
/lstm_3/while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>21
/lstm_3/while/lstm_cell_3/dropout/GreaterEqual/y?
-lstm_3/while/lstm_cell_3/dropout/GreaterEqualGreaterEqualFlstm_3/while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:08lstm_3/while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22/
-lstm_3/while/lstm_cell_3/dropout/GreaterEqual?
%lstm_3/while/lstm_cell_3/dropout/CastCast1lstm_3/while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22'
%lstm_3/while/lstm_cell_3/dropout/Cast?
&lstm_3/while/lstm_cell_3/dropout/Mul_1Mul(lstm_3/while/lstm_cell_3/dropout/Mul:z:0)lstm_3/while/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22(
&lstm_3/while/lstm_cell_3/dropout/Mul_1?
(lstm_3/while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2*
(lstm_3/while/lstm_cell_3/dropout_1/Const?
&lstm_3/while/lstm_cell_3/dropout_1/MulMul+lstm_3/while/lstm_cell_3/ones_like:output:01lstm_3/while/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22(
&lstm_3/while/lstm_cell_3/dropout_1/Mul?
(lstm_3/while/lstm_cell_3/dropout_1/ShapeShape+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_3/while/lstm_cell_3/dropout_1/Shape?
?lstm_3/while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2A
?lstm_3/while/lstm_cell_3/dropout_1/random_uniform/RandomUniform?
1lstm_3/while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1lstm_3/while/lstm_cell_3/dropout_1/GreaterEqual/y?
/lstm_3/while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????221
/lstm_3/while/lstm_cell_3/dropout_1/GreaterEqual?
'lstm_3/while/lstm_cell_3/dropout_1/CastCast3lstm_3/while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22)
'lstm_3/while/lstm_cell_3/dropout_1/Cast?
(lstm_3/while/lstm_cell_3/dropout_1/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_1/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22*
(lstm_3/while/lstm_cell_3/dropout_1/Mul_1?
(lstm_3/while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2*
(lstm_3/while/lstm_cell_3/dropout_2/Const?
&lstm_3/while/lstm_cell_3/dropout_2/MulMul+lstm_3/while/lstm_cell_3/ones_like:output:01lstm_3/while/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22(
&lstm_3/while/lstm_cell_3/dropout_2/Mul?
(lstm_3/while/lstm_cell_3/dropout_2/ShapeShape+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_3/while/lstm_cell_3/dropout_2/Shape?
?lstm_3/while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??X2A
?lstm_3/while/lstm_cell_3/dropout_2/random_uniform/RandomUniform?
1lstm_3/while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1lstm_3/while/lstm_cell_3/dropout_2/GreaterEqual/y?
/lstm_3/while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????221
/lstm_3/while/lstm_cell_3/dropout_2/GreaterEqual?
'lstm_3/while/lstm_cell_3/dropout_2/CastCast3lstm_3/while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22)
'lstm_3/while/lstm_cell_3/dropout_2/Cast?
(lstm_3/while/lstm_cell_3/dropout_2/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_2/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22*
(lstm_3/while/lstm_cell_3/dropout_2/Mul_1?
(lstm_3/while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2*
(lstm_3/while/lstm_cell_3/dropout_3/Const?
&lstm_3/while/lstm_cell_3/dropout_3/MulMul+lstm_3/while/lstm_cell_3/ones_like:output:01lstm_3/while/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22(
&lstm_3/while/lstm_cell_3/dropout_3/Mul?
(lstm_3/while/lstm_cell_3/dropout_3/ShapeShape+lstm_3/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_3/while/lstm_cell_3/dropout_3/Shape?
?lstm_3/while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??k2A
?lstm_3/while/lstm_cell_3/dropout_3/random_uniform/RandomUniform?
1lstm_3/while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1lstm_3/while/lstm_cell_3/dropout_3/GreaterEqual/y?
/lstm_3/while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????221
/lstm_3/while/lstm_cell_3/dropout_3/GreaterEqual?
'lstm_3/while/lstm_cell_3/dropout_3/CastCast3lstm_3/while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22)
'lstm_3/while/lstm_cell_3/dropout_3/Cast?
(lstm_3/while/lstm_cell_3/dropout_3/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_3/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22*
(lstm_3/while/lstm_cell_3/dropout_3/Mul_1?
*lstm_3/while/lstm_cell_3/ones_like_1/ShapeShapelstm_3_while_placeholder_2*
T0*
_output_shapes
:2,
*lstm_3/while/lstm_cell_3/ones_like_1/Shape?
*lstm_3/while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*lstm_3/while/lstm_cell_3/ones_like_1/Const?
$lstm_3/while/lstm_cell_3/ones_like_1Fill3lstm_3/while/lstm_cell_3/ones_like_1/Shape:output:03lstm_3/while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$lstm_3/while/lstm_cell_3/ones_like_1?
(lstm_3/while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2*
(lstm_3/while/lstm_cell_3/dropout_4/Const?
&lstm_3/while/lstm_cell_3/dropout_4/MulMul-lstm_3/while/lstm_cell_3/ones_like_1:output:01lstm_3/while/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2(
&lstm_3/while/lstm_cell_3/dropout_4/Mul?
(lstm_3/while/lstm_cell_3/dropout_4/ShapeShape-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2*
(lstm_3/while/lstm_cell_3/dropout_4/Shape?
?lstm_3/while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2A
?lstm_3/while/lstm_cell_3/dropout_4/random_uniform/RandomUniform?
1lstm_3/while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1lstm_3/while/lstm_cell_3/dropout_4/GreaterEqual/y?
/lstm_3/while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????21
/lstm_3/while/lstm_cell_3/dropout_4/GreaterEqual?
'lstm_3/while/lstm_cell_3/dropout_4/CastCast3lstm_3/while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2)
'lstm_3/while/lstm_cell_3/dropout_4/Cast?
(lstm_3/while/lstm_cell_3/dropout_4/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_4/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2*
(lstm_3/while/lstm_cell_3/dropout_4/Mul_1?
(lstm_3/while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2*
(lstm_3/while/lstm_cell_3/dropout_5/Const?
&lstm_3/while/lstm_cell_3/dropout_5/MulMul-lstm_3/while/lstm_cell_3/ones_like_1:output:01lstm_3/while/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2(
&lstm_3/while/lstm_cell_3/dropout_5/Mul?
(lstm_3/while/lstm_cell_3/dropout_5/ShapeShape-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2*
(lstm_3/while/lstm_cell_3/dropout_5/Shape?
?lstm_3/while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2A
?lstm_3/while/lstm_cell_3/dropout_5/random_uniform/RandomUniform?
1lstm_3/while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1lstm_3/while/lstm_cell_3/dropout_5/GreaterEqual/y?
/lstm_3/while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????21
/lstm_3/while/lstm_cell_3/dropout_5/GreaterEqual?
'lstm_3/while/lstm_cell_3/dropout_5/CastCast3lstm_3/while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2)
'lstm_3/while/lstm_cell_3/dropout_5/Cast?
(lstm_3/while/lstm_cell_3/dropout_5/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_5/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2*
(lstm_3/while/lstm_cell_3/dropout_5/Mul_1?
(lstm_3/while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2*
(lstm_3/while/lstm_cell_3/dropout_6/Const?
&lstm_3/while/lstm_cell_3/dropout_6/MulMul-lstm_3/while/lstm_cell_3/ones_like_1:output:01lstm_3/while/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2(
&lstm_3/while/lstm_cell_3/dropout_6/Mul?
(lstm_3/while/lstm_cell_3/dropout_6/ShapeShape-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2*
(lstm_3/while/lstm_cell_3/dropout_6/Shape?
?lstm_3/while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??D2A
?lstm_3/while/lstm_cell_3/dropout_6/random_uniform/RandomUniform?
1lstm_3/while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1lstm_3/while/lstm_cell_3/dropout_6/GreaterEqual/y?
/lstm_3/while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????21
/lstm_3/while/lstm_cell_3/dropout_6/GreaterEqual?
'lstm_3/while/lstm_cell_3/dropout_6/CastCast3lstm_3/while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2)
'lstm_3/while/lstm_cell_3/dropout_6/Cast?
(lstm_3/while/lstm_cell_3/dropout_6/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_6/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2*
(lstm_3/while/lstm_cell_3/dropout_6/Mul_1?
(lstm_3/while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2*
(lstm_3/while/lstm_cell_3/dropout_7/Const?
&lstm_3/while/lstm_cell_3/dropout_7/MulMul-lstm_3/while/lstm_cell_3/ones_like_1:output:01lstm_3/while/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2(
&lstm_3/while/lstm_cell_3/dropout_7/Mul?
(lstm_3/while/lstm_cell_3/dropout_7/ShapeShape-lstm_3/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2*
(lstm_3/while/lstm_cell_3/dropout_7/Shape?
?lstm_3/while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform1lstm_3/while/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ѹ2A
?lstm_3/while/lstm_cell_3/dropout_7/random_uniform/RandomUniform?
1lstm_3/while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>23
1lstm_3/while/lstm_cell_3/dropout_7/GreaterEqual/y?
/lstm_3/while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualHlstm_3/while/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0:lstm_3/while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????21
/lstm_3/while/lstm_cell_3/dropout_7/GreaterEqual?
'lstm_3/while/lstm_cell_3/dropout_7/CastCast3lstm_3/while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2)
'lstm_3/while/lstm_cell_3/dropout_7/Cast?
(lstm_3/while/lstm_cell_3/dropout_7/Mul_1Mul*lstm_3/while/lstm_cell_3/dropout_7/Mul:z:0+lstm_3/while/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2*
(lstm_3/while/lstm_cell_3/dropout_7/Mul_1?
lstm_3/while/lstm_cell_3/mulMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm_3/while/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_3/while/lstm_cell_3/mul?
lstm_3/while/lstm_cell_3/mul_1Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_3/while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22 
lstm_3/while/lstm_cell_3/mul_1?
lstm_3/while/lstm_cell_3/mul_2Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_3/while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22 
lstm_3/while/lstm_cell_3/mul_2?
lstm_3/while/lstm_cell_3/mul_3Mul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_3/while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22 
lstm_3/while/lstm_cell_3/mul_3?
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_3/split/split_dim?
-lstm_3/while/lstm_cell_3/split/ReadVariableOpReadVariableOp8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	2?	*
dtype02/
-lstm_3/while/lstm_cell_3/split/ReadVariableOp?
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:05lstm_3/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2 
lstm_3/while/lstm_cell_3/split?
lstm_3/while/lstm_cell_3/MatMulMatMul lstm_3/while/lstm_cell_3/mul:z:0'lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2!
lstm_3/while/lstm_cell_3/MatMul?
!lstm_3/while/lstm_cell_3/MatMul_1MatMul"lstm_3/while/lstm_cell_3/mul_1:z:0'lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_1?
!lstm_3/while/lstm_cell_3/MatMul_2MatMul"lstm_3/while/lstm_cell_3/mul_2:z:0'lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_2?
!lstm_3/while/lstm_cell_3/MatMul_3MatMul"lstm_3/while/lstm_cell_3/mul_3:z:0'lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_3?
*lstm_3/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_3/while/lstm_cell_3/split_1/split_dim?
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?	*
dtype021
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp?
 lstm_3/while/lstm_cell_3/split_1Split3lstm_3/while/lstm_cell_3/split_1/split_dim:output:07lstm_3/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2"
 lstm_3/while/lstm_cell_3/split_1?
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd)lstm_3/while/lstm_cell_3/MatMul:product:0)lstm_3/while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/while/lstm_cell_3/BiasAdd?
"lstm_3/while/lstm_cell_3/BiasAdd_1BiasAdd+lstm_3/while/lstm_cell_3/MatMul_1:product:0)lstm_3/while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/BiasAdd_1?
"lstm_3/while/lstm_cell_3/BiasAdd_2BiasAdd+lstm_3/while/lstm_cell_3/MatMul_2:product:0)lstm_3/while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/BiasAdd_2?
"lstm_3/while/lstm_cell_3/BiasAdd_3BiasAdd+lstm_3/while/lstm_cell_3/MatMul_3:product:0)lstm_3/while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/BiasAdd_3?
lstm_3/while/lstm_cell_3/mul_4Mullstm_3_while_placeholder_2,lstm_3/while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_4?
lstm_3/while/lstm_cell_3/mul_5Mullstm_3_while_placeholder_2,lstm_3/while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_5?
lstm_3/while/lstm_cell_3/mul_6Mullstm_3_while_placeholder_2,lstm_3/while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_6?
lstm_3/while/lstm_cell_3/mul_7Mullstm_3_while_placeholder_2,lstm_3/while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_7?
'lstm_3/while/lstm_cell_3/ReadVariableOpReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02)
'lstm_3/while/lstm_cell_3/ReadVariableOp?
,lstm_3/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_3/while/lstm_cell_3/strided_slice/stack?
.lstm_3/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  20
.lstm_3/while/lstm_cell_3/strided_slice/stack_1?
.lstm_3/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_3/while/lstm_cell_3/strided_slice/stack_2?
&lstm_3/while/lstm_cell_3/strided_sliceStridedSlice/lstm_3/while/lstm_cell_3/ReadVariableOp:value:05lstm_3/while/lstm_cell_3/strided_slice/stack:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_1:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2(
&lstm_3/while/lstm_cell_3/strided_slice?
!lstm_3/while/lstm_cell_3/MatMul_4MatMul"lstm_3/while/lstm_cell_3/mul_4:z:0/lstm_3/while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_4?
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/BiasAdd:output:0+lstm_3/while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/while/lstm_cell_3/add?
 lstm_3/while/lstm_cell_3/SigmoidSigmoid lstm_3/while/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/while/lstm_cell_3/Sigmoid?
)lstm_3/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02+
)lstm_3/while/lstm_cell_3/ReadVariableOp_1?
.lstm_3/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  20
.lstm_3/while/lstm_cell_3/strided_slice_1/stack?
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  22
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_1?
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_2?
(lstm_3/while/lstm_cell_3/strided_slice_1StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_1:value:07lstm_3/while/lstm_cell_3/strided_slice_1/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(lstm_3/while/lstm_cell_3/strided_slice_1?
!lstm_3/while/lstm_cell_3/MatMul_5MatMul"lstm_3/while/lstm_cell_3/mul_5:z:01lstm_3/while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_5?
lstm_3/while/lstm_cell_3/add_1AddV2+lstm_3/while/lstm_cell_3/BiasAdd_1:output:0+lstm_3/while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/add_1?
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/Sigmoid_1?
lstm_3/while/lstm_cell_3/mul_8Mul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_8?
)lstm_3/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02+
)lstm_3/while/lstm_cell_3/ReadVariableOp_2?
.lstm_3/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  20
.lstm_3/while/lstm_cell_3/strided_slice_2/stack?
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  22
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_1?
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_2?
(lstm_3/while/lstm_cell_3/strided_slice_2StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_2:value:07lstm_3/while/lstm_cell_3/strided_slice_2/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(lstm_3/while/lstm_cell_3/strided_slice_2?
!lstm_3/while/lstm_cell_3/MatMul_6MatMul"lstm_3/while/lstm_cell_3/mul_6:z:01lstm_3/while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_6?
lstm_3/while/lstm_cell_3/add_2AddV2+lstm_3/while/lstm_cell_3/BiasAdd_2:output:0+lstm_3/while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/add_2?
lstm_3/while/lstm_cell_3/TanhTanh"lstm_3/while/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/while/lstm_cell_3/Tanh?
lstm_3/while/lstm_cell_3/mul_9Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0!lstm_3/while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/mul_9?
lstm_3/while/lstm_cell_3/add_3AddV2"lstm_3/while/lstm_cell_3/mul_8:z:0"lstm_3/while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/add_3?
)lstm_3/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02+
)lstm_3/while/lstm_cell_3/ReadVariableOp_3?
.lstm_3/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  20
.lstm_3/while/lstm_cell_3/strided_slice_3/stack?
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_1?
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_2?
(lstm_3/while/lstm_cell_3/strided_slice_3StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_3:value:07lstm_3/while/lstm_cell_3/strided_slice_3/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(lstm_3/while/lstm_cell_3/strided_slice_3?
!lstm_3/while/lstm_cell_3/MatMul_7MatMul"lstm_3/while/lstm_cell_3/mul_7:z:01lstm_3/while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_7?
lstm_3/while/lstm_cell_3/add_4AddV2+lstm_3/while/lstm_cell_3/BiasAdd_3:output:0+lstm_3/while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2 
lstm_3/while/lstm_cell_3/add_4?
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid"lstm_3/while/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2$
"lstm_3/while/lstm_cell_3/Sigmoid_2?
lstm_3/while/lstm_cell_3/Tanh_1Tanh"lstm_3/while/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2!
lstm_3/while/lstm_cell_3/Tanh_1?
lstm_3/while/lstm_cell_3/mul_10Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0#lstm_3/while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2!
lstm_3/while/lstm_cell_3/mul_10?
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder#lstm_3/while/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/y?
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/y?
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1?
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity?
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1?
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2?
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3?
lstm_3/while/Identity_4Identity#lstm_3/while/lstm_cell_3/mul_10:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_3/while/Identity_4?
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_3:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_3/while/Identity_5?
lstm_3/while/NoOpNoOp(^lstm_3/while/lstm_cell_3/ReadVariableOp*^lstm_3/while/lstm_cell_3/ReadVariableOp_1*^lstm_3/while/lstm_cell_3/ReadVariableOp_2*^lstm_3/while/lstm_cell_3/ReadVariableOp_3.^lstm_3/while/lstm_cell_3/split/ReadVariableOp0^lstm_3/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_3/while/NoOp"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"f
0lstm_3_while_lstm_cell_3_readvariableop_resource2lstm_3_while_lstm_cell_3_readvariableop_resource_0"v
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0"r
6lstm_3_while_lstm_cell_3_split_readvariableop_resource8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0"?
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2R
'lstm_3/while/lstm_cell_3/ReadVariableOp'lstm_3/while/lstm_cell_3/ReadVariableOp2V
)lstm_3/while/lstm_cell_3/ReadVariableOp_1)lstm_3/while/lstm_cell_3/ReadVariableOp_12V
)lstm_3/while/lstm_cell_3/ReadVariableOp_2)lstm_3/while/lstm_cell_3/ReadVariableOp_22V
)lstm_3/while/lstm_cell_3/ReadVariableOp_3)lstm_3/while/lstm_cell_3/ReadVariableOp_32^
-lstm_3/while/lstm_cell_3/split/ReadVariableOp-lstm_3/while/lstm_cell_3/split/ReadVariableOp2b
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?L
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_73945

inputs

states
states_10
split_readvariableop_resource:	2?	.
split_1_readvariableop_resource:	?	+
readvariableop_resource:
??	
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpX
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
ones_like_1_
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????22
mulc
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????22
mul_1c
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????22
mul_2c
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????22
mul_3d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	2?	*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3f
mul_4Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_4f
mul_5Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_5f
mul_6Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_6f
mul_7Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:??????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:??????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:??????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??	*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:??????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
mul_10f
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????2:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
+__inference_embedding_9_layer_call_fn_76931

inputs
unknown:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_9_layer_call_and_return_conditional_losses_746332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_78295

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consty
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????2
dropout/Mul_1k
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_lstm_cell_3_layer_call_fn_78737

inputs
states_0
states_1
unknown:	2?	
	unknown_0:	?	
	unknown_1:
??	
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_739452
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????2:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?>
?
D__inference_Taghipour_layer_call_and_return_conditional_losses_75920
response
applied%
embedding_9_75873:
??%
conv1d_5_75879:?2
conv1d_5_75881:2
lstm_3_75884:	2?	
lstm_3_75886:	?	 
lstm_3_75888:
??	%
attention_3_75892:
?? 
attention_3_75894:	? 
dense_5_75897:	?@
dense_5_75899:@
dense_6_75902:
dense_6_75904:
dense_7_75909:P 
dense_7_75911: 
score_75914: 
score_75916:
identity??#attention_3/StatefulPartitionedCall? conv1d_5/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?score/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallresponseembedding_9_75873*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embedding_9_layer_call_and_return_conditional_losses_746332%
#embedding_9/StatefulPartitionedCallu
embedding_9/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_9/NotEqual/y?
embedding_9/NotEqualNotEqualresponseembedding_9/NotEqual/y:output:0*
T0*(
_output_shapes
:??????????2
embedding_9/NotEqual?
pos_x_maskedout/PartitionedCallPartitionedCall,embedding_9/StatefulPartitionedCall:output:0embedding_9/NotEqual:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_pos_x_maskedout_layer_call_and_return_conditional_losses_746522!
pos_x_maskedout/PartitionedCall?
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall(pos_x_maskedout/PartitionedCall:output:0conv1d_5_75879conv1d_5_75881*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_746692"
 conv1d_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0lstm_3_75884lstm_3_75886lstm_3_75888*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_756192 
lstm_3/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_752242#
!dropout_4/StatefulPartitionedCall?
#attention_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0attention_3_75892attention_3_75894*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_attention_3_layer_call_and_return_conditional_losses_750012%
#attention_3/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall,attention_3/StatefulPartitionedCall:output:0dense_5_75897dense_5_75899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_750172!
dense_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallapplieddense_6_75902dense_6_75904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_750332!
dense_6/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_750462
concatenate_1/PartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_751642#
!dropout_5/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_7_75909dense_7_75911*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_750652!
dense_7/StatefulPartitionedCall?
score/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0score_75914score_75916*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_score_layer_call_and_return_conditional_losses_750822
score/StatefulPartitionedCall?
IdentityIdentity&score/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp$^attention_3/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^score/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 2J
#attention_3/StatefulPartitionedCall#attention_3/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2>
score/StatefulPartitionedCallscore/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
response:PL
'
_output_shapes
:?????????
!
_user_specified_name	applied
?%
?
while_body_73959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_3_73983_0:	2?	(
while_lstm_cell_3_73985_0:	?	-
while_lstm_cell_3_73987_0:
??	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_3_73983:	2?	&
while_lstm_cell_3_73985:	?	+
while_lstm_cell_3_73987:
??	??)while/lstm_cell_3/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_73983_0while_lstm_cell_3_73985_0while_lstm_cell_3_73987_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_739452+
)while/lstm_cell_3/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_73983while_lstm_cell_3_73983_0"4
while_lstm_cell_3_73985while_lstm_cell_3_73985_0"4
while_lstm_cell_3_73987while_lstm_cell_3_73987_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?	
while_body_77406
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	2?	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	?	?
+while_lstm_cell_3_readvariableop_resource_0:
??	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	2?	@
1while_lstm_cell_3_split_1_readvariableop_resource:	?	=
)while_lstm_cell_3_readvariableop_resource:
??	?? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/ones_like/Shape?
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_3/ones_like/Const?
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/ones_like?
while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2!
while/lstm_cell_3/dropout/Const?
while/lstm_cell_3/dropout/MulMul$while/lstm_cell_3/ones_like:output:0(while/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/dropout/Mul?
while/lstm_cell_3/dropout/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_3/dropout/Shape?
6while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???28
6while/lstm_cell_3/dropout/random_uniform/RandomUniform?
(while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(while/lstm_cell_3/dropout/GreaterEqual/y?
&while/lstm_cell_3/dropout/GreaterEqualGreaterEqual?while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22(
&while/lstm_cell_3/dropout/GreaterEqual?
while/lstm_cell_3/dropout/CastCast*while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22 
while/lstm_cell_3/dropout/Cast?
while/lstm_cell_3/dropout/Mul_1Mul!while/lstm_cell_3/dropout/Mul:z:0"while/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout/Mul_1?
!while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_1/Const?
while/lstm_cell_3/dropout_1/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout_1/Mul?
!while/lstm_cell_3/dropout_1/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_1/Shape?
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??2:
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_1/GreaterEqual/y?
(while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22*
(while/lstm_cell_3/dropout_1/GreaterEqual?
 while/lstm_cell_3/dropout_1/CastCast,while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22"
 while/lstm_cell_3/dropout_1/Cast?
!while/lstm_cell_3/dropout_1/Mul_1Mul#while/lstm_cell_3/dropout_1/Mul:z:0$while/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22#
!while/lstm_cell_3/dropout_1/Mul_1?
!while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_2/Const?
while/lstm_cell_3/dropout_2/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout_2/Mul?
!while/lstm_cell_3/dropout_2/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_2/Shape?
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_2/GreaterEqual/y?
(while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22*
(while/lstm_cell_3/dropout_2/GreaterEqual?
 while/lstm_cell_3/dropout_2/CastCast,while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22"
 while/lstm_cell_3/dropout_2/Cast?
!while/lstm_cell_3/dropout_2/Mul_1Mul#while/lstm_cell_3/dropout_2/Mul:z:0$while/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22#
!while/lstm_cell_3/dropout_2/Mul_1?
!while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_3/Const?
while/lstm_cell_3/dropout_3/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22!
while/lstm_cell_3/dropout_3/Mul?
!while/lstm_cell_3/dropout_3/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_3/Shape?
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_3/GreaterEqual/y?
(while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22*
(while/lstm_cell_3/dropout_3/GreaterEqual?
 while/lstm_cell_3/dropout_3/CastCast,while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22"
 while/lstm_cell_3/dropout_3/Cast?
!while/lstm_cell_3/dropout_3/Mul_1Mul#while/lstm_cell_3/dropout_3/Mul:z:0$while/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22#
!while/lstm_cell_3/dropout_3/Mul_1?
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_3/ones_like_1/Shape?
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_3/ones_like_1/Const?
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/ones_like_1?
!while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_4/Const?
while/lstm_cell_3/dropout_4/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_4/Mul?
!while/lstm_cell_3/dropout_4/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_4/Shape?
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_4/GreaterEqual/y?
(while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_4/GreaterEqual?
 while/lstm_cell_3/dropout_4/CastCast,while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_4/Cast?
!while/lstm_cell_3/dropout_4/Mul_1Mul#while/lstm_cell_3/dropout_4/Mul:z:0$while/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_4/Mul_1?
!while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_5/Const?
while/lstm_cell_3/dropout_5/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_5/Mul?
!while/lstm_cell_3/dropout_5/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_5/Shape?
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2췥2:
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_5/GreaterEqual/y?
(while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_5/GreaterEqual?
 while/lstm_cell_3/dropout_5/CastCast,while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_5/Cast?
!while/lstm_cell_3/dropout_5/Mul_1Mul#while/lstm_cell_3/dropout_5/Mul:z:0$while/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_5/Mul_1?
!while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_6/Const?
while/lstm_cell_3/dropout_6/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_6/Mul?
!while/lstm_cell_3/dropout_6/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_6/Shape?
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_6/GreaterEqual/y?
(while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_6/GreaterEqual?
 while/lstm_cell_3/dropout_6/CastCast,while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_6/Cast?
!while/lstm_cell_3/dropout_6/Mul_1Mul#while/lstm_cell_3/dropout_6/Mul:z:0$while/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_6/Mul_1?
!while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2#
!while/lstm_cell_3/dropout_7/Const?
while/lstm_cell_3/dropout_7/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell_3/dropout_7/Mul?
!while/lstm_cell_3/dropout_7/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/dropout_7/Shape?
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2:
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniform?
*while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*while/lstm_cell_3/dropout_7/GreaterEqual/y?
(while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2*
(while/lstm_cell_3/dropout_7/GreaterEqual?
 while/lstm_cell_3/dropout_7/CastCast,while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2"
 while/lstm_cell_3/dropout_7/Cast?
!while/lstm_cell_3/dropout_7/Mul_1Mul#while/lstm_cell_3/dropout_7/Mul:z:0$while/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2#
!while/lstm_cell_3/dropout_7/Mul_1?
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul?
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_1?
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_2?
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_3?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	2?	*
dtype02(
&while/lstm_cell_3/split/ReadVariableOp?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_2?
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_3?
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_3/split_1/split_dim?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?	*
dtype02*
(while/lstm_cell_3/split_1/ReadVariableOp?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_3/split_1?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd?
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_1?
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_2?
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_3?
while/lstm_cell_3/mul_4Mulwhile_placeholder_2%while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_4?
while/lstm_cell_3/mul_5Mulwhile_placeholder_2%while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_5?
while/lstm_cell_3/mul_6Mulwhile_placeholder_2%while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_6?
while/lstm_cell_3/mul_7Mulwhile_placeholder_2%while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_7?
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02"
 while/lstm_cell_3/ReadVariableOp?
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_3/strided_slice/stack?
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice/stack_1?
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_3/strided_slice/stack_2?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_3/strided_slice?
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_4?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid?
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_1?
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice_1/stack?
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2+
)while/lstm_cell_3/strided_slice_1/stack_1?
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_1/stack_2?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_1?
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_5?
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_8?
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_2?
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell_3/strided_slice_2/stack?
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2+
)while/lstm_cell_3/strided_slice_2/stack_1?
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_2/stack_2?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_2?
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_6?
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_2?
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh?
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_9?
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_3?
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_3?
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell_3/strided_slice_3/stack?
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_3/strided_slice_3/stack_1?
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_3/stack_2?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_3?
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_7?
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_4?
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh_1?
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
ψ
?	
while_body_77721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	2?	B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	?	?
+while_lstm_cell_3_readvariableop_resource_0:
??	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	2?	@
1while_lstm_cell_3_split_1_readvariableop_resource:	?	=
)while_lstm_cell_3_readvariableop_resource:
??	?? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2#
!while/lstm_cell_3/ones_like/Shape?
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!while/lstm_cell_3/ones_like/Const?
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/ones_like?
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2%
#while/lstm_cell_3/ones_like_1/Shape?
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#while/lstm_cell_3/ones_like_1/Const?
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/ones_like_1?
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul?
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_1?
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_2?
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_3/mul_3?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	2?	*
dtype02(
&while/lstm_cell_3/split/ReadVariableOp?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_2?
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_3?
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_3/split_1/split_dim?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?	*
dtype02*
(while/lstm_cell_3/split_1/ReadVariableOp?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell_3/split_1?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd?
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_1?
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_2?
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAdd_3?
while/lstm_cell_3/mul_4Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_4?
while/lstm_cell_3/mul_5Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_5?
while/lstm_cell_3/mul_6Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_6?
while/lstm_cell_3/mul_7Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_7?
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02"
 while/lstm_cell_3/ReadVariableOp?
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_3/strided_slice/stack?
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice/stack_1?
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_3/strided_slice/stack_2?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell_3/strided_slice?
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_4?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid?
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_1?
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2)
'while/lstm_cell_3/strided_slice_1/stack?
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2+
)while/lstm_cell_3/strided_slice_1/stack_1?
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_1/stack_2?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_1?
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_5?
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_8?
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_2?
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell_3/strided_slice_2/stack?
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2+
)while/lstm_cell_3/strided_slice_2/stack_1?
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_2/stack_2?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_2?
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_6?
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_2?
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh?
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_9?
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_3?
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??	*
dtype02$
"while/lstm_cell_3/ReadVariableOp_3?
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell_3/strided_slice_3/stack?
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_3/strided_slice_3/stack_1?
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_3/strided_slice_3/stack_2?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2#
!while/lstm_cell_3/strided_slice_3?
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_7?
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add_4?
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/Tanh_1?
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/mul_10?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_dense_6_layer_call_fn_78413

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_750332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
D__inference_Taghipour_layer_call_and_return_conditional_losses_76838
inputs_response#
inputs_whether_criteria_applied6
"embedding_9_embedding_lookup_76336:
??K
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:?26
(conv1d_5_biasadd_readvariableop_resource:2C
0lstm_3_lstm_cell_3_split_readvariableop_resource:	2?	A
2lstm_3_lstm_cell_3_split_1_readvariableop_resource:	?	>
*lstm_3_lstm_cell_3_readvariableop_resource:
??	?
+attention_3_shape_1_readvariableop_resource:
??<
-attention_3_tensordot_readvariableop_resource:	?9
&dense_5_matmul_readvariableop_resource:	?@5
'dense_5_biasadd_readvariableop_resource:@8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:P 5
'dense_7_biasadd_readvariableop_resource: 6
$score_matmul_readvariableop_resource: 3
%score_biasadd_readvariableop_resource:
identity??$attention_3/Tensordot/ReadVariableOp?$attention_3/transpose/ReadVariableOp?conv1d_5/BiasAdd/ReadVariableOp?+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?embedding_9/embedding_lookup?!lstm_3/lstm_cell_3/ReadVariableOp?#lstm_3/lstm_cell_3/ReadVariableOp_1?#lstm_3/lstm_cell_3/ReadVariableOp_2?#lstm_3/lstm_cell_3/ReadVariableOp_3?'lstm_3/lstm_cell_3/split/ReadVariableOp?)lstm_3/lstm_cell_3/split_1/ReadVariableOp?lstm_3/while?score/BiasAdd/ReadVariableOp?score/MatMul/ReadVariableOp
embedding_9/CastCastinputs_response*

DstT0*

SrcT0*(
_output_shapes
:??????????2
embedding_9/Cast?
embedding_9/embedding_lookupResourceGather"embedding_9_embedding_lookup_76336embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*5
_class+
)'loc:@embedding_9/embedding_lookup/76336*-
_output_shapes
:???????????*
dtype02
embedding_9/embedding_lookup?
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*5
_class+
)'loc:@embedding_9/embedding_lookup/76336*-
_output_shapes
:???????????2'
%embedding_9/embedding_lookup/Identity?
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????2)
'embedding_9/embedding_lookup/Identity_1u
embedding_9/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_9/NotEqual/y?
embedding_9/NotEqualNotEqualinputs_responseembedding_9/NotEqual/y:output:0*
T0*(
_output_shapes
:??????????2
embedding_9/NotEqual?
pos_x_maskedout/CastCastembedding_9/NotEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
pos_x_maskedout/Cast?
pos_x_maskedout/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
pos_x_maskedout/ExpandDims/dim?
pos_x_maskedout/ExpandDims
ExpandDimspos_x_maskedout/Cast:y:0'pos_x_maskedout/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
pos_x_maskedout/ExpandDims?
pos_x_maskedout/stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2
pos_x_maskedout/stack?
pos_x_maskedout/TileTile#pos_x_maskedout/ExpandDims:output:0pos_x_maskedout/stack:output:0*
T0*-
_output_shapes
:???????????2
pos_x_maskedout/Tile?
pos_x_maskedout/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
pos_x_maskedout/transpose/perm?
pos_x_maskedout/transpose	Transposepos_x_maskedout/Tile:output:0'pos_x_maskedout/transpose/perm:output:0*
T0*-
_output_shapes
:???????????2
pos_x_maskedout/transpose?
pos_x_maskedout/mulMul0embedding_9/embedding_lookup/Identity_1:output:0pos_x_maskedout/transpose:y:0*
T0*-
_output_shapes
:???????????2
pos_x_maskedout/mul?
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_5/conv1d/ExpandDims/dim?
conv1d_5/conv1d/ExpandDims
ExpandDimspos_x_maskedout/mul:z:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_5/conv1d/ExpandDims?
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?2*
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim?
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?22
conv1d_5/conv1d/ExpandDims_1?
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????2*
paddingVALID*
strides
2
conv1d_5/conv1d?
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*,
_output_shapes
:??????????2*
squeeze_dims

?????????2
conv1d_5/conv1d/Squeeze?
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp?
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????22
conv1d_5/BiasAdde
lstm_3/ShapeShapeconv1d_5/BiasAdd:output:0*
T0*
_output_shapes
:2
lstm_3/Shape?
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack?
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1?
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2?
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicek
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/mul/y?
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/Less/y?
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessq
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/packed/1?
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const?
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/zeroso
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/mul/y?
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/Less/y?
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lessu
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/packed/1?
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const?
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/zeros_1?
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/perm?
lstm_3/transpose	Transposeconv1d_5/BiasAdd:output:0lstm_3/transpose/perm:output:0*
T0*,
_output_shapes
:??????????22
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1?
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stack?
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1?
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2?
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1?
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_3/TensorArrayV2/element_shape?
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2?
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensor?
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stack?
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1?
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2?
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_3/strided_slice_2?
"lstm_3/lstm_cell_3/ones_like/ShapeShapelstm_3/strided_slice_2:output:0*
T0*
_output_shapes
:2$
"lstm_3/lstm_cell_3/ones_like/Shape?
"lstm_3/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"lstm_3/lstm_cell_3/ones_like/Const?
lstm_3/lstm_cell_3/ones_likeFill+lstm_3/lstm_cell_3/ones_like/Shape:output:0+lstm_3/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/ones_like?
 lstm_3/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2"
 lstm_3/lstm_cell_3/dropout/Const?
lstm_3/lstm_cell_3/dropout/MulMul%lstm_3/lstm_cell_3/ones_like:output:0)lstm_3/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22 
lstm_3/lstm_cell_3/dropout/Mul?
 lstm_3/lstm_cell_3/dropout/ShapeShape%lstm_3/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_3/lstm_cell_3/dropout/Shape?
7lstm_3/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform)lstm_3/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???29
7lstm_3/lstm_cell_3/dropout/random_uniform/RandomUniform?
)lstm_3/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2+
)lstm_3/lstm_cell_3/dropout/GreaterEqual/y?
'lstm_3/lstm_cell_3/dropout/GreaterEqualGreaterEqual@lstm_3/lstm_cell_3/dropout/random_uniform/RandomUniform:output:02lstm_3/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22)
'lstm_3/lstm_cell_3/dropout/GreaterEqual?
lstm_3/lstm_cell_3/dropout/CastCast+lstm_3/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22!
lstm_3/lstm_cell_3/dropout/Cast?
 lstm_3/lstm_cell_3/dropout/Mul_1Mul"lstm_3/lstm_cell_3/dropout/Mul:z:0#lstm_3/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22"
 lstm_3/lstm_cell_3/dropout/Mul_1?
"lstm_3/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2$
"lstm_3/lstm_cell_3/dropout_1/Const?
 lstm_3/lstm_cell_3/dropout_1/MulMul%lstm_3/lstm_cell_3/ones_like:output:0+lstm_3/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_3/lstm_cell_3/dropout_1/Mul?
"lstm_3/lstm_cell_3/dropout_1/ShapeShape%lstm_3/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_3/lstm_cell_3/dropout_1/Shape?
9lstm_3/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???2;
9lstm_3/lstm_cell_3/dropout_1/random_uniform/RandomUniform?
+lstm_3/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+lstm_3/lstm_cell_3/dropout_1/GreaterEqual/y?
)lstm_3/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22+
)lstm_3/lstm_cell_3/dropout_1/GreaterEqual?
!lstm_3/lstm_cell_3/dropout_1/CastCast-lstm_3/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22#
!lstm_3/lstm_cell_3/dropout_1/Cast?
"lstm_3/lstm_cell_3/dropout_1/Mul_1Mul$lstm_3/lstm_cell_3/dropout_1/Mul:z:0%lstm_3/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22$
"lstm_3/lstm_cell_3/dropout_1/Mul_1?
"lstm_3/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2$
"lstm_3/lstm_cell_3/dropout_2/Const?
 lstm_3/lstm_cell_3/dropout_2/MulMul%lstm_3/lstm_cell_3/ones_like:output:0+lstm_3/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_3/lstm_cell_3/dropout_2/Mul?
"lstm_3/lstm_cell_3/dropout_2/ShapeShape%lstm_3/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_3/lstm_cell_3/dropout_2/Shape?
9lstm_3/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??p2;
9lstm_3/lstm_cell_3/dropout_2/random_uniform/RandomUniform?
+lstm_3/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+lstm_3/lstm_cell_3/dropout_2/GreaterEqual/y?
)lstm_3/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22+
)lstm_3/lstm_cell_3/dropout_2/GreaterEqual?
!lstm_3/lstm_cell_3/dropout_2/CastCast-lstm_3/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22#
!lstm_3/lstm_cell_3/dropout_2/Cast?
"lstm_3/lstm_cell_3/dropout_2/Mul_1Mul$lstm_3/lstm_cell_3/dropout_2/Mul:z:0%lstm_3/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22$
"lstm_3/lstm_cell_3/dropout_2/Mul_1?
"lstm_3/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2$
"lstm_3/lstm_cell_3/dropout_3/Const?
 lstm_3/lstm_cell_3/dropout_3/MulMul%lstm_3/lstm_cell_3/ones_like:output:0+lstm_3/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_3/lstm_cell_3/dropout_3/Mul?
"lstm_3/lstm_cell_3/dropout_3/ShapeShape%lstm_3/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_3/lstm_cell_3/dropout_3/Shape?
9lstm_3/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2ɼ?2;
9lstm_3/lstm_cell_3/dropout_3/random_uniform/RandomUniform?
+lstm_3/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+lstm_3/lstm_cell_3/dropout_3/GreaterEqual/y?
)lstm_3/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22+
)lstm_3/lstm_cell_3/dropout_3/GreaterEqual?
!lstm_3/lstm_cell_3/dropout_3/CastCast-lstm_3/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22#
!lstm_3/lstm_cell_3/dropout_3/Cast?
"lstm_3/lstm_cell_3/dropout_3/Mul_1Mul$lstm_3/lstm_cell_3/dropout_3/Mul:z:0%lstm_3/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22$
"lstm_3/lstm_cell_3/dropout_3/Mul_1?
$lstm_3/lstm_cell_3/ones_like_1/ShapeShapelstm_3/zeros:output:0*
T0*
_output_shapes
:2&
$lstm_3/lstm_cell_3/ones_like_1/Shape?
$lstm_3/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$lstm_3/lstm_cell_3/ones_like_1/Const?
lstm_3/lstm_cell_3/ones_like_1Fill-lstm_3/lstm_cell_3/ones_like_1/Shape:output:0-lstm_3/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2 
lstm_3/lstm_cell_3/ones_like_1?
"lstm_3/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2$
"lstm_3/lstm_cell_3/dropout_4/Const?
 lstm_3/lstm_cell_3/dropout_4/MulMul'lstm_3/lstm_cell_3/ones_like_1:output:0+lstm_3/lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/lstm_cell_3/dropout_4/Mul?
"lstm_3/lstm_cell_3/dropout_4/ShapeShape'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2$
"lstm_3/lstm_cell_3/dropout_4/Shape?
9lstm_3/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2;
9lstm_3/lstm_cell_3/dropout_4/random_uniform/RandomUniform?
+lstm_3/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+lstm_3/lstm_cell_3/dropout_4/GreaterEqual/y?
)lstm_3/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)lstm_3/lstm_cell_3/dropout_4/GreaterEqual?
!lstm_3/lstm_cell_3/dropout_4/CastCast-lstm_3/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!lstm_3/lstm_cell_3/dropout_4/Cast?
"lstm_3/lstm_cell_3/dropout_4/Mul_1Mul$lstm_3/lstm_cell_3/dropout_4/Mul:z:0%lstm_3/lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"lstm_3/lstm_cell_3/dropout_4/Mul_1?
"lstm_3/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2$
"lstm_3/lstm_cell_3/dropout_5/Const?
 lstm_3/lstm_cell_3/dropout_5/MulMul'lstm_3/lstm_cell_3/ones_like_1:output:0+lstm_3/lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/lstm_cell_3/dropout_5/Mul?
"lstm_3/lstm_cell_3/dropout_5/ShapeShape'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2$
"lstm_3/lstm_cell_3/dropout_5/Shape?
9lstm_3/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2;
9lstm_3/lstm_cell_3/dropout_5/random_uniform/RandomUniform?
+lstm_3/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+lstm_3/lstm_cell_3/dropout_5/GreaterEqual/y?
)lstm_3/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)lstm_3/lstm_cell_3/dropout_5/GreaterEqual?
!lstm_3/lstm_cell_3/dropout_5/CastCast-lstm_3/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!lstm_3/lstm_cell_3/dropout_5/Cast?
"lstm_3/lstm_cell_3/dropout_5/Mul_1Mul$lstm_3/lstm_cell_3/dropout_5/Mul:z:0%lstm_3/lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"lstm_3/lstm_cell_3/dropout_5/Mul_1?
"lstm_3/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2$
"lstm_3/lstm_cell_3/dropout_6/Const?
 lstm_3/lstm_cell_3/dropout_6/MulMul'lstm_3/lstm_cell_3/ones_like_1:output:0+lstm_3/lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/lstm_cell_3/dropout_6/Mul?
"lstm_3/lstm_cell_3/dropout_6/ShapeShape'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2$
"lstm_3/lstm_cell_3/dropout_6/Shape?
9lstm_3/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?К2;
9lstm_3/lstm_cell_3/dropout_6/random_uniform/RandomUniform?
+lstm_3/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+lstm_3/lstm_cell_3/dropout_6/GreaterEqual/y?
)lstm_3/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)lstm_3/lstm_cell_3/dropout_6/GreaterEqual?
!lstm_3/lstm_cell_3/dropout_6/CastCast-lstm_3/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!lstm_3/lstm_cell_3/dropout_6/Cast?
"lstm_3/lstm_cell_3/dropout_6/Mul_1Mul$lstm_3/lstm_cell_3/dropout_6/Mul:z:0%lstm_3/lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"lstm_3/lstm_cell_3/dropout_6/Mul_1?
"lstm_3/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2$
"lstm_3/lstm_cell_3/dropout_7/Const?
 lstm_3/lstm_cell_3/dropout_7/MulMul'lstm_3/lstm_cell_3/ones_like_1:output:0+lstm_3/lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/lstm_cell_3/dropout_7/Mul?
"lstm_3/lstm_cell_3/dropout_7/ShapeShape'lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2$
"lstm_3/lstm_cell_3/dropout_7/Shape?
9lstm_3/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform+lstm_3/lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??.2;
9lstm_3/lstm_cell_3/dropout_7/random_uniform/RandomUniform?
+lstm_3/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2-
+lstm_3/lstm_cell_3/dropout_7/GreaterEqual/y?
)lstm_3/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualBlstm_3/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:04lstm_3/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)lstm_3/lstm_cell_3/dropout_7/GreaterEqual?
!lstm_3/lstm_cell_3/dropout_7/CastCast-lstm_3/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!lstm_3/lstm_cell_3/dropout_7/Cast?
"lstm_3/lstm_cell_3/dropout_7/Mul_1Mul$lstm_3/lstm_cell_3/dropout_7/Mul:z:0%lstm_3/lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"lstm_3/lstm_cell_3/dropout_7/Mul_1?
lstm_3/lstm_cell_3/mulMullstm_3/strided_slice_2:output:0$lstm_3/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/mul?
lstm_3/lstm_cell_3/mul_1Mullstm_3/strided_slice_2:output:0&lstm_3/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/mul_1?
lstm_3/lstm_cell_3/mul_2Mullstm_3/strided_slice_2:output:0&lstm_3/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/mul_2?
lstm_3/lstm_cell_3/mul_3Mullstm_3/strided_slice_2:output:0&lstm_3/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_3/lstm_cell_3/mul_3?
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_3/split/split_dim?
'lstm_3/lstm_cell_3/split/ReadVariableOpReadVariableOp0lstm_3_lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	2?	*
dtype02)
'lstm_3/lstm_cell_3/split/ReadVariableOp?
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0/lstm_3/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
lstm_3/lstm_cell_3/split?
lstm_3/lstm_cell_3/MatMulMatMullstm_3/lstm_cell_3/mul:z:0!lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul?
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/lstm_cell_3/mul_1:z:0!lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_1?
lstm_3/lstm_cell_3/MatMul_2MatMullstm_3/lstm_cell_3/mul_2:z:0!lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_2?
lstm_3/lstm_cell_3/MatMul_3MatMullstm_3/lstm_cell_3/mul_3:z:0!lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_3?
$lstm_3/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_3/lstm_cell_3/split_1/split_dim?
)lstm_3/lstm_cell_3/split_1/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02+
)lstm_3/lstm_cell_3/split_1/ReadVariableOp?
lstm_3/lstm_cell_3/split_1Split-lstm_3/lstm_cell_3/split_1/split_dim:output:01lstm_3/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_3/lstm_cell_3/split_1?
lstm_3/lstm_cell_3/BiasAddBiasAdd#lstm_3/lstm_cell_3/MatMul:product:0#lstm_3/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAdd?
lstm_3/lstm_cell_3/BiasAdd_1BiasAdd%lstm_3/lstm_cell_3/MatMul_1:product:0#lstm_3/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAdd_1?
lstm_3/lstm_cell_3/BiasAdd_2BiasAdd%lstm_3/lstm_cell_3/MatMul_2:product:0#lstm_3/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAdd_2?
lstm_3/lstm_cell_3/BiasAdd_3BiasAdd%lstm_3/lstm_cell_3/MatMul_3:product:0#lstm_3/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAdd_3?
lstm_3/lstm_cell_3/mul_4Mullstm_3/zeros:output:0&lstm_3/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_4?
lstm_3/lstm_cell_3/mul_5Mullstm_3/zeros:output:0&lstm_3/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_5?
lstm_3/lstm_cell_3/mul_6Mullstm_3/zeros:output:0&lstm_3/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_6?
lstm_3/lstm_cell_3/mul_7Mullstm_3/zeros:output:0&lstm_3/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_7?
!lstm_3/lstm_cell_3/ReadVariableOpReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02#
!lstm_3/lstm_cell_3/ReadVariableOp?
&lstm_3/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_3/lstm_cell_3/strided_slice/stack?
(lstm_3/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2*
(lstm_3/lstm_cell_3/strided_slice/stack_1?
(lstm_3/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_3/lstm_cell_3/strided_slice/stack_2?
 lstm_3/lstm_cell_3/strided_sliceStridedSlice)lstm_3/lstm_cell_3/ReadVariableOp:value:0/lstm_3/lstm_cell_3/strided_slice/stack:output:01lstm_3/lstm_cell_3/strided_slice/stack_1:output:01lstm_3/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2"
 lstm_3/lstm_cell_3/strided_slice?
lstm_3/lstm_cell_3/MatMul_4MatMullstm_3/lstm_cell_3/mul_4:z:0)lstm_3/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_4?
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/BiasAdd:output:0%lstm_3/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add?
lstm_3/lstm_cell_3/SigmoidSigmoidlstm_3/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Sigmoid?
#lstm_3/lstm_cell_3/ReadVariableOp_1ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02%
#lstm_3/lstm_cell_3/ReadVariableOp_1?
(lstm_3/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2*
(lstm_3/lstm_cell_3/strided_slice_1/stack?
*lstm_3/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2,
*lstm_3/lstm_cell_3/strided_slice_1/stack_1?
*lstm_3/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_3/lstm_cell_3/strided_slice_1/stack_2?
"lstm_3/lstm_cell_3/strided_slice_1StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_1:value:01lstm_3/lstm_cell_3/strided_slice_1/stack:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"lstm_3/lstm_cell_3/strided_slice_1?
lstm_3/lstm_cell_3/MatMul_5MatMullstm_3/lstm_cell_3/mul_5:z:0+lstm_3/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_5?
lstm_3/lstm_cell_3/add_1AddV2%lstm_3/lstm_cell_3/BiasAdd_1:output:0%lstm_3/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add_1?
lstm_3/lstm_cell_3/Sigmoid_1Sigmoidlstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Sigmoid_1?
lstm_3/lstm_cell_3/mul_8Mul lstm_3/lstm_cell_3/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_8?
#lstm_3/lstm_cell_3/ReadVariableOp_2ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02%
#lstm_3/lstm_cell_3/ReadVariableOp_2?
(lstm_3/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2*
(lstm_3/lstm_cell_3/strided_slice_2/stack?
*lstm_3/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*lstm_3/lstm_cell_3/strided_slice_2/stack_1?
*lstm_3/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_3/lstm_cell_3/strided_slice_2/stack_2?
"lstm_3/lstm_cell_3/strided_slice_2StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_2:value:01lstm_3/lstm_cell_3/strided_slice_2/stack:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"lstm_3/lstm_cell_3/strided_slice_2?
lstm_3/lstm_cell_3/MatMul_6MatMullstm_3/lstm_cell_3/mul_6:z:0+lstm_3/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_6?
lstm_3/lstm_cell_3/add_2AddV2%lstm_3/lstm_cell_3/BiasAdd_2:output:0%lstm_3/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add_2?
lstm_3/lstm_cell_3/TanhTanhlstm_3/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Tanh?
lstm_3/lstm_cell_3/mul_9Mullstm_3/lstm_cell_3/Sigmoid:y:0lstm_3/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_9?
lstm_3/lstm_cell_3/add_3AddV2lstm_3/lstm_cell_3/mul_8:z:0lstm_3/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add_3?
#lstm_3/lstm_cell_3/ReadVariableOp_3ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02%
#lstm_3/lstm_cell_3/ReadVariableOp_3?
(lstm_3/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2*
(lstm_3/lstm_cell_3/strided_slice_3/stack?
*lstm_3/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_3/lstm_cell_3/strided_slice_3/stack_1?
*lstm_3/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_3/lstm_cell_3/strided_slice_3/stack_2?
"lstm_3/lstm_cell_3/strided_slice_3StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_3:value:01lstm_3/lstm_cell_3/strided_slice_3/stack:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"lstm_3/lstm_cell_3/strided_slice_3?
lstm_3/lstm_cell_3/MatMul_7MatMullstm_3/lstm_cell_3/mul_7:z:0+lstm_3/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_7?
lstm_3/lstm_cell_3/add_4AddV2%lstm_3/lstm_cell_3/BiasAdd_3:output:0%lstm_3/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add_4?
lstm_3/lstm_cell_3/Sigmoid_2Sigmoidlstm_3/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Sigmoid_2?
lstm_3/lstm_cell_3/Tanh_1Tanhlstm_3/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/Tanh_1?
lstm_3/lstm_cell_3/mul_10Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0lstm_3/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/mul_10?
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2&
$lstm_3/TensorArrayV2_1/element_shape?
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/time?
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counter?
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_3_lstm_cell_3_split_readvariableop_resource2lstm_3_lstm_cell_3_split_1_readvariableop_resource*lstm_3_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_76540*#
condR
lstm_3_while_cond_76539*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_3/while?
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStack?
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_3/strided_slice_3/stack?
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1?
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2?
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_3/strided_slice_3?
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/perm?
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimew
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_4/dropout/Const?
dropout_4/dropout/MulMullstm_3/transpose_1:y:0 dropout_4/dropout/Const:output:0*
T0*-
_output_shapes
:???????????2
dropout_4/dropout/Mulx
dropout_4/dropout/ShapeShapelstm_3/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*-
_output_shapes
:???????????2
dropout_4/dropout/Mul_1q
attention_3/ShapeShapedropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
attention_3/Shape?
attention_3/unstackUnpackattention_3/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
attention_3/unstack?
"attention_3/Shape_1/ReadVariableOpReadVariableOp+attention_3_shape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"attention_3/Shape_1/ReadVariableOp{
attention_3/Shape_1Const*
_output_shapes
:*
dtype0*
valueB",  ,  2
attention_3/Shape_1?
attention_3/unstack_1Unpackattention_3/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
attention_3/unstack_1?
attention_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
attention_3/Reshape/shape?
attention_3/ReshapeReshapedropout_4/dropout/Mul_1:z:0"attention_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
attention_3/Reshape?
$attention_3/transpose/ReadVariableOpReadVariableOp+attention_3_shape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$attention_3/transpose/ReadVariableOp?
attention_3/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_3/transpose/perm?
attention_3/transpose	Transpose,attention_3/transpose/ReadVariableOp:value:0#attention_3/transpose/perm:output:0*
T0* 
_output_shapes
:
??2
attention_3/transpose?
attention_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB",  ????2
attention_3/Reshape_1/shape?
attention_3/Reshape_1Reshapeattention_3/transpose:y:0$attention_3/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
??2
attention_3/Reshape_1?
attention_3/MatMulMatMulattention_3/Reshape:output:0attention_3/Reshape_1:output:0*
T0*(
_output_shapes
:??????????2
attention_3/MatMul?
attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
attention_3/Reshape_2/shape/1?
attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
attention_3/Reshape_2/shape/2?
attention_3/Reshape_2/shapePackattention_3/unstack:output:0&attention_3/Reshape_2/shape/1:output:0&attention_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
attention_3/Reshape_2/shape?
attention_3/Reshape_2Reshapeattention_3/MatMul:product:0$attention_3/Reshape_2/shape:output:0*
T0*-
_output_shapes
:???????????2
attention_3/Reshape_2?
attention_3/TanhTanhattention_3/Reshape_2:output:0*
T0*-
_output_shapes
:???????????2
attention_3/Tanh?
$attention_3/Tensordot/ReadVariableOpReadVariableOp-attention_3_tensordot_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$attention_3/Tensordot/ReadVariableOp?
#attention_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  2%
#attention_3/Tensordot/Reshape/shape?
attention_3/Tensordot/ReshapeReshape,attention_3/Tensordot/ReadVariableOp:value:0,attention_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
attention_3/Tensordot/Reshape?
attention_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
attention_3/Tensordot/axes?
attention_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_3/Tensordot/free~
attention_3/Tensordot/ShapeShapeattention_3/Tanh:y:0*
T0*
_output_shapes
:2
attention_3/Tensordot/Shape?
#attention_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#attention_3/Tensordot/GatherV2/axis?
attention_3/Tensordot/GatherV2GatherV2$attention_3/Tensordot/Shape:output:0#attention_3/Tensordot/free:output:0,attention_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
attention_3/Tensordot/GatherV2?
%attention_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%attention_3/Tensordot/GatherV2_1/axis?
 attention_3/Tensordot/GatherV2_1GatherV2$attention_3/Tensordot/Shape:output:0#attention_3/Tensordot/axes:output:0.attention_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 attention_3/Tensordot/GatherV2_1?
attention_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
attention_3/Tensordot/Const?
attention_3/Tensordot/ProdProd'attention_3/Tensordot/GatherV2:output:0$attention_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
attention_3/Tensordot/Prod?
attention_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
attention_3/Tensordot/Const_1?
attention_3/Tensordot/Prod_1Prod)attention_3/Tensordot/GatherV2_1:output:0&attention_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
attention_3/Tensordot/Prod_1?
!attention_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!attention_3/Tensordot/concat/axis?
attention_3/Tensordot/concatConcatV2#attention_3/Tensordot/axes:output:0#attention_3/Tensordot/free:output:0*attention_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
attention_3/Tensordot/concat?
attention_3/Tensordot/stackPack%attention_3/Tensordot/Prod_1:output:0#attention_3/Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2
attention_3/Tensordot/stack?
attention_3/Tensordot/transpose	Transposeattention_3/Tanh:y:0%attention_3/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2!
attention_3/Tensordot/transpose?
attention_3/Tensordot/Reshape_1Reshape#attention_3/Tensordot/transpose:y:0$attention_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2!
attention_3/Tensordot/Reshape_1?
attention_3/Tensordot/MatMulMatMul&attention_3/Tensordot/Reshape:output:0(attention_3/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2
attention_3/Tensordot/MatMul?
attention_3/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
attention_3/Tensordot/Const_2?
#attention_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#attention_3/Tensordot/concat_1/axis?
attention_3/Tensordot/concat_1ConcatV2&attention_3/Tensordot/Const_2:output:0'attention_3/Tensordot/GatherV2:output:0,attention_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2 
attention_3/Tensordot/concat_1?
attention_3/TensordotReshape&attention_3/Tensordot/MatMul:product:0'attention_3/Tensordot/concat_1:output:0*
T0*(
_output_shapes
:??????????2
attention_3/Tensordot?
attention_3/SoftmaxSoftmaxattention_3/Tensordot:output:0*
T0*(
_output_shapes
:??????????2
attention_3/Softmaxz
attention_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
attention_3/ExpandDims/dim?
attention_3/ExpandDims
ExpandDimsattention_3/Softmax:softmax:0#attention_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
attention_3/ExpandDims{
attention_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2
attention_3/stack?
attention_3/TileTileattention_3/ExpandDims:output:0attention_3/stack:output:0*
T0*-
_output_shapes
:???????????2
attention_3/Tile?
attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
attention_3/transpose_1/perm?
attention_3/transpose_1	Transposeattention_3/Tile:output:0%attention_3/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
attention_3/transpose_1?
attention_3/mulMuldropout_4/dropout/Mul_1:z:0attention_3/transpose_1:y:0*
T0*-
_output_shapes
:???????????2
attention_3/mul?
!attention_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2#
!attention_3/Sum/reduction_indices?
attention_3/SumSumattention_3/mul:z:0*attention_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
attention_3/Sum?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulattention_3/Sum:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs_whether_criteria_applied%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2dense_5/BiasAdd:output:0dense_6/BiasAdd:output:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????P2
concatenate_1/concatw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_5/dropout/Const?
dropout_5/dropout/MulMulconcatenate_1/concat:output:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout_5/dropout/Mul
dropout_5/dropout/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout_5/dropout/Mul_1?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:P *
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_7/BiasAdd?
score/MatMul/ReadVariableOpReadVariableOp$score_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
score/MatMul/ReadVariableOp?
score/MatMulMatMuldense_7/BiasAdd:output:0#score/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
score/MatMul?
score/BiasAdd/ReadVariableOpReadVariableOp%score_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
score/BiasAdd/ReadVariableOp?
score/BiasAddBiasAddscore/MatMul:product:0$score/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
score/BiasAdds
score/SoftmaxSoftmaxscore/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
score/Softmaxr
IdentityIdentityscore/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp%^attention_3/Tensordot/ReadVariableOp%^attention_3/transpose/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/conv1d/ExpandDims_1/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^embedding_9/embedding_lookup"^lstm_3/lstm_cell_3/ReadVariableOp$^lstm_3/lstm_cell_3/ReadVariableOp_1$^lstm_3/lstm_cell_3/ReadVariableOp_2$^lstm_3/lstm_cell_3/ReadVariableOp_3(^lstm_3/lstm_cell_3/split/ReadVariableOp*^lstm_3/lstm_cell_3/split_1/ReadVariableOp^lstm_3/while^score/BiasAdd/ReadVariableOp^score/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 2L
$attention_3/Tensordot/ReadVariableOp$attention_3/Tensordot/ReadVariableOp2L
$attention_3/transpose/ReadVariableOp$attention_3/transpose/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2<
embedding_9/embedding_lookupembedding_9/embedding_lookup2F
!lstm_3/lstm_cell_3/ReadVariableOp!lstm_3/lstm_cell_3/ReadVariableOp2J
#lstm_3/lstm_cell_3/ReadVariableOp_1#lstm_3/lstm_cell_3/ReadVariableOp_12J
#lstm_3/lstm_cell_3/ReadVariableOp_2#lstm_3/lstm_cell_3/ReadVariableOp_22J
#lstm_3/lstm_cell_3/ReadVariableOp_3#lstm_3/lstm_cell_3/ReadVariableOp_32R
'lstm_3/lstm_cell_3/split/ReadVariableOp'lstm_3/lstm_cell_3/split/ReadVariableOp2V
)lstm_3/lstm_cell_3/split_1/ReadVariableOp)lstm_3/lstm_cell_3/split_1/ReadVariableOp2
lstm_3/whilelstm_3/while2<
score/BiasAdd/ReadVariableOpscore/BiasAdd/ReadVariableOp2:
score/MatMul/ReadVariableOpscore/MatMul/ReadVariableOp:Y U
(
_output_shapes
:??????????
)
_user_specified_nameinputs/response:hd
'
_output_shapes
:?????????
9
_user_specified_name!inputs/whether_criteria_applied
?
?
'__inference_dense_7_layer_call_fn_78472

inputs
unknown:P 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_750652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_75619

inputs<
)lstm_cell_3_split_readvariableop_resource:	2?	:
+lstm_cell_3_split_1_readvariableop_resource:	?	7
#lstm_cell_3_readvariableop_resource:
??	
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like/Shape
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like/Const?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/ones_like{
lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout/Const?
lstm_cell_3/dropout/MulMullstm_cell_3/ones_like:output:0"lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout/Mul?
lstm_cell_3/dropout/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout/Shape?
0lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2??R22
0lstm_cell_3/dropout/random_uniform/RandomUniform?
"lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2$
"lstm_cell_3/dropout/GreaterEqual/y?
 lstm_cell_3/dropout/GreaterEqualGreaterEqual9lstm_cell_3/dropout/random_uniform/RandomUniform:output:0+lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_cell_3/dropout/GreaterEqual?
lstm_cell_3/dropout/CastCast$lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout/Cast?
lstm_cell_3/dropout/Mul_1Mullstm_cell_3/dropout/Mul:z:0lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout/Mul_1
lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_1/Const?
lstm_cell_3/dropout_1/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_1/Mul?
lstm_cell_3/dropout_1/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_1/Shape?
2lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_1/random_uniform/RandomUniform?
$lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_1/GreaterEqual/y?
"lstm_cell_3/dropout_1/GreaterEqualGreaterEqual;lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_cell_3/dropout_1/GreaterEqual?
lstm_cell_3/dropout_1/CastCast&lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout_1/Cast?
lstm_cell_3/dropout_1/Mul_1Mullstm_cell_3/dropout_1/Mul:z:0lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_1/Mul_1
lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_2/Const?
lstm_cell_3/dropout_2/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_2/Mul?
lstm_cell_3/dropout_2/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_2/Shape?
2lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_2/random_uniform/RandomUniform?
$lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_2/GreaterEqual/y?
"lstm_cell_3/dropout_2/GreaterEqualGreaterEqual;lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_cell_3/dropout_2/GreaterEqual?
lstm_cell_3/dropout_2/CastCast&lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout_2/Cast?
lstm_cell_3/dropout_2/Mul_1Mullstm_cell_3/dropout_2/Mul:z:0lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_2/Mul_1
lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_3/Const?
lstm_cell_3/dropout_3/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_3/Mul?
lstm_cell_3/dropout_3/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_3/Shape?
2lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????2*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_3/random_uniform/RandomUniform?
$lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_3/GreaterEqual/y?
"lstm_cell_3/dropout_3/GreaterEqualGreaterEqual;lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????22$
"lstm_cell_3/dropout_3/GreaterEqual?
lstm_cell_3/dropout_3/CastCast&lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????22
lstm_cell_3/dropout_3/Cast?
lstm_cell_3/dropout_3/Mul_1Mullstm_cell_3/dropout_3/Mul:z:0lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/dropout_3/Mul_1|
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_3/ones_like_1/Shape?
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm_cell_3/ones_like_1/Const?
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/ones_like_1
lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_4/Const?
lstm_cell_3/dropout_4/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_4/Mul?
lstm_cell_3/dropout_4/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_4/Shape?
2lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_4/random_uniform/RandomUniform?
$lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_4/GreaterEqual/y?
"lstm_cell_3/dropout_4/GreaterEqualGreaterEqual;lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_4/GreaterEqual?
lstm_cell_3/dropout_4/CastCast&lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_4/Cast?
lstm_cell_3/dropout_4/Mul_1Mullstm_cell_3/dropout_4/Mul:z:0lstm_cell_3/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_4/Mul_1
lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_5/Const?
lstm_cell_3/dropout_5/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_5/Mul?
lstm_cell_3/dropout_5/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_5/Shape?
2lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_5/random_uniform/RandomUniform?
$lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_5/GreaterEqual/y?
"lstm_cell_3/dropout_5/GreaterEqualGreaterEqual;lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_5/GreaterEqual?
lstm_cell_3/dropout_5/CastCast&lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_5/Cast?
lstm_cell_3/dropout_5/Mul_1Mullstm_cell_3/dropout_5/Mul:z:0lstm_cell_3/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_5/Mul_1
lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_6/Const?
lstm_cell_3/dropout_6/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_6/Mul?
lstm_cell_3/dropout_6/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_6/Shape?
2lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_6/random_uniform/RandomUniform?
$lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_6/GreaterEqual/y?
"lstm_cell_3/dropout_6/GreaterEqualGreaterEqual;lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_6/GreaterEqual?
lstm_cell_3/dropout_6/CastCast&lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_6/Cast?
lstm_cell_3/dropout_6/Mul_1Mullstm_cell_3/dropout_6/Mul:z:0lstm_cell_3/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_6/Mul_1
lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
lstm_cell_3/dropout_7/Const?
lstm_cell_3/dropout_7/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_7/Mul?
lstm_cell_3/dropout_7/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell_3/dropout_7/Shape?
2lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???24
2lstm_cell_3/dropout_7/random_uniform/RandomUniform?
$lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$lstm_cell_3/dropout_7/GreaterEqual/y?
"lstm_cell_3/dropout_7/GreaterEqualGreaterEqual;lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_cell_3/dropout_7/GreaterEqual?
lstm_cell_3/dropout_7/CastCast&lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
lstm_cell_3/dropout_7/Cast?
lstm_cell_3/dropout_7/Mul_1Mullstm_cell_3/dropout_7/Mul:z:0lstm_cell_3/dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/dropout_7/Mul_1?
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul?
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_1?
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_2?
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_3/mul_3|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	2?	*
dtype02"
 lstm_cell_3/split/ReadVariableOp?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2
lstm_cell_3/split?
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_2?
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_3?
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_3/split_1/split_dim?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?	*
dtype02$
"lstm_cell_3/split_1/ReadVariableOp?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell_3/split_1?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd?
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_1?
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_2?
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAdd_3?
lstm_cell_3/mul_4Mulzeros:output:0lstm_cell_3/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_4?
lstm_cell_3/mul_5Mulzeros:output:0lstm_cell_3/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_5?
lstm_cell_3/mul_6Mulzeros:output:0lstm_cell_3/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_6?
lstm_cell_3/mul_7Mulzeros:output:0lstm_cell_3/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_7?
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp?
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_3/strided_slice/stack?
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice/stack_1?
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_3/strided_slice/stack_2?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice?
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_4?
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add}
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid?
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_1?
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!lstm_cell_3/strided_slice_1/stack?
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2%
#lstm_cell_3/strided_slice_1/stack_1?
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_1/stack_2?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_1?
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_5?
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_8?
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_2?
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell_3/strided_slice_2/stack?
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#lstm_cell_3/strided_slice_2/stack_1?
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_2/stack_2?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_2?
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_6?
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_2v
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh?
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_9?
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_3?
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02
lstm_cell_3/ReadVariableOp_3?
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell_3/strided_slice_3/stack?
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_3/strided_slice_3/stack_1?
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_3/strided_slice_3/stack_2?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell_3/strided_slice_3?
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_7?
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add_4?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Sigmoid_2z
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/Tanh_1?
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/mul_10?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_75421*
condR
while_cond_75420*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimep
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:???????????2

Identity?
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????2: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????2
 
_user_specified_nameinputs
?

?
B__inference_dense_7_layer_call_and_return_conditional_losses_78463

inputs0
matmul_readvariableop_resource:P -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
!Taghipour_lstm_3_while_cond_73599>
:taghipour_lstm_3_while_taghipour_lstm_3_while_loop_counterD
@taghipour_lstm_3_while_taghipour_lstm_3_while_maximum_iterations&
"taghipour_lstm_3_while_placeholder(
$taghipour_lstm_3_while_placeholder_1(
$taghipour_lstm_3_while_placeholder_2(
$taghipour_lstm_3_while_placeholder_3@
<taghipour_lstm_3_while_less_taghipour_lstm_3_strided_slice_1U
Qtaghipour_lstm_3_while_taghipour_lstm_3_while_cond_73599___redundant_placeholder0U
Qtaghipour_lstm_3_while_taghipour_lstm_3_while_cond_73599___redundant_placeholder1U
Qtaghipour_lstm_3_while_taghipour_lstm_3_while_cond_73599___redundant_placeholder2U
Qtaghipour_lstm_3_while_taghipour_lstm_3_while_cond_73599___redundant_placeholder3#
taghipour_lstm_3_while_identity
?
Taghipour/lstm_3/while/LessLess"taghipour_lstm_3_while_placeholder<taghipour_lstm_3_while_less_taghipour_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
Taghipour/lstm_3/while/Less?
Taghipour/lstm_3/while/IdentityIdentityTaghipour/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2!
Taghipour/lstm_3/while/Identity"K
taghipour_lstm_3_while_identity(Taghipour/lstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
 __inference__wrapped_model_73820
response
applied@
,taghipour_embedding_9_embedding_lookup_73460:
??U
>taghipour_conv1d_5_conv1d_expanddims_1_readvariableop_resource:?2@
2taghipour_conv1d_5_biasadd_readvariableop_resource:2M
:taghipour_lstm_3_lstm_cell_3_split_readvariableop_resource:	2?	K
<taghipour_lstm_3_lstm_cell_3_split_1_readvariableop_resource:	?	H
4taghipour_lstm_3_lstm_cell_3_readvariableop_resource:
??	I
5taghipour_attention_3_shape_1_readvariableop_resource:
??F
7taghipour_attention_3_tensordot_readvariableop_resource:	?C
0taghipour_dense_5_matmul_readvariableop_resource:	?@?
1taghipour_dense_5_biasadd_readvariableop_resource:@B
0taghipour_dense_6_matmul_readvariableop_resource:?
1taghipour_dense_6_biasadd_readvariableop_resource:B
0taghipour_dense_7_matmul_readvariableop_resource:P ?
1taghipour_dense_7_biasadd_readvariableop_resource: @
.taghipour_score_matmul_readvariableop_resource: =
/taghipour_score_biasadd_readvariableop_resource:
identity??.Taghipour/attention_3/Tensordot/ReadVariableOp?.Taghipour/attention_3/transpose/ReadVariableOp?)Taghipour/conv1d_5/BiasAdd/ReadVariableOp?5Taghipour/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?(Taghipour/dense_5/BiasAdd/ReadVariableOp?'Taghipour/dense_5/MatMul/ReadVariableOp?(Taghipour/dense_6/BiasAdd/ReadVariableOp?'Taghipour/dense_6/MatMul/ReadVariableOp?(Taghipour/dense_7/BiasAdd/ReadVariableOp?'Taghipour/dense_7/MatMul/ReadVariableOp?&Taghipour/embedding_9/embedding_lookup?+Taghipour/lstm_3/lstm_cell_3/ReadVariableOp?-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_1?-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_2?-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_3?1Taghipour/lstm_3/lstm_cell_3/split/ReadVariableOp?3Taghipour/lstm_3/lstm_cell_3/split_1/ReadVariableOp?Taghipour/lstm_3/while?&Taghipour/score/BiasAdd/ReadVariableOp?%Taghipour/score/MatMul/ReadVariableOp?
Taghipour/embedding_9/CastCastresponse*

DstT0*

SrcT0*(
_output_shapes
:??????????2
Taghipour/embedding_9/Cast?
&Taghipour/embedding_9/embedding_lookupResourceGather,taghipour_embedding_9_embedding_lookup_73460Taghipour/embedding_9/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*?
_class5
31loc:@Taghipour/embedding_9/embedding_lookup/73460*-
_output_shapes
:???????????*
dtype02(
&Taghipour/embedding_9/embedding_lookup?
/Taghipour/embedding_9/embedding_lookup/IdentityIdentity/Taghipour/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@Taghipour/embedding_9/embedding_lookup/73460*-
_output_shapes
:???????????21
/Taghipour/embedding_9/embedding_lookup/Identity?
1Taghipour/embedding_9/embedding_lookup/Identity_1Identity8Taghipour/embedding_9/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????23
1Taghipour/embedding_9/embedding_lookup/Identity_1?
 Taghipour/embedding_9/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 Taghipour/embedding_9/NotEqual/y?
Taghipour/embedding_9/NotEqualNotEqualresponse)Taghipour/embedding_9/NotEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
Taghipour/embedding_9/NotEqual?
Taghipour/pos_x_maskedout/CastCast"Taghipour/embedding_9/NotEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2 
Taghipour/pos_x_maskedout/Cast?
(Taghipour/pos_x_maskedout/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(Taghipour/pos_x_maskedout/ExpandDims/dim?
$Taghipour/pos_x_maskedout/ExpandDims
ExpandDims"Taghipour/pos_x_maskedout/Cast:y:01Taghipour/pos_x_maskedout/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2&
$Taghipour/pos_x_maskedout/ExpandDims?
Taghipour/pos_x_maskedout/stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2!
Taghipour/pos_x_maskedout/stack?
Taghipour/pos_x_maskedout/TileTile-Taghipour/pos_x_maskedout/ExpandDims:output:0(Taghipour/pos_x_maskedout/stack:output:0*
T0*-
_output_shapes
:???????????2 
Taghipour/pos_x_maskedout/Tile?
(Taghipour/pos_x_maskedout/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(Taghipour/pos_x_maskedout/transpose/perm?
#Taghipour/pos_x_maskedout/transpose	Transpose'Taghipour/pos_x_maskedout/Tile:output:01Taghipour/pos_x_maskedout/transpose/perm:output:0*
T0*-
_output_shapes
:???????????2%
#Taghipour/pos_x_maskedout/transpose?
Taghipour/pos_x_maskedout/mulMul:Taghipour/embedding_9/embedding_lookup/Identity_1:output:0'Taghipour/pos_x_maskedout/transpose:y:0*
T0*-
_output_shapes
:???????????2
Taghipour/pos_x_maskedout/mul?
(Taghipour/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(Taghipour/conv1d_5/conv1d/ExpandDims/dim?
$Taghipour/conv1d_5/conv1d/ExpandDims
ExpandDims!Taghipour/pos_x_maskedout/mul:z:01Taghipour/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2&
$Taghipour/conv1d_5/conv1d/ExpandDims?
5Taghipour/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>taghipour_conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?2*
dtype027
5Taghipour/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?
*Taghipour/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Taghipour/conv1d_5/conv1d/ExpandDims_1/dim?
&Taghipour/conv1d_5/conv1d/ExpandDims_1
ExpandDims=Taghipour/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:03Taghipour/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?22(
&Taghipour/conv1d_5/conv1d/ExpandDims_1?
Taghipour/conv1d_5/conv1dConv2D-Taghipour/conv1d_5/conv1d/ExpandDims:output:0/Taghipour/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????2*
paddingVALID*
strides
2
Taghipour/conv1d_5/conv1d?
!Taghipour/conv1d_5/conv1d/SqueezeSqueeze"Taghipour/conv1d_5/conv1d:output:0*
T0*,
_output_shapes
:??????????2*
squeeze_dims

?????????2#
!Taghipour/conv1d_5/conv1d/Squeeze?
)Taghipour/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp2taghipour_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)Taghipour/conv1d_5/BiasAdd/ReadVariableOp?
Taghipour/conv1d_5/BiasAddBiasAdd*Taghipour/conv1d_5/conv1d/Squeeze:output:01Taghipour/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????22
Taghipour/conv1d_5/BiasAdd?
Taghipour/lstm_3/ShapeShape#Taghipour/conv1d_5/BiasAdd:output:0*
T0*
_output_shapes
:2
Taghipour/lstm_3/Shape?
$Taghipour/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Taghipour/lstm_3/strided_slice/stack?
&Taghipour/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Taghipour/lstm_3/strided_slice/stack_1?
&Taghipour/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Taghipour/lstm_3/strided_slice/stack_2?
Taghipour/lstm_3/strided_sliceStridedSliceTaghipour/lstm_3/Shape:output:0-Taghipour/lstm_3/strided_slice/stack:output:0/Taghipour/lstm_3/strided_slice/stack_1:output:0/Taghipour/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Taghipour/lstm_3/strided_slice
Taghipour/lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
Taghipour/lstm_3/zeros/mul/y?
Taghipour/lstm_3/zeros/mulMul'Taghipour/lstm_3/strided_slice:output:0%Taghipour/lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
Taghipour/lstm_3/zeros/mul?
Taghipour/lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
Taghipour/lstm_3/zeros/Less/y?
Taghipour/lstm_3/zeros/LessLessTaghipour/lstm_3/zeros/mul:z:0&Taghipour/lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
Taghipour/lstm_3/zeros/Less?
Taghipour/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2!
Taghipour/lstm_3/zeros/packed/1?
Taghipour/lstm_3/zeros/packedPack'Taghipour/lstm_3/strided_slice:output:0(Taghipour/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
Taghipour/lstm_3/zeros/packed?
Taghipour/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Taghipour/lstm_3/zeros/Const?
Taghipour/lstm_3/zerosFill&Taghipour/lstm_3/zeros/packed:output:0%Taghipour/lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
Taghipour/lstm_3/zeros?
Taghipour/lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2 
Taghipour/lstm_3/zeros_1/mul/y?
Taghipour/lstm_3/zeros_1/mulMul'Taghipour/lstm_3/strided_slice:output:0'Taghipour/lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
Taghipour/lstm_3/zeros_1/mul?
Taghipour/lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
Taghipour/lstm_3/zeros_1/Less/y?
Taghipour/lstm_3/zeros_1/LessLess Taghipour/lstm_3/zeros_1/mul:z:0(Taghipour/lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
Taghipour/lstm_3/zeros_1/Less?
!Taghipour/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2#
!Taghipour/lstm_3/zeros_1/packed/1?
Taghipour/lstm_3/zeros_1/packedPack'Taghipour/lstm_3/strided_slice:output:0*Taghipour/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
Taghipour/lstm_3/zeros_1/packed?
Taghipour/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
Taghipour/lstm_3/zeros_1/Const?
Taghipour/lstm_3/zeros_1Fill(Taghipour/lstm_3/zeros_1/packed:output:0'Taghipour/lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
Taghipour/lstm_3/zeros_1?
Taghipour/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
Taghipour/lstm_3/transpose/perm?
Taghipour/lstm_3/transpose	Transpose#Taghipour/conv1d_5/BiasAdd:output:0(Taghipour/lstm_3/transpose/perm:output:0*
T0*,
_output_shapes
:??????????22
Taghipour/lstm_3/transpose?
Taghipour/lstm_3/Shape_1ShapeTaghipour/lstm_3/transpose:y:0*
T0*
_output_shapes
:2
Taghipour/lstm_3/Shape_1?
&Taghipour/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Taghipour/lstm_3/strided_slice_1/stack?
(Taghipour/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Taghipour/lstm_3/strided_slice_1/stack_1?
(Taghipour/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Taghipour/lstm_3/strided_slice_1/stack_2?
 Taghipour/lstm_3/strided_slice_1StridedSlice!Taghipour/lstm_3/Shape_1:output:0/Taghipour/lstm_3/strided_slice_1/stack:output:01Taghipour/lstm_3/strided_slice_1/stack_1:output:01Taghipour/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Taghipour/lstm_3/strided_slice_1?
,Taghipour/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,Taghipour/lstm_3/TensorArrayV2/element_shape?
Taghipour/lstm_3/TensorArrayV2TensorListReserve5Taghipour/lstm_3/TensorArrayV2/element_shape:output:0)Taghipour/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
Taghipour/lstm_3/TensorArrayV2?
FTaghipour/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2H
FTaghipour/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape?
8Taghipour/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorTaghipour/lstm_3/transpose:y:0OTaghipour/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8Taghipour/lstm_3/TensorArrayUnstack/TensorListFromTensor?
&Taghipour/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Taghipour/lstm_3/strided_slice_2/stack?
(Taghipour/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Taghipour/lstm_3/strided_slice_2/stack_1?
(Taghipour/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Taghipour/lstm_3/strided_slice_2/stack_2?
 Taghipour/lstm_3/strided_slice_2StridedSliceTaghipour/lstm_3/transpose:y:0/Taghipour/lstm_3/strided_slice_2/stack:output:01Taghipour/lstm_3/strided_slice_2/stack_1:output:01Taghipour/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2"
 Taghipour/lstm_3/strided_slice_2?
,Taghipour/lstm_3/lstm_cell_3/ones_like/ShapeShape)Taghipour/lstm_3/strided_slice_2:output:0*
T0*
_output_shapes
:2.
,Taghipour/lstm_3/lstm_cell_3/ones_like/Shape?
,Taghipour/lstm_3/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,Taghipour/lstm_3/lstm_cell_3/ones_like/Const?
&Taghipour/lstm_3/lstm_cell_3/ones_likeFill5Taghipour/lstm_3/lstm_cell_3/ones_like/Shape:output:05Taghipour/lstm_3/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????22(
&Taghipour/lstm_3/lstm_cell_3/ones_like?
.Taghipour/lstm_3/lstm_cell_3/ones_like_1/ShapeShapeTaghipour/lstm_3/zeros:output:0*
T0*
_output_shapes
:20
.Taghipour/lstm_3/lstm_cell_3/ones_like_1/Shape?
.Taghipour/lstm_3/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.Taghipour/lstm_3/lstm_cell_3/ones_like_1/Const?
(Taghipour/lstm_3/lstm_cell_3/ones_like_1Fill7Taghipour/lstm_3/lstm_cell_3/ones_like_1/Shape:output:07Taghipour/lstm_3/lstm_cell_3/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????2*
(Taghipour/lstm_3/lstm_cell_3/ones_like_1?
 Taghipour/lstm_3/lstm_cell_3/mulMul)Taghipour/lstm_3/strided_slice_2:output:0/Taghipour/lstm_3/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22"
 Taghipour/lstm_3/lstm_cell_3/mul?
"Taghipour/lstm_3/lstm_cell_3/mul_1Mul)Taghipour/lstm_3/strided_slice_2:output:0/Taghipour/lstm_3/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22$
"Taghipour/lstm_3/lstm_cell_3/mul_1?
"Taghipour/lstm_3/lstm_cell_3/mul_2Mul)Taghipour/lstm_3/strided_slice_2:output:0/Taghipour/lstm_3/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22$
"Taghipour/lstm_3/lstm_cell_3/mul_2?
"Taghipour/lstm_3/lstm_cell_3/mul_3Mul)Taghipour/lstm_3/strided_slice_2:output:0/Taghipour/lstm_3/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:?????????22$
"Taghipour/lstm_3/lstm_cell_3/mul_3?
,Taghipour/lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,Taghipour/lstm_3/lstm_cell_3/split/split_dim?
1Taghipour/lstm_3/lstm_cell_3/split/ReadVariableOpReadVariableOp:taghipour_lstm_3_lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	2?	*
dtype023
1Taghipour/lstm_3/lstm_cell_3/split/ReadVariableOp?
"Taghipour/lstm_3/lstm_cell_3/splitSplit5Taghipour/lstm_3/lstm_cell_3/split/split_dim:output:09Taghipour/lstm_3/lstm_cell_3/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	2?:	2?:	2?:	2?*
	num_split2$
"Taghipour/lstm_3/lstm_cell_3/split?
#Taghipour/lstm_3/lstm_cell_3/MatMulMatMul$Taghipour/lstm_3/lstm_cell_3/mul:z:0+Taghipour/lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:??????????2%
#Taghipour/lstm_3/lstm_cell_3/MatMul?
%Taghipour/lstm_3/lstm_cell_3/MatMul_1MatMul&Taghipour/lstm_3/lstm_cell_3/mul_1:z:0+Taghipour/lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:??????????2'
%Taghipour/lstm_3/lstm_cell_3/MatMul_1?
%Taghipour/lstm_3/lstm_cell_3/MatMul_2MatMul&Taghipour/lstm_3/lstm_cell_3/mul_2:z:0+Taghipour/lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:??????????2'
%Taghipour/lstm_3/lstm_cell_3/MatMul_2?
%Taghipour/lstm_3/lstm_cell_3/MatMul_3MatMul&Taghipour/lstm_3/lstm_cell_3/mul_3:z:0+Taghipour/lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????2'
%Taghipour/lstm_3/lstm_cell_3/MatMul_3?
.Taghipour/lstm_3/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.Taghipour/lstm_3/lstm_cell_3/split_1/split_dim?
3Taghipour/lstm_3/lstm_cell_3/split_1/ReadVariableOpReadVariableOp<taghipour_lstm_3_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?	*
dtype025
3Taghipour/lstm_3/lstm_cell_3/split_1/ReadVariableOp?
$Taghipour/lstm_3/lstm_cell_3/split_1Split7Taghipour/lstm_3/lstm_cell_3/split_1/split_dim:output:0;Taghipour/lstm_3/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2&
$Taghipour/lstm_3/lstm_cell_3/split_1?
$Taghipour/lstm_3/lstm_cell_3/BiasAddBiasAdd-Taghipour/lstm_3/lstm_cell_3/MatMul:product:0-Taghipour/lstm_3/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:??????????2&
$Taghipour/lstm_3/lstm_cell_3/BiasAdd?
&Taghipour/lstm_3/lstm_cell_3/BiasAdd_1BiasAdd/Taghipour/lstm_3/lstm_cell_3/MatMul_1:product:0-Taghipour/lstm_3/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:??????????2(
&Taghipour/lstm_3/lstm_cell_3/BiasAdd_1?
&Taghipour/lstm_3/lstm_cell_3/BiasAdd_2BiasAdd/Taghipour/lstm_3/lstm_cell_3/MatMul_2:product:0-Taghipour/lstm_3/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:??????????2(
&Taghipour/lstm_3/lstm_cell_3/BiasAdd_2?
&Taghipour/lstm_3/lstm_cell_3/BiasAdd_3BiasAdd/Taghipour/lstm_3/lstm_cell_3/MatMul_3:product:0-Taghipour/lstm_3/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:??????????2(
&Taghipour/lstm_3/lstm_cell_3/BiasAdd_3?
"Taghipour/lstm_3/lstm_cell_3/mul_4MulTaghipour/lstm_3/zeros:output:01Taghipour/lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/mul_4?
"Taghipour/lstm_3/lstm_cell_3/mul_5MulTaghipour/lstm_3/zeros:output:01Taghipour/lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/mul_5?
"Taghipour/lstm_3/lstm_cell_3/mul_6MulTaghipour/lstm_3/zeros:output:01Taghipour/lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/mul_6?
"Taghipour/lstm_3/lstm_cell_3/mul_7MulTaghipour/lstm_3/zeros:output:01Taghipour/lstm_3/lstm_cell_3/ones_like_1:output:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/mul_7?
+Taghipour/lstm_3/lstm_cell_3/ReadVariableOpReadVariableOp4taghipour_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02-
+Taghipour/lstm_3/lstm_cell_3/ReadVariableOp?
0Taghipour/lstm_3/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0Taghipour/lstm_3/lstm_cell_3/strided_slice/stack?
2Taghipour/lstm_3/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  24
2Taghipour/lstm_3/lstm_cell_3/strided_slice/stack_1?
2Taghipour/lstm_3/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2Taghipour/lstm_3/lstm_cell_3/strided_slice/stack_2?
*Taghipour/lstm_3/lstm_cell_3/strided_sliceStridedSlice3Taghipour/lstm_3/lstm_cell_3/ReadVariableOp:value:09Taghipour/lstm_3/lstm_cell_3/strided_slice/stack:output:0;Taghipour/lstm_3/lstm_cell_3/strided_slice/stack_1:output:0;Taghipour/lstm_3/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2,
*Taghipour/lstm_3/lstm_cell_3/strided_slice?
%Taghipour/lstm_3/lstm_cell_3/MatMul_4MatMul&Taghipour/lstm_3/lstm_cell_3/mul_4:z:03Taghipour/lstm_3/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:??????????2'
%Taghipour/lstm_3/lstm_cell_3/MatMul_4?
 Taghipour/lstm_3/lstm_cell_3/addAddV2-Taghipour/lstm_3/lstm_cell_3/BiasAdd:output:0/Taghipour/lstm_3/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2"
 Taghipour/lstm_3/lstm_cell_3/add?
$Taghipour/lstm_3/lstm_cell_3/SigmoidSigmoid$Taghipour/lstm_3/lstm_cell_3/add:z:0*
T0*(
_output_shapes
:??????????2&
$Taghipour/lstm_3/lstm_cell_3/Sigmoid?
-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_1ReadVariableOp4taghipour_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02/
-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_1?
2Taghipour/lstm_3/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  24
2Taghipour/lstm_3/lstm_cell_3/strided_slice_1/stack?
4Taghipour/lstm_3/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  26
4Taghipour/lstm_3/lstm_cell_3/strided_slice_1/stack_1?
4Taghipour/lstm_3/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4Taghipour/lstm_3/lstm_cell_3/strided_slice_1/stack_2?
,Taghipour/lstm_3/lstm_cell_3/strided_slice_1StridedSlice5Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_1:value:0;Taghipour/lstm_3/lstm_cell_3/strided_slice_1/stack:output:0=Taghipour/lstm_3/lstm_cell_3/strided_slice_1/stack_1:output:0=Taghipour/lstm_3/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2.
,Taghipour/lstm_3/lstm_cell_3/strided_slice_1?
%Taghipour/lstm_3/lstm_cell_3/MatMul_5MatMul&Taghipour/lstm_3/lstm_cell_3/mul_5:z:05Taghipour/lstm_3/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2'
%Taghipour/lstm_3/lstm_cell_3/MatMul_5?
"Taghipour/lstm_3/lstm_cell_3/add_1AddV2/Taghipour/lstm_3/lstm_cell_3/BiasAdd_1:output:0/Taghipour/lstm_3/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/add_1?
&Taghipour/lstm_3/lstm_cell_3/Sigmoid_1Sigmoid&Taghipour/lstm_3/lstm_cell_3/add_1:z:0*
T0*(
_output_shapes
:??????????2(
&Taghipour/lstm_3/lstm_cell_3/Sigmoid_1?
"Taghipour/lstm_3/lstm_cell_3/mul_8Mul*Taghipour/lstm_3/lstm_cell_3/Sigmoid_1:y:0!Taghipour/lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/mul_8?
-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_2ReadVariableOp4taghipour_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02/
-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_2?
2Taghipour/lstm_3/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  24
2Taghipour/lstm_3/lstm_cell_3/strided_slice_2/stack?
4Taghipour/lstm_3/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  26
4Taghipour/lstm_3/lstm_cell_3/strided_slice_2/stack_1?
4Taghipour/lstm_3/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4Taghipour/lstm_3/lstm_cell_3/strided_slice_2/stack_2?
,Taghipour/lstm_3/lstm_cell_3/strided_slice_2StridedSlice5Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_2:value:0;Taghipour/lstm_3/lstm_cell_3/strided_slice_2/stack:output:0=Taghipour/lstm_3/lstm_cell_3/strided_slice_2/stack_1:output:0=Taghipour/lstm_3/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2.
,Taghipour/lstm_3/lstm_cell_3/strided_slice_2?
%Taghipour/lstm_3/lstm_cell_3/MatMul_6MatMul&Taghipour/lstm_3/lstm_cell_3/mul_6:z:05Taghipour/lstm_3/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2'
%Taghipour/lstm_3/lstm_cell_3/MatMul_6?
"Taghipour/lstm_3/lstm_cell_3/add_2AddV2/Taghipour/lstm_3/lstm_cell_3/BiasAdd_2:output:0/Taghipour/lstm_3/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/add_2?
!Taghipour/lstm_3/lstm_cell_3/TanhTanh&Taghipour/lstm_3/lstm_cell_3/add_2:z:0*
T0*(
_output_shapes
:??????????2#
!Taghipour/lstm_3/lstm_cell_3/Tanh?
"Taghipour/lstm_3/lstm_cell_3/mul_9Mul(Taghipour/lstm_3/lstm_cell_3/Sigmoid:y:0%Taghipour/lstm_3/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/mul_9?
"Taghipour/lstm_3/lstm_cell_3/add_3AddV2&Taghipour/lstm_3/lstm_cell_3/mul_8:z:0&Taghipour/lstm_3/lstm_cell_3/mul_9:z:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/add_3?
-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_3ReadVariableOp4taghipour_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??	*
dtype02/
-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_3?
2Taghipour/lstm_3/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  24
2Taghipour/lstm_3/lstm_cell_3/strided_slice_3/stack?
4Taghipour/lstm_3/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        26
4Taghipour/lstm_3/lstm_cell_3/strided_slice_3/stack_1?
4Taghipour/lstm_3/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4Taghipour/lstm_3/lstm_cell_3/strided_slice_3/stack_2?
,Taghipour/lstm_3/lstm_cell_3/strided_slice_3StridedSlice5Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_3:value:0;Taghipour/lstm_3/lstm_cell_3/strided_slice_3/stack:output:0=Taghipour/lstm_3/lstm_cell_3/strided_slice_3/stack_1:output:0=Taghipour/lstm_3/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2.
,Taghipour/lstm_3/lstm_cell_3/strided_slice_3?
%Taghipour/lstm_3/lstm_cell_3/MatMul_7MatMul&Taghipour/lstm_3/lstm_cell_3/mul_7:z:05Taghipour/lstm_3/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2'
%Taghipour/lstm_3/lstm_cell_3/MatMul_7?
"Taghipour/lstm_3/lstm_cell_3/add_4AddV2/Taghipour/lstm_3/lstm_cell_3/BiasAdd_3:output:0/Taghipour/lstm_3/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2$
"Taghipour/lstm_3/lstm_cell_3/add_4?
&Taghipour/lstm_3/lstm_cell_3/Sigmoid_2Sigmoid&Taghipour/lstm_3/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:??????????2(
&Taghipour/lstm_3/lstm_cell_3/Sigmoid_2?
#Taghipour/lstm_3/lstm_cell_3/Tanh_1Tanh&Taghipour/lstm_3/lstm_cell_3/add_3:z:0*
T0*(
_output_shapes
:??????????2%
#Taghipour/lstm_3/lstm_cell_3/Tanh_1?
#Taghipour/lstm_3/lstm_cell_3/mul_10Mul*Taghipour/lstm_3/lstm_cell_3/Sigmoid_2:y:0'Taghipour/lstm_3/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????2%
#Taghipour/lstm_3/lstm_cell_3/mul_10?
.Taghipour/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  20
.Taghipour/lstm_3/TensorArrayV2_1/element_shape?
 Taghipour/lstm_3/TensorArrayV2_1TensorListReserve7Taghipour/lstm_3/TensorArrayV2_1/element_shape:output:0)Taghipour/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 Taghipour/lstm_3/TensorArrayV2_1p
Taghipour/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
Taghipour/lstm_3/time?
)Taghipour/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)Taghipour/lstm_3/while/maximum_iterations?
#Taghipour/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#Taghipour/lstm_3/while/loop_counter?
Taghipour/lstm_3/whileWhile,Taghipour/lstm_3/while/loop_counter:output:02Taghipour/lstm_3/while/maximum_iterations:output:0Taghipour/lstm_3/time:output:0)Taghipour/lstm_3/TensorArrayV2_1:handle:0Taghipour/lstm_3/zeros:output:0!Taghipour/lstm_3/zeros_1:output:0)Taghipour/lstm_3/strided_slice_1:output:0HTaghipour/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0:taghipour_lstm_3_lstm_cell_3_split_readvariableop_resource<taghipour_lstm_3_lstm_cell_3_split_1_readvariableop_resource4taghipour_lstm_3_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!Taghipour_lstm_3_while_body_73600*-
cond%R#
!Taghipour_lstm_3_while_cond_73599*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
Taghipour/lstm_3/while?
ATaghipour/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2C
ATaghipour/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape?
3Taghipour/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStackTaghipour/lstm_3/while:output:3JTaghipour/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype025
3Taghipour/lstm_3/TensorArrayV2Stack/TensorListStack?
&Taghipour/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&Taghipour/lstm_3/strided_slice_3/stack?
(Taghipour/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(Taghipour/lstm_3/strided_slice_3/stack_1?
(Taghipour/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Taghipour/lstm_3/strided_slice_3/stack_2?
 Taghipour/lstm_3/strided_slice_3StridedSlice<Taghipour/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0/Taghipour/lstm_3/strided_slice_3/stack:output:01Taghipour/lstm_3/strided_slice_3/stack_1:output:01Taghipour/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2"
 Taghipour/lstm_3/strided_slice_3?
!Taghipour/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!Taghipour/lstm_3/transpose_1/perm?
Taghipour/lstm_3/transpose_1	Transpose<Taghipour/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0*Taghipour/lstm_3/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2
Taghipour/lstm_3/transpose_1?
Taghipour/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
Taghipour/lstm_3/runtime?
Taghipour/dropout_4/IdentityIdentity Taghipour/lstm_3/transpose_1:y:0*
T0*-
_output_shapes
:???????????2
Taghipour/dropout_4/Identity?
Taghipour/attention_3/ShapeShape%Taghipour/dropout_4/Identity:output:0*
T0*
_output_shapes
:2
Taghipour/attention_3/Shape?
Taghipour/attention_3/unstackUnpack$Taghipour/attention_3/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
Taghipour/attention_3/unstack?
,Taghipour/attention_3/Shape_1/ReadVariableOpReadVariableOp5taghipour_attention_3_shape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,Taghipour/attention_3/Shape_1/ReadVariableOp?
Taghipour/attention_3/Shape_1Const*
_output_shapes
:*
dtype0*
valueB",  ,  2
Taghipour/attention_3/Shape_1?
Taghipour/attention_3/unstack_1Unpack&Taghipour/attention_3/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2!
Taghipour/attention_3/unstack_1?
#Taghipour/attention_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????,  2%
#Taghipour/attention_3/Reshape/shape?
Taghipour/attention_3/ReshapeReshape%Taghipour/dropout_4/Identity:output:0,Taghipour/attention_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
Taghipour/attention_3/Reshape?
.Taghipour/attention_3/transpose/ReadVariableOpReadVariableOp5taghipour_attention_3_shape_1_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.Taghipour/attention_3/transpose/ReadVariableOp?
$Taghipour/attention_3/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2&
$Taghipour/attention_3/transpose/perm?
Taghipour/attention_3/transpose	Transpose6Taghipour/attention_3/transpose/ReadVariableOp:value:0-Taghipour/attention_3/transpose/perm:output:0*
T0* 
_output_shapes
:
??2!
Taghipour/attention_3/transpose?
%Taghipour/attention_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB",  ????2'
%Taghipour/attention_3/Reshape_1/shape?
Taghipour/attention_3/Reshape_1Reshape#Taghipour/attention_3/transpose:y:0.Taghipour/attention_3/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
??2!
Taghipour/attention_3/Reshape_1?
Taghipour/attention_3/MatMulMatMul&Taghipour/attention_3/Reshape:output:0(Taghipour/attention_3/Reshape_1:output:0*
T0*(
_output_shapes
:??????????2
Taghipour/attention_3/MatMul?
'Taghipour/attention_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2)
'Taghipour/attention_3/Reshape_2/shape/1?
'Taghipour/attention_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2)
'Taghipour/attention_3/Reshape_2/shape/2?
%Taghipour/attention_3/Reshape_2/shapePack&Taghipour/attention_3/unstack:output:00Taghipour/attention_3/Reshape_2/shape/1:output:00Taghipour/attention_3/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%Taghipour/attention_3/Reshape_2/shape?
Taghipour/attention_3/Reshape_2Reshape&Taghipour/attention_3/MatMul:product:0.Taghipour/attention_3/Reshape_2/shape:output:0*
T0*-
_output_shapes
:???????????2!
Taghipour/attention_3/Reshape_2?
Taghipour/attention_3/TanhTanh(Taghipour/attention_3/Reshape_2:output:0*
T0*-
_output_shapes
:???????????2
Taghipour/attention_3/Tanh?
.Taghipour/attention_3/Tensordot/ReadVariableOpReadVariableOp7taghipour_attention_3_tensordot_readvariableop_resource*
_output_shapes	
:?*
dtype020
.Taghipour/attention_3/Tensordot/ReadVariableOp?
-Taghipour/attention_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  2/
-Taghipour/attention_3/Tensordot/Reshape/shape?
'Taghipour/attention_3/Tensordot/ReshapeReshape6Taghipour/attention_3/Tensordot/ReadVariableOp:value:06Taghipour/attention_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	?2)
'Taghipour/attention_3/Tensordot/Reshape?
$Taghipour/attention_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$Taghipour/attention_3/Tensordot/axes?
$Taghipour/attention_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$Taghipour/attention_3/Tensordot/free?
%Taghipour/attention_3/Tensordot/ShapeShapeTaghipour/attention_3/Tanh:y:0*
T0*
_output_shapes
:2'
%Taghipour/attention_3/Tensordot/Shape?
-Taghipour/attention_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-Taghipour/attention_3/Tensordot/GatherV2/axis?
(Taghipour/attention_3/Tensordot/GatherV2GatherV2.Taghipour/attention_3/Tensordot/Shape:output:0-Taghipour/attention_3/Tensordot/free:output:06Taghipour/attention_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(Taghipour/attention_3/Tensordot/GatherV2?
/Taghipour/attention_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/Taghipour/attention_3/Tensordot/GatherV2_1/axis?
*Taghipour/attention_3/Tensordot/GatherV2_1GatherV2.Taghipour/attention_3/Tensordot/Shape:output:0-Taghipour/attention_3/Tensordot/axes:output:08Taghipour/attention_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*Taghipour/attention_3/Tensordot/GatherV2_1?
%Taghipour/attention_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Taghipour/attention_3/Tensordot/Const?
$Taghipour/attention_3/Tensordot/ProdProd1Taghipour/attention_3/Tensordot/GatherV2:output:0.Taghipour/attention_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$Taghipour/attention_3/Tensordot/Prod?
'Taghipour/attention_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'Taghipour/attention_3/Tensordot/Const_1?
&Taghipour/attention_3/Tensordot/Prod_1Prod3Taghipour/attention_3/Tensordot/GatherV2_1:output:00Taghipour/attention_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&Taghipour/attention_3/Tensordot/Prod_1?
+Taghipour/attention_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+Taghipour/attention_3/Tensordot/concat/axis?
&Taghipour/attention_3/Tensordot/concatConcatV2-Taghipour/attention_3/Tensordot/axes:output:0-Taghipour/attention_3/Tensordot/free:output:04Taghipour/attention_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&Taghipour/attention_3/Tensordot/concat?
%Taghipour/attention_3/Tensordot/stackPack/Taghipour/attention_3/Tensordot/Prod_1:output:0-Taghipour/attention_3/Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:2'
%Taghipour/attention_3/Tensordot/stack?
)Taghipour/attention_3/Tensordot/transpose	TransposeTaghipour/attention_3/Tanh:y:0/Taghipour/attention_3/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2+
)Taghipour/attention_3/Tensordot/transpose?
)Taghipour/attention_3/Tensordot/Reshape_1Reshape-Taghipour/attention_3/Tensordot/transpose:y:0.Taghipour/attention_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)Taghipour/attention_3/Tensordot/Reshape_1?
&Taghipour/attention_3/Tensordot/MatMulMatMul0Taghipour/attention_3/Tensordot/Reshape:output:02Taghipour/attention_3/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2(
&Taghipour/attention_3/Tensordot/MatMul?
'Taghipour/attention_3/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2)
'Taghipour/attention_3/Tensordot/Const_2?
-Taghipour/attention_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-Taghipour/attention_3/Tensordot/concat_1/axis?
(Taghipour/attention_3/Tensordot/concat_1ConcatV20Taghipour/attention_3/Tensordot/Const_2:output:01Taghipour/attention_3/Tensordot/GatherV2:output:06Taghipour/attention_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(Taghipour/attention_3/Tensordot/concat_1?
Taghipour/attention_3/TensordotReshape0Taghipour/attention_3/Tensordot/MatMul:product:01Taghipour/attention_3/Tensordot/concat_1:output:0*
T0*(
_output_shapes
:??????????2!
Taghipour/attention_3/Tensordot?
Taghipour/attention_3/SoftmaxSoftmax(Taghipour/attention_3/Tensordot:output:0*
T0*(
_output_shapes
:??????????2
Taghipour/attention_3/Softmax?
$Taghipour/attention_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$Taghipour/attention_3/ExpandDims/dim?
 Taghipour/attention_3/ExpandDims
ExpandDims'Taghipour/attention_3/Softmax:softmax:0-Taghipour/attention_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2"
 Taghipour/attention_3/ExpandDims?
Taghipour/attention_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"   ,     2
Taghipour/attention_3/stack?
Taghipour/attention_3/TileTile)Taghipour/attention_3/ExpandDims:output:0$Taghipour/attention_3/stack:output:0*
T0*-
_output_shapes
:???????????2
Taghipour/attention_3/Tile?
&Taghipour/attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&Taghipour/attention_3/transpose_1/perm?
!Taghipour/attention_3/transpose_1	Transpose#Taghipour/attention_3/Tile:output:0/Taghipour/attention_3/transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????2#
!Taghipour/attention_3/transpose_1?
Taghipour/attention_3/mulMul%Taghipour/dropout_4/Identity:output:0%Taghipour/attention_3/transpose_1:y:0*
T0*-
_output_shapes
:???????????2
Taghipour/attention_3/mul?
+Taghipour/attention_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+Taghipour/attention_3/Sum/reduction_indices?
Taghipour/attention_3/SumSumTaghipour/attention_3/mul:z:04Taghipour/attention_3/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
Taghipour/attention_3/Sum?
'Taghipour/dense_5/MatMul/ReadVariableOpReadVariableOp0taghipour_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02)
'Taghipour/dense_5/MatMul/ReadVariableOp?
Taghipour/dense_5/MatMulMatMul"Taghipour/attention_3/Sum:output:0/Taghipour/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Taghipour/dense_5/MatMul?
(Taghipour/dense_5/BiasAdd/ReadVariableOpReadVariableOp1taghipour_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Taghipour/dense_5/BiasAdd/ReadVariableOp?
Taghipour/dense_5/BiasAddBiasAdd"Taghipour/dense_5/MatMul:product:00Taghipour/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Taghipour/dense_5/BiasAdd?
'Taghipour/dense_6/MatMul/ReadVariableOpReadVariableOp0taghipour_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Taghipour/dense_6/MatMul/ReadVariableOp?
Taghipour/dense_6/MatMulMatMulapplied/Taghipour/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Taghipour/dense_6/MatMul?
(Taghipour/dense_6/BiasAdd/ReadVariableOpReadVariableOp1taghipour_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Taghipour/dense_6/BiasAdd/ReadVariableOp?
Taghipour/dense_6/BiasAddBiasAdd"Taghipour/dense_6/MatMul:product:00Taghipour/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Taghipour/dense_6/BiasAdd?
#Taghipour/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#Taghipour/concatenate_1/concat/axis?
Taghipour/concatenate_1/concatConcatV2"Taghipour/dense_5/BiasAdd:output:0"Taghipour/dense_6/BiasAdd:output:0,Taghipour/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????P2 
Taghipour/concatenate_1/concat?
Taghipour/dropout_5/IdentityIdentity'Taghipour/concatenate_1/concat:output:0*
T0*'
_output_shapes
:?????????P2
Taghipour/dropout_5/Identity?
'Taghipour/dense_7/MatMul/ReadVariableOpReadVariableOp0taghipour_dense_7_matmul_readvariableop_resource*
_output_shapes

:P *
dtype02)
'Taghipour/dense_7/MatMul/ReadVariableOp?
Taghipour/dense_7/MatMulMatMul%Taghipour/dropout_5/Identity:output:0/Taghipour/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Taghipour/dense_7/MatMul?
(Taghipour/dense_7/BiasAdd/ReadVariableOpReadVariableOp1taghipour_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Taghipour/dense_7/BiasAdd/ReadVariableOp?
Taghipour/dense_7/BiasAddBiasAdd"Taghipour/dense_7/MatMul:product:00Taghipour/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Taghipour/dense_7/BiasAdd?
%Taghipour/score/MatMul/ReadVariableOpReadVariableOp.taghipour_score_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%Taghipour/score/MatMul/ReadVariableOp?
Taghipour/score/MatMulMatMul"Taghipour/dense_7/BiasAdd:output:0-Taghipour/score/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Taghipour/score/MatMul?
&Taghipour/score/BiasAdd/ReadVariableOpReadVariableOp/taghipour_score_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&Taghipour/score/BiasAdd/ReadVariableOp?
Taghipour/score/BiasAddBiasAdd Taghipour/score/MatMul:product:0.Taghipour/score/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Taghipour/score/BiasAdd?
Taghipour/score/SoftmaxSoftmax Taghipour/score/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Taghipour/score/Softmax|
IdentityIdentity!Taghipour/score/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp/^Taghipour/attention_3/Tensordot/ReadVariableOp/^Taghipour/attention_3/transpose/ReadVariableOp*^Taghipour/conv1d_5/BiasAdd/ReadVariableOp6^Taghipour/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp)^Taghipour/dense_5/BiasAdd/ReadVariableOp(^Taghipour/dense_5/MatMul/ReadVariableOp)^Taghipour/dense_6/BiasAdd/ReadVariableOp(^Taghipour/dense_6/MatMul/ReadVariableOp)^Taghipour/dense_7/BiasAdd/ReadVariableOp(^Taghipour/dense_7/MatMul/ReadVariableOp'^Taghipour/embedding_9/embedding_lookup,^Taghipour/lstm_3/lstm_cell_3/ReadVariableOp.^Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_1.^Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_2.^Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_32^Taghipour/lstm_3/lstm_cell_3/split/ReadVariableOp4^Taghipour/lstm_3/lstm_cell_3/split_1/ReadVariableOp^Taghipour/lstm_3/while'^Taghipour/score/BiasAdd/ReadVariableOp&^Taghipour/score/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:??????????:?????????: : : : : : : : : : : : : : : : 2`
.Taghipour/attention_3/Tensordot/ReadVariableOp.Taghipour/attention_3/Tensordot/ReadVariableOp2`
.Taghipour/attention_3/transpose/ReadVariableOp.Taghipour/attention_3/transpose/ReadVariableOp2V
)Taghipour/conv1d_5/BiasAdd/ReadVariableOp)Taghipour/conv1d_5/BiasAdd/ReadVariableOp2n
5Taghipour/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp5Taghipour/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2T
(Taghipour/dense_5/BiasAdd/ReadVariableOp(Taghipour/dense_5/BiasAdd/ReadVariableOp2R
'Taghipour/dense_5/MatMul/ReadVariableOp'Taghipour/dense_5/MatMul/ReadVariableOp2T
(Taghipour/dense_6/BiasAdd/ReadVariableOp(Taghipour/dense_6/BiasAdd/ReadVariableOp2R
'Taghipour/dense_6/MatMul/ReadVariableOp'Taghipour/dense_6/MatMul/ReadVariableOp2T
(Taghipour/dense_7/BiasAdd/ReadVariableOp(Taghipour/dense_7/BiasAdd/ReadVariableOp2R
'Taghipour/dense_7/MatMul/ReadVariableOp'Taghipour/dense_7/MatMul/ReadVariableOp2P
&Taghipour/embedding_9/embedding_lookup&Taghipour/embedding_9/embedding_lookup2Z
+Taghipour/lstm_3/lstm_cell_3/ReadVariableOp+Taghipour/lstm_3/lstm_cell_3/ReadVariableOp2^
-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_1-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_12^
-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_2-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_22^
-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_3-Taghipour/lstm_3/lstm_cell_3/ReadVariableOp_32f
1Taghipour/lstm_3/lstm_cell_3/split/ReadVariableOp1Taghipour/lstm_3/lstm_cell_3/split/ReadVariableOp2j
3Taghipour/lstm_3/lstm_cell_3/split_1/ReadVariableOp3Taghipour/lstm_3/lstm_cell_3/split_1/ReadVariableOp20
Taghipour/lstm_3/whileTaghipour/lstm_3/while2P
&Taghipour/score/BiasAdd/ReadVariableOp&Taghipour/score/BiasAdd/ReadVariableOp2N
%Taghipour/score/MatMul/ReadVariableOp%Taghipour/score/MatMul/ReadVariableOp:R N
(
_output_shapes
:??????????
"
_user_specified_name
response:PL
'
_output_shapes
:?????????
!
_user_specified_name	applied
?
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_75164

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????P2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????P2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????P2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
t
H__inference_concatenate_1_layer_call_and_return_conditional_losses_78420
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????P2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????@:?????????:Q M
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
B__inference_dense_6_layer_call_and_return_conditional_losses_75033

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_5_layer_call_fn_78453

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_751642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
applied0
serving_default_applied:0?????????
>
response2
serving_default_response:0??????????9
score0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
	optimizer
loss
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
?

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
%cell
&
state_spec
'	variables
(regularization_losses
)trainable_variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	/att_v
	0att_W
1	variables
2regularization_losses
3trainable_variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
"
_tf_keras_input_layer
?

5kernel
6bias
7	variables
8regularization_losses
9trainable_variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Ikernel
Jbias
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Okernel
Pbias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem? m?/m?0m?5m?6m?;m?<m?Im?Jm?Om?Pm?Zm?[m?\m?v? v?/v?0v?5v?6v?;v?<v?Iv?Jv?Ov?Pv?Zv?[v?\v?"
	optimizer
 "
trackable_dict_wrapper
?
0
1
 2
Z3
[4
\5
/6
07
58
69
;10
<11
I12
J13
O14
P15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
 1
Z2
[3
\4
/5
06
57
68
;9
<10
I11
J12
O13
P14"
trackable_list_wrapper
?
]non_trainable_variables
	variables
^layer_metrics
_metrics
`layer_regularization_losses
regularization_losses
trainable_variables

alayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(
??2embedding_9/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
bnon_trainable_variables
	variables
clayer_metrics
dmetrics
elayer_regularization_losses
regularization_losses
trainable_variables

flayers
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
gnon_trainable_variables
	variables
hlayer_metrics
imetrics
jlayer_regularization_losses
regularization_losses
trainable_variables

klayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$?22conv1d_5/kernel
:22conv1d_5/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
lnon_trainable_variables
!	variables
mlayer_metrics
nmetrics
olayer_regularization_losses
"regularization_losses
#trainable_variables

players
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
q
state_size

Zkernel
[recurrent_kernel
\bias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
Z0
[1
\2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
Z0
[1
\2"
trackable_list_wrapper
?
vnon_trainable_variables
'	variables
wlayer_metrics
xmetrics
ylayer_regularization_losses
(regularization_losses
)trainable_variables

zstates

{layers
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
|non_trainable_variables
+	variables
}layer_metrics
~metrics
layer_regularization_losses
,regularization_losses
-trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :?2attention_3/att_v
%:#
??2attention_3/att_W
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
?non_trainable_variables
1	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
2regularization_losses
3trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?@2dense_5/kernel
:@2dense_5/bias
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
?non_trainable_variables
7	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
8regularization_losses
9trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_6/kernel
:2dense_6/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
?non_trainable_variables
=	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
>regularization_losses
?trainable_variables
?layers
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
?non_trainable_variables
A	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
Bregularization_losses
Ctrainable_variables
?layers
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
?non_trainable_variables
E	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
Fregularization_losses
Gtrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :P 2dense_7/kernel
: 2dense_7/bias
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
?non_trainable_variables
K	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
Lregularization_losses
Mtrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2score/kernel
:2
score/bias
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
?non_trainable_variables
Q	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
Rregularization_losses
Strainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	2?	2lstm_3/lstm_cell_3/kernel
7:5
??	2#lstm_3/lstm_cell_3/recurrent_kernel
&:$?	2lstm_3/lstm_cell_3/bias
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
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
10
11
12
13"
trackable_list_wrapper
'
0"
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
5
Z0
[1
\2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
Z0
[1
\2"
trackable_list_wrapper
?
?non_trainable_variables
r	variables
?layer_metrics
?metrics
 ?layer_regularization_losses
sregularization_losses
ttrainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
'
%0"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
+:)?22Adam/conv1d_5/kernel/m
 :22Adam/conv1d_5/bias/m
%:#?2Adam/attention_3/att_v/m
*:(
??2Adam/attention_3/att_W/m
&:$	?@2Adam/dense_5/kernel/m
:@2Adam/dense_5/bias/m
%:#2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
%:#P 2Adam/dense_7/kernel/m
: 2Adam/dense_7/bias/m
#:! 2Adam/score/kernel/m
:2Adam/score/bias/m
1:/	2?	2 Adam/lstm_3/lstm_cell_3/kernel/m
<::
??	2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
+:)?	2Adam/lstm_3/lstm_cell_3/bias/m
+:)?22Adam/conv1d_5/kernel/v
 :22Adam/conv1d_5/bias/v
%:#?2Adam/attention_3/att_v/v
*:(
??2Adam/attention_3/att_W/v
&:$	?@2Adam/dense_5/kernel/v
:@2Adam/dense_5/bias/v
%:#2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
%:#P 2Adam/dense_7/kernel/v
: 2Adam/dense_7/bias/v
#:! 2Adam/score/kernel/v
:2Adam/score/bias/v
1:/	2?	2 Adam/lstm_3/lstm_cell_3/kernel/v
<::
??	2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
+:)?	2Adam/lstm_3/lstm_cell_3/bias/v
?B?
 __inference__wrapped_model_73820responseapplied"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_Taghipour_layer_call_and_return_conditional_losses_76331
D__inference_Taghipour_layer_call_and_return_conditional_losses_76838
D__inference_Taghipour_layer_call_and_return_conditional_losses_75869
D__inference_Taghipour_layer_call_and_return_conditional_losses_75920?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_Taghipour_layer_call_fn_75124
)__inference_Taghipour_layer_call_fn_76876
)__inference_Taghipour_layer_call_fn_76914
)__inference_Taghipour_layer_call_fn_75818?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_embedding_9_layer_call_and_return_conditional_losses_76924?
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
+__inference_embedding_9_layer_call_fn_76931?
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
J__inference_pos_x_maskedout_layer_call_and_return_conditional_losses_76944?
???
FullArgSpec 
args?
jself
jx
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
/__inference_pos_x_maskedout_layer_call_fn_76950?
???
FullArgSpec 
args?
jself
jx
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
C__inference_conv1d_5_layer_call_and_return_conditional_losses_76965?
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
(__inference_conv1d_5_layer_call_fn_76974?
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_77225
A__inference_lstm_3_layer_call_and_return_conditional_losses_77604
A__inference_lstm_3_layer_call_and_return_conditional_losses_77855
A__inference_lstm_3_layer_call_and_return_conditional_losses_78234?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lstm_3_layer_call_fn_78245
&__inference_lstm_3_layer_call_fn_78256
&__inference_lstm_3_layer_call_fn_78267
&__inference_lstm_3_layer_call_fn_78278?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_4_layer_call_and_return_conditional_losses_78283
D__inference_dropout_4_layer_call_and_return_conditional_losses_78295?
???
FullArgSpec)
args!?
jself
jinputs

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
)__inference_dropout_4_layer_call_fn_78300
)__inference_dropout_4_layer_call_fn_78305?
???
FullArgSpec)
args!?
jself
jinputs

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
F__inference_attention_3_layer_call_and_return_conditional_losses_78366?
???
FullArgSpec 
args?
jself
jx
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
+__inference_attention_3_layer_call_fn_78375?
???
FullArgSpec 
args?
jself
jx
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
B__inference_dense_5_layer_call_and_return_conditional_losses_78385?
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
'__inference_dense_5_layer_call_fn_78394?
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
B__inference_dense_6_layer_call_and_return_conditional_losses_78404?
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
'__inference_dense_6_layer_call_fn_78413?
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_78420?
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
-__inference_concatenate_1_layer_call_fn_78426?
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
D__inference_dropout_5_layer_call_and_return_conditional_losses_78431
D__inference_dropout_5_layer_call_and_return_conditional_losses_78443?
???
FullArgSpec)
args!?
jself
jinputs

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
)__inference_dropout_5_layer_call_fn_78448
)__inference_dropout_5_layer_call_fn_78453?
???
FullArgSpec)
args!?
jself
jinputs

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
B__inference_dense_7_layer_call_and_return_conditional_losses_78463?
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
'__inference_dense_7_layer_call_fn_78472?
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
@__inference_score_layer_call_and_return_conditional_losses_78483?
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
%__inference_score_layer_call_fn_78492?
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
?B?
#__inference_signature_wrapper_75966appliedresponse"?
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
 
?2?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_78574
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_78720?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

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
+__inference_lstm_cell_3_layer_call_fn_78737
+__inference_lstm_cell_3_layer_call_fn_78754?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

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
 ?
D__inference_Taghipour_layer_call_and_return_conditional_losses_75869? Z\[0/56;<IJOP???
??}
s?p
/
response#? 
response??????????
=
whether_criteria_applied!?
applied?????????
p 

 
? "9?6
/?,
*
score!?
0/score?????????
? ?
D__inference_Taghipour_layer_call_and_return_conditional_losses_75920? Z\[0/56;<IJOP???
??}
s?p
/
response#? 
response??????????
=
whether_criteria_applied!?
applied?????????
p

 
? "9?6
/?,
*
score!?
0/score?????????
? ?
D__inference_Taghipour_layer_call_and_return_conditional_losses_76331? Z\[0/56;<IJOP???
???
???
6
response*?'
inputs/response??????????
U
whether_criteria_applied9?6
inputs/whether_criteria_applied?????????
p 

 
? "9?6
/?,
*
score!?
0/score?????????
? ?
D__inference_Taghipour_layer_call_and_return_conditional_losses_76838? Z\[0/56;<IJOP???
???
???
6
response*?'
inputs/response??????????
U
whether_criteria_applied9?6
inputs/whether_criteria_applied?????????
p

 
? "9?6
/?,
*
score!?
0/score?????????
? ?
)__inference_Taghipour_layer_call_fn_75124? Z\[0/56;<IJOP???
??}
s?p
/
response#? 
response??????????
=
whether_criteria_applied!?
applied?????????
p 

 
? "-?*
(
score?
score??????????
)__inference_Taghipour_layer_call_fn_75818? Z\[0/56;<IJOP???
??}
s?p
/
response#? 
response??????????
=
whether_criteria_applied!?
applied?????????
p

 
? "-?*
(
score?
score??????????
)__inference_Taghipour_layer_call_fn_76876? Z\[0/56;<IJOP???
???
???
6
response*?'
inputs/response??????????
U
whether_criteria_applied9?6
inputs/whether_criteria_applied?????????
p 

 
? "-?*
(
score?
score??????????
)__inference_Taghipour_layer_call_fn_76914? Z\[0/56;<IJOP???
???
???
6
response*?'
inputs/response??????????
U
whether_criteria_applied9?6
inputs/whether_criteria_applied?????????
p

 
? "-?*
(
score?
score??????????
 __inference__wrapped_model_73820? Z\[0/56;<IJOP??
x?u
s?p
/
response#? 
response??????????
=
whether_criteria_applied!?
applied?????????
? "-?*
(
score?
score??????????
F__inference_attention_3_layer_call_and_return_conditional_losses_78366b0/4?1
*?'
!?
x???????????

 
? "&?#
?
0??????????
? ?
+__inference_attention_3_layer_call_fn_78375U0/4?1
*?'
!?
x???????????

 
? "????????????
H__inference_concatenate_1_layer_call_and_return_conditional_losses_78420?Z?W
P?M
K?H
"?
inputs/0?????????@
"?
inputs/1?????????
? "%?"
?
0?????????P
? ?
-__inference_concatenate_1_layer_call_fn_78426vZ?W
P?M
K?H
"?
inputs/0?????????@
"?
inputs/1?????????
? "??????????P?
C__inference_conv1d_5_layer_call_and_return_conditional_losses_76965g 5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0??????????2
? ?
(__inference_conv1d_5_layer_call_fn_76974Z 5?2
+?(
&?#
inputs???????????
? "???????????2?
B__inference_dense_5_layer_call_and_return_conditional_losses_78385]560?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? {
'__inference_dense_5_layer_call_fn_78394P560?-
&?#
!?
inputs??????????
? "??????????@?
B__inference_dense_6_layer_call_and_return_conditional_losses_78404\;</?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_6_layer_call_fn_78413O;</?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_7_layer_call_and_return_conditional_losses_78463\IJ/?,
%?"
 ?
inputs?????????P
? "%?"
?
0????????? 
? z
'__inference_dense_7_layer_call_fn_78472OIJ/?,
%?"
 ?
inputs?????????P
? "?????????? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_78283h9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_78295h9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
)__inference_dropout_4_layer_call_fn_78300[9?6
/?,
&?#
inputs???????????
p 
? "?????????????
)__inference_dropout_4_layer_call_fn_78305[9?6
/?,
&?#
inputs???????????
p
? "?????????????
D__inference_dropout_5_layer_call_and_return_conditional_losses_78431\3?0
)?&
 ?
inputs?????????P
p 
? "%?"
?
0?????????P
? ?
D__inference_dropout_5_layer_call_and_return_conditional_losses_78443\3?0
)?&
 ?
inputs?????????P
p
? "%?"
?
0?????????P
? |
)__inference_dropout_5_layer_call_fn_78448O3?0
)?&
 ?
inputs?????????P
p 
? "??????????P|
)__inference_dropout_5_layer_call_fn_78453O3?0
)?&
 ?
inputs?????????P
p
? "??????????P?
F__inference_embedding_9_layer_call_and_return_conditional_losses_76924b0?-
&?#
!?
inputs??????????
? "+?(
!?
0???????????
? ?
+__inference_embedding_9_layer_call_fn_76931U0?-
&?#
!?
inputs??????????
? "?????????????
A__inference_lstm_3_layer_call_and_return_conditional_losses_77225?Z\[O?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p 

 
? "3?0
)?&
0???????????????????
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_77604?Z\[O?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p

 
? "3?0
)?&
0???????????????????
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_77855tZ\[@?=
6?3
%?"
inputs??????????2

 
p 

 
? "+?(
!?
0???????????
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_78234tZ\[@?=
6?3
%?"
inputs??????????2

 
p

 
? "+?(
!?
0???????????
? ?
&__inference_lstm_3_layer_call_fn_78245~Z\[O?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p 

 
? "&?#????????????????????
&__inference_lstm_3_layer_call_fn_78256~Z\[O?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p

 
? "&?#????????????????????
&__inference_lstm_3_layer_call_fn_78267gZ\[@?=
6?3
%?"
inputs??????????2

 
p 

 
? "?????????????
&__inference_lstm_3_layer_call_fn_78278gZ\[@?=
6?3
%?"
inputs??????????2

 
p

 
? "?????????????
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_78574?Z\[??
x?u
 ?
inputs?????????2
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_78720?Z\[??
x?u
 ?
inputs?????????2
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
+__inference_lstm_cell_3_layer_call_fn_78737?Z\[??
x?u
 ?
inputs?????????2
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
+__inference_lstm_cell_3_layer_call_fn_78754?Z\[??
x?u
 ?
inputs?????????2
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
J__inference_pos_x_maskedout_layer_call_and_return_conditional_losses_76944?Q?N
G?D
!?
x???????????
?
mask??????????

? "+?(
!?
0???????????
? ?
/__inference_pos_x_maskedout_layer_call_fn_76950sQ?N
G?D
!?
x???????????
?
mask??????????

? "?????????????
@__inference_score_layer_call_and_return_conditional_losses_78483\OP/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? x
%__inference_score_layer_call_fn_78492OOP/?,
%?"
 ?
inputs????????? 
? "???????????
#__inference_signature_wrapper_75966? Z\[0/56;<IJOPl?i
? 
b?_
,
applied!?
applied?????????
/
response#? 
response??????????"-?*
(
score?
score?????????