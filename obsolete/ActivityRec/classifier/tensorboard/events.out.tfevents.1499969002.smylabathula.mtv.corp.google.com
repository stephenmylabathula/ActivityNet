       �K"	  �z�Y�Abrain.Event:2`��e�r      J��H	O��z�Y�A"��
�
PlaceholderPlaceholder*
dtype0*%
shape:����������*0
_output_shapes
:����������
p
Placeholder_1Placeholder*
dtype0*
shape:���������*'
_output_shapes
:���������
o
truncated_normal/shapeConst*
dtype0*%
valueB"   <      <   *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0*&
_output_shapes
:<<
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:<<
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:<<
�
Variable
VariableV2*
dtype0*
shape:<<*
	container *
shared_name *&
_output_shapes
:<<
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
:<<
q
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*&
_output_shapes
:<<
T
ConstConst*
dtype0*
valueB�*    *
_output_shapes	
:�
x

Variable_1
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
h
depthwise/ShapeConst*
dtype0*%
valueB"   <      <   *
_output_shapes
:
h
depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
�
	depthwiseDepthwiseConv2dNativePlaceholderVariable/read*
paddingVALID*
strides
*1
_output_shapes
:�����������*
T0*
data_formatNHWC
b
AddAdd	depthwiseVariable_1/read*
T0*1
_output_shapes
:�����������
M
ReluReluAdd*
T0*1
_output_shapes
:�����������
�
MaxPoolMaxPoolRelu*0
_output_shapes
:���������=�*
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"      �      *
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_1/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:�
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:�
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:�
�

Variable_2
VariableV2*
dtype0*
shape:�*
	container *
shared_name *'
_output_shapes
:�
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*'
_output_shapes
:�
x
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*'
_output_shapes
:�
V
Const_1Const*
dtype0*
valueB�*    *
_output_shapes	
:�
x

Variable_3
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
j
depthwise_1/ShapeConst*
dtype0*%
valueB"      �      *
_output_shapes
:
j
depthwise_1/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
�
depthwise_1DepthwiseConv2dNativeMaxPoolVariable_2/read*
paddingVALID*
strides
*0
_output_shapes
:���������8�*
T0*
data_formatNHWC
e
Add_1Adddepthwise_1Variable_3/read*
T0*0
_output_shapes
:���������8�
P
Relu_1ReluAdd_1*
T0*0
_output_shapes
:���������8�
^
Reshape/shapeConst*
dtype0*
valueB"����@�  *
_output_shapes
:
k
ReshapeReshapeRelu_1Reshape/shape*)
_output_shapes
:�����������*
T0*
Tshape0
i
truncated_normal_2/shapeConst*
dtype0*
valueB"@�  �  *
_output_shapes
:
\
truncated_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0*!
_output_shapes
:���
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*!
_output_shapes
:���
v
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*!
_output_shapes
:���
�

Variable_4
VariableV2*
dtype0*
shape:���*
	container *
shared_name *!
_output_shapes
:���
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*!
_output_shapes
:���
r
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*!
_output_shapes
:���
V
Const_2Const*
dtype0*
valueB�*    *
_output_shapes	
:�
x

Variable_5
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
�
MatMulMatMulReshapeVariable_4/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:����������
X
Add_2AddMatMulVariable_5/read*
T0*(
_output_shapes
:����������
F
TanhTanhAdd_2*
T0*(
_output_shapes
:����������
i
truncated_normal_3/shapeConst*
dtype0*
valueB"�     *
_output_shapes
:
\
truncated_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_3/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	�
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	�
�

Variable_6
VariableV2*
dtype0*
shape:	�*
	container *
shared_name *
_output_shapes
:	�
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
T
Const_3Const*
dtype0*
valueB*    *
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
�
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
MatMul_1MatMulTanhVariable_6/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
W
addAddMatMul_1Variable_7/read*
T0*'
_output_shapes
:���������
I
SoftmaxSoftmaxadd*
T0*'
_output_shapes
:���������
E
LogLogSoftmax*
T0*'
_output_shapes
:���������
P
mulMulPlaceholder_1Log*
T0*'
_output_shapes
:���������
X
Const_4Const*
dtype0*
valueB"       *
_output_shapes
:
V
SumSummulConst_4*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
0
NegNegSum*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
q
 gradients/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
[
gradients/Sum_grad/ShapeShapemul*
out_type0*
T0*
_output_shapes
:
�
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
e
gradients/mul_grad/ShapeShapePlaceholder_1*
out_type0*
T0*
_output_shapes
:
]
gradients/mul_grad/Shape_1ShapeLog*
out_type0*
T0*
_output_shapes
:
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
m
gradients/mul_grad/mulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:���������
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
y
gradients/mul_grad/mul_1MulPlaceholder_1gradients/Sum_grad/Tile*
T0*'
_output_shapes
:���������
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
gradients/Log_grad/Reciprocal
ReciprocalSoftmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:���������
t
gradients/Softmax_grad/mulMulgradients/Log_grad/mulSoftmax*
T0*'
_output_shapes
:���������
v
,gradients/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*#
_output_shapes
:���������*
T0*
	keep_dims( *

Tidx0
u
$gradients/Softmax_grad/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
gradients/Softmax_grad/ReshapeReshapegradients/Softmax_grad/Sum$gradients/Softmax_grad/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/Softmax_grad/subSubgradients/Log_grad/mulgradients/Softmax_grad/Reshape*
T0*'
_output_shapes
:���������
z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:���������
`
gradients/add_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Softmax_grad/mul_1(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/add_grad/Sum_1Sumgradients/Softmax_grad/mul_1*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
 gradients/MatMul_1_grad/MatMul_1MatMulTanh+gradients/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	�
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0*(
_output_shapes
:����������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	�
�
gradients/Tanh_grad/TanhGradTanhGradTanh0gradients/MatMul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
`
gradients/Add_2_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
g
gradients/Add_2_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_2_grad/SumSumgradients/Tanh_grad/TanhGrad*gradients/Add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
gradients/Add_2_grad/Sum_1Sumgradients/Tanh_grad/TanhGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
�
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_2_grad/Reshape*
T0*(
_output_shapes
:����������
�
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
transpose_a( *
T0*)
_output_shapes
:�����������
�
gradients/MatMul_grad/MatMul_1MatMulReshape-gradients/Add_2_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*!
_output_shapes
:���
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*)
_output_shapes
:�����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*!
_output_shapes
:���
b
gradients/Reshape_grad/ShapeShapeRelu_1*
out_type0*
T0*
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*0
_output_shapes
:���������8�*
T0*
Tshape0
�
gradients/Relu_1_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_1*
T0*0
_output_shapes
:���������8�
e
gradients/Add_1_grad/ShapeShapedepthwise_1*
out_type0*
T0*
_output_shapes
:
g
gradients/Add_1_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*0
_output_shapes
:���������8�*
T0*
Tshape0
�
gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
�
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*
T0*0
_output_shapes
:���������8�
�
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
T0*
_output_shapes	
:�
g
 gradients/depthwise_1_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
�
=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInput gradients/depthwise_1_grad/ShapeVariable_2/read-gradients/Add_1_grad/tuple/control_dependency*
paddingVALID*
strides
*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC
{
"gradients/depthwise_1_grad/Shape_1Const*
dtype0*%
valueB"      �      *
_output_shapes
:
�
>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterMaxPool"gradients/depthwise_1_grad/Shape_1-gradients/Add_1_grad/tuple/control_dependency*
paddingVALID*
strides
*'
_output_shapes
:�*
T0*
data_formatNHWC
�
+gradients/depthwise_1_grad/tuple/group_depsNoOp>^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput?^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter
�
3gradients/depthwise_1_grad/tuple/control_dependencyIdentity=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput,^gradients/depthwise_1_grad/tuple/group_deps*P
_classF
DBloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput*
T0*0
_output_shapes
:���������=�
�
5gradients/depthwise_1_grad/tuple/control_dependency_1Identity>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter,^gradients/depthwise_1_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter*
T0*'
_output_shapes
:�
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool3gradients/depthwise_1_grad/tuple/control_dependency*1
_output_shapes
:�����������*
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*1
_output_shapes
:�����������
a
gradients/Add_grad/ShapeShape	depthwise*
out_type0*
T0*
_output_shapes
:
e
gradients/Add_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*1
_output_shapes
:�����������*
T0*
Tshape0
�
gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
�
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*-
_class#
!loc:@gradients/Add_grad/Reshape*
T0*1
_output_shapes
:�����������
�
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
T0*
_output_shapes	
:�
i
gradients/depthwise_grad/ShapeShapePlaceholder*
out_type0*
T0*
_output_shapes
:
�
;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInputgradients/depthwise_grad/ShapeVariable/read+gradients/Add_grad/tuple/control_dependency*
paddingVALID*
strides
*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC
y
 gradients/depthwise_grad/Shape_1Const*
dtype0*%
valueB"   <      <   *
_output_shapes
:
�
<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterPlaceholder gradients/depthwise_grad/Shape_1+gradients/Add_grad/tuple/control_dependency*
paddingVALID*
strides
*&
_output_shapes
:<<*
T0*
data_formatNHWC
�
)gradients/depthwise_grad/tuple/group_depsNoOp<^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput=^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter
�
1gradients/depthwise_grad/tuple/control_dependencyIdentity;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput*^gradients/depthwise_grad/tuple/group_deps*N
_classD
B@loc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput*
T0*0
_output_shapes
:����������
�
3gradients/depthwise_grad/tuple/control_dependency_1Identity<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter*^gradients/depthwise_grad/tuple/group_deps*O
_classE
CAloc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter*
T0*&
_output_shapes
:<<
b
GradientDescent/learning_rateConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
�
4GradientDescent/update_Variable/ApplyGradientDescentApplyGradientDescentVariableGradientDescent/learning_rate3gradients/depthwise_grad/tuple/control_dependency_1*
_class
loc:@Variable*
use_locking( *
T0*&
_output_shapes
:<<
�
6GradientDescent/update_Variable_1/ApplyGradientDescentApplyGradientDescent
Variable_1GradientDescent/learning_rate-gradients/Add_grad/tuple/control_dependency_1*
_class
loc:@Variable_1*
use_locking( *
T0*
_output_shapes	
:�
�
6GradientDescent/update_Variable_2/ApplyGradientDescentApplyGradientDescent
Variable_2GradientDescent/learning_rate5gradients/depthwise_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_2*
use_locking( *
T0*'
_output_shapes
:�
�
6GradientDescent/update_Variable_3/ApplyGradientDescentApplyGradientDescent
Variable_3GradientDescent/learning_rate/gradients/Add_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_3*
use_locking( *
T0*
_output_shapes	
:�
�
6GradientDescent/update_Variable_4/ApplyGradientDescentApplyGradientDescent
Variable_4GradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
_class
loc:@Variable_4*
use_locking( *
T0*!
_output_shapes
:���
�
6GradientDescent/update_Variable_5/ApplyGradientDescentApplyGradientDescent
Variable_5GradientDescent/learning_rate/gradients/Add_2_grad/tuple/control_dependency_1*
_class
loc:@Variable_5*
use_locking( *
T0*
_output_shapes	
:�
�
6GradientDescent/update_Variable_6/ApplyGradientDescentApplyGradientDescent
Variable_6GradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_6*
use_locking( *
T0*
_output_shapes
:	�
�
6GradientDescent/update_Variable_7/ApplyGradientDescentApplyGradientDescent
Variable_7GradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
_class
loc:@Variable_7*
use_locking( *
T0*
_output_shapes
:
�
GradientDescentNoOp5^GradientDescent/update_Variable/ApplyGradientDescent7^GradientDescent/update_Variable_1/ApplyGradientDescent7^GradientDescent/update_Variable_2/ApplyGradientDescent7^GradientDescent/update_Variable_3/ApplyGradientDescent7^GradientDescent/update_Variable_4/ApplyGradientDescent7^GradientDescent/update_Variable_5/ApplyGradientDescent7^GradientDescent/update_Variable_6/ApplyGradientDescent7^GradientDescent/update_Variable_7/ApplyGradientDescent
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
e
ArgMaxArgMaxSoftmaxArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
o
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*#
_output_shapes
:���������*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
P
CastCastEqual*

DstT0*

SrcT0
*#
_output_shapes
:���������
Q
Const_5Const*
dtype0*
valueB: *
_output_shapes
:
Y
MeanMeanCastConst_5*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0"-��      �=H�	���z�Y�AJ��
��
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyGradientDescent
var"T�

alpha"T

delta"T
out"T�"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
�
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
#DepthwiseConv2dNativeBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
"DepthwiseConv2dNativeBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
4
Fill
dims

value"T
output"T"	
Ttype
.
Identity

input"T
output"T"	
Ttype
+
Log
x"T
y"T"
Ttype:	
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
�
MaxPool

input"T
output"T"
Ttype0:
2		"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2		
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
4

Reciprocal
x"T
y"T"
Ttype:
	2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
,
Tanh
x"T
y"T"
Ttype:	
2
8
TanhGrad
x"T
y"T
z"T"
Ttype:	
2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.2.12v1.2.0-5-g435cdfc��
�
PlaceholderPlaceholder*
dtype0*%
shape:����������*0
_output_shapes
:����������
p
Placeholder_1Placeholder*
dtype0*
shape:���������*'
_output_shapes
:���������
o
truncated_normal/shapeConst*
dtype0*%
valueB"   <      <   *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0*&
_output_shapes
:<<
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:<<
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:<<
�
Variable
VariableV2*
dtype0*
shape:<<*
shared_name *
	container *&
_output_shapes
:<<
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
:<<
q
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*&
_output_shapes
:<<
T
ConstConst*
dtype0*
valueB�*    *
_output_shapes	
:�
x

Variable_1
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
h
depthwise/ShapeConst*
dtype0*%
valueB"   <      <   *
_output_shapes
:
h
depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
�
	depthwiseDepthwiseConv2dNativePlaceholderVariable/read*
paddingVALID*
strides
*
data_formatNHWC*
T0*1
_output_shapes
:�����������
b
AddAdd	depthwiseVariable_1/read*
T0*1
_output_shapes
:�����������
M
ReluReluAdd*
T0*1
_output_shapes
:�����������
�
MaxPoolMaxPoolRelu*0
_output_shapes
:���������=�*
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"      �      *
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_1/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:�
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:�
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:�
�

Variable_2
VariableV2*
dtype0*
shape:�*
shared_name *
	container *'
_output_shapes
:�
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*'
_output_shapes
:�
x
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*'
_output_shapes
:�
V
Const_1Const*
dtype0*
valueB�*    *
_output_shapes	
:�
x

Variable_3
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
j
depthwise_1/ShapeConst*
dtype0*%
valueB"      �      *
_output_shapes
:
j
depthwise_1/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
�
depthwise_1DepthwiseConv2dNativeMaxPoolVariable_2/read*
paddingVALID*
strides
*
data_formatNHWC*
T0*0
_output_shapes
:���������8�
e
Add_1Adddepthwise_1Variable_3/read*
T0*0
_output_shapes
:���������8�
P
Relu_1ReluAdd_1*
T0*0
_output_shapes
:���������8�
^
Reshape/shapeConst*
dtype0*
valueB"����@�  *
_output_shapes
:
k
ReshapeReshapeRelu_1Reshape/shape*
Tshape0*
T0*)
_output_shapes
:�����������
i
truncated_normal_2/shapeConst*
dtype0*
valueB"@�  �  *
_output_shapes
:
\
truncated_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0*!
_output_shapes
:���
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*!
_output_shapes
:���
v
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*!
_output_shapes
:���
�

Variable_4
VariableV2*
dtype0*
shape:���*
shared_name *
	container *!
_output_shapes
:���
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*!
_output_shapes
:���
r
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*!
_output_shapes
:���
V
Const_2Const*
dtype0*
valueB�*    *
_output_shapes	
:�
x

Variable_5
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
�
MatMulMatMulReshapeVariable_4/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:����������
X
Add_2AddMatMulVariable_5/read*
T0*(
_output_shapes
:����������
F
TanhTanhAdd_2*
T0*(
_output_shapes
:����������
i
truncated_normal_3/shapeConst*
dtype0*
valueB"�     *
_output_shapes
:
\
truncated_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_3/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	�
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	�
t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	�
�

Variable_6
VariableV2*
dtype0*
shape:	�*
shared_name *
	container *
_output_shapes
:	�
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
T
Const_3Const*
dtype0*
valueB*    *
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
�
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
MatMul_1MatMulTanhVariable_6/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
W
addAddMatMul_1Variable_7/read*
T0*'
_output_shapes
:���������
I
SoftmaxSoftmaxadd*
T0*'
_output_shapes
:���������
E
LogLogSoftmax*
T0*'
_output_shapes
:���������
P
mulMulPlaceholder_1Log*
T0*'
_output_shapes
:���������
X
Const_4Const*
dtype0*
valueB"       *
_output_shapes
:
V
SumSummulConst_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
0
NegNegSum*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
q
 gradients/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
[
gradients/Sum_grad/ShapeShapemul*
out_type0*
T0*
_output_shapes
:
�
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
e
gradients/mul_grad/ShapeShapePlaceholder_1*
out_type0*
T0*
_output_shapes
:
]
gradients/mul_grad/Shape_1ShapeLog*
out_type0*
T0*
_output_shapes
:
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
m
gradients/mul_grad/mulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:���������
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
y
gradients/mul_grad/mul_1MulPlaceholder_1gradients/Sum_grad/Tile*
T0*'
_output_shapes
:���������
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
gradients/Log_grad/Reciprocal
ReciprocalSoftmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:���������
t
gradients/Softmax_grad/mulMulgradients/Log_grad/mulSoftmax*
T0*'
_output_shapes
:���������
v
,gradients/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *#
_output_shapes
:���������
u
$gradients/Softmax_grad/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
gradients/Softmax_grad/ReshapeReshapegradients/Softmax_grad/Sum$gradients/Softmax_grad/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:���������
�
gradients/Softmax_grad/subSubgradients/Log_grad/mulgradients/Softmax_grad/Reshape*
T0*'
_output_shapes
:���������
z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:���������
`
gradients/add_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Softmax_grad/mul_1(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sumgradients/Softmax_grad/mul_1*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
 gradients/MatMul_1_grad/MatMul_1MatMulTanh+gradients/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	�
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0*(
_output_shapes
:����������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	�
�
gradients/Tanh_grad/TanhGradTanhGradTanh0gradients/MatMul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
`
gradients/Add_2_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
g
gradients/Add_2_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_2_grad/SumSumgradients/Tanh_grad/TanhGrad*gradients/Add_2_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*
Tshape0*
T0*(
_output_shapes
:����������
�
gradients/Add_2_grad/Sum_1Sumgradients/Tanh_grad/TanhGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
�
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_2_grad/Reshape*
T0*(
_output_shapes
:����������
�
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
transpose_a( *
T0*)
_output_shapes
:�����������
�
gradients/MatMul_grad/MatMul_1MatMulReshape-gradients/Add_2_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*!
_output_shapes
:���
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*)
_output_shapes
:�����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*!
_output_shapes
:���
b
gradients/Reshape_grad/ShapeShapeRelu_1*
out_type0*
T0*
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
Tshape0*
T0*0
_output_shapes
:���������8�
�
gradients/Relu_1_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_1*
T0*0
_output_shapes
:���������8�
e
gradients/Add_1_grad/ShapeShapedepthwise_1*
out_type0*
T0*
_output_shapes
:
g
gradients/Add_1_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
Tshape0*
T0*0
_output_shapes
:���������8�
�
gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
�
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*
T0*0
_output_shapes
:���������8�
�
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
T0*
_output_shapes	
:�
g
 gradients/depthwise_1_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
�
=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInput gradients/depthwise_1_grad/ShapeVariable_2/read-gradients/Add_1_grad/tuple/control_dependency*
paddingVALID*
strides
*
data_formatNHWC*
T0*J
_output_shapes8
6:4������������������������������������
{
"gradients/depthwise_1_grad/Shape_1Const*
dtype0*%
valueB"      �      *
_output_shapes
:
�
>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterMaxPool"gradients/depthwise_1_grad/Shape_1-gradients/Add_1_grad/tuple/control_dependency*
paddingVALID*
strides
*
data_formatNHWC*
T0*'
_output_shapes
:�
�
+gradients/depthwise_1_grad/tuple/group_depsNoOp>^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput?^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter
�
3gradients/depthwise_1_grad/tuple/control_dependencyIdentity=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput,^gradients/depthwise_1_grad/tuple/group_deps*P
_classF
DBloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput*
T0*0
_output_shapes
:���������=�
�
5gradients/depthwise_1_grad/tuple/control_dependency_1Identity>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter,^gradients/depthwise_1_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter*
T0*'
_output_shapes
:�
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool3gradients/depthwise_1_grad/tuple/control_dependency*1
_output_shapes
:�����������*
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*1
_output_shapes
:�����������
a
gradients/Add_grad/ShapeShape	depthwise*
out_type0*
T0*
_output_shapes
:
e
gradients/Add_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
Tshape0*
T0*1
_output_shapes
:�����������
�
gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
Tshape0*
T0*
_output_shapes	
:�
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
�
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*-
_class#
!loc:@gradients/Add_grad/Reshape*
T0*1
_output_shapes
:�����������
�
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
T0*
_output_shapes	
:�
i
gradients/depthwise_grad/ShapeShapePlaceholder*
out_type0*
T0*
_output_shapes
:
�
;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInputgradients/depthwise_grad/ShapeVariable/read+gradients/Add_grad/tuple/control_dependency*
paddingVALID*
strides
*
data_formatNHWC*
T0*J
_output_shapes8
6:4������������������������������������
y
 gradients/depthwise_grad/Shape_1Const*
dtype0*%
valueB"   <      <   *
_output_shapes
:
�
<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterPlaceholder gradients/depthwise_grad/Shape_1+gradients/Add_grad/tuple/control_dependency*
paddingVALID*
strides
*
data_formatNHWC*
T0*&
_output_shapes
:<<
�
)gradients/depthwise_grad/tuple/group_depsNoOp<^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput=^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter
�
1gradients/depthwise_grad/tuple/control_dependencyIdentity;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput*^gradients/depthwise_grad/tuple/group_deps*N
_classD
B@loc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput*
T0*0
_output_shapes
:����������
�
3gradients/depthwise_grad/tuple/control_dependency_1Identity<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter*^gradients/depthwise_grad/tuple/group_deps*O
_classE
CAloc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter*
T0*&
_output_shapes
:<<
b
GradientDescent/learning_rateConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
�
4GradientDescent/update_Variable/ApplyGradientDescentApplyGradientDescentVariableGradientDescent/learning_rate3gradients/depthwise_grad/tuple/control_dependency_1*
_class
loc:@Variable*
use_locking( *
T0*&
_output_shapes
:<<
�
6GradientDescent/update_Variable_1/ApplyGradientDescentApplyGradientDescent
Variable_1GradientDescent/learning_rate-gradients/Add_grad/tuple/control_dependency_1*
_class
loc:@Variable_1*
use_locking( *
T0*
_output_shapes	
:�
�
6GradientDescent/update_Variable_2/ApplyGradientDescentApplyGradientDescent
Variable_2GradientDescent/learning_rate5gradients/depthwise_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_2*
use_locking( *
T0*'
_output_shapes
:�
�
6GradientDescent/update_Variable_3/ApplyGradientDescentApplyGradientDescent
Variable_3GradientDescent/learning_rate/gradients/Add_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_3*
use_locking( *
T0*
_output_shapes	
:�
�
6GradientDescent/update_Variable_4/ApplyGradientDescentApplyGradientDescent
Variable_4GradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
_class
loc:@Variable_4*
use_locking( *
T0*!
_output_shapes
:���
�
6GradientDescent/update_Variable_5/ApplyGradientDescentApplyGradientDescent
Variable_5GradientDescent/learning_rate/gradients/Add_2_grad/tuple/control_dependency_1*
_class
loc:@Variable_5*
use_locking( *
T0*
_output_shapes	
:�
�
6GradientDescent/update_Variable_6/ApplyGradientDescentApplyGradientDescent
Variable_6GradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_6*
use_locking( *
T0*
_output_shapes
:	�
�
6GradientDescent/update_Variable_7/ApplyGradientDescentApplyGradientDescent
Variable_7GradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
_class
loc:@Variable_7*
use_locking( *
T0*
_output_shapes
:
�
GradientDescentNoOp5^GradientDescent/update_Variable/ApplyGradientDescent7^GradientDescent/update_Variable_1/ApplyGradientDescent7^GradientDescent/update_Variable_2/ApplyGradientDescent7^GradientDescent/update_Variable_3/ApplyGradientDescent7^GradientDescent/update_Variable_4/ApplyGradientDescent7^GradientDescent/update_Variable_5/ApplyGradientDescent7^GradientDescent/update_Variable_6/ApplyGradientDescent7^GradientDescent/update_Variable_7/ApplyGradientDescent
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
e
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
o
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*

Tidx0*
T0*#
_output_shapes
:���������
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
P
CastCastEqual*

DstT0*

SrcT0
*#
_output_shapes
:���������
Q
Const_5Const*
dtype0*
valueB: *
_output_shapes
:
Y
MeanMeanCastConst_5*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: ""
train_op

GradientDescent"�
	variables��
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0
4
Variable_5:0Variable_5/AssignVariable_5/read:0
4
Variable_6:0Variable_6/AssignVariable_6/read:0
4
Variable_7:0Variable_7/AssignVariable_7/read:0"�
trainable_variables��
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0
4
Variable_5:0Variable_5/AssignVariable_5/read:0
4
Variable_6:0Variable_6/AssignVariable_6/read:0
4
Variable_7:0Variable_7/AssignVariable_7/read:0���%7       ���Y	� ~�Y�A*,

	Loss/Losso�A

Accuracy/Accuracy
�c?,�o9       �7�	0��Y�A*,

	Loss/Loss)��A

Accuracy/Accuracy33s?w�9       �7�	�Y\��Y�A*,

	Loss/LossM�XA

Accuracy/AccuracyH�z?tj[�9       �7�	nn���Y�A*,

	Loss/Loss��8A

Accuracy/Accuracy�p}?w�?�9       �7�	M�΋�Y�A*,

	Loss/Loss��A

Accuracy/Accuracy�p}?�̓�9       �7�	ga@��Y�A*,

	Loss/Loss��
A

Accuracy/Accuracy�p}?��s9       �7�	�^Ȓ�Y�A*,

	Loss/Loss`�@

Accuracy/Accuracy�p}?&�dr9       �7�	}"���Y�A*,

	Loss/Loss�Z�@

Accuracy/Accuracy  �?�Fb�      ���	ob��Y�A�*��
��
Test_Confusion_Matrix/image/0"ٓ��"͓�PNG

   IHDR  �  �   5���    IDATx���wTW�g�RV�`�T,A�E��#�+6,�J�]P,�5+,Q#�(��J� M����k6j�a�Y��3�w�̾3aٗ�F"� """"Rb@DDDDE�	 ��aHDDD�f� �&�DDDDj�	 ��aHDDD�f� �&�DDDDj�	 ��aHDDD�f� �&�DDDDj�	 ��aHT�]�~������`hh�z��a���HLLT�{_�znnn011�D"A``���C"����_��U%طo��D�����DDśDA� �H9֭[�#F���#F�@��Ց���˗/cݺu�]�6��ݫ���[�.222�t�R�.]����J�
}��/���
VVV
=�*144D�=���<�>Dݺu��������XbHTB]�p͚5C�6m��o�AGGGnNN�9��;*-mmm2+W�T�{��/I 333����D������b0Q	 �D��k�~����_����|,\�U�V���.���ѿ<y�D�-Z���	���h֬���Q�R%̟?��� �w?���bժU�H$���������Ǻ,O�<�-Z����J�BŊѽ{w�y�FV�c]�7n�@�ΝQ�L���N�:���D��۷cʔ)������1Z�n��w����}wׯ_��_ccc���a�ر�����۷Ѯ];���?�����o߾����Q�N������...������I$ddd 88Xv[�h!wώ;�A��\�r���GVV������066��_-w��'OBSSS�N��k&���	 Q	�����'O�����֟u�������ߣM�6ؿ?fϞ�#G����/^��������}��_�~ؿ?<<<��燐� @�p�� @�=p������:@GG7nđ#G0�| ;;���ݽ{����y�&�.]�={��z������?�?y�d<~��ׯ�ڵkq��}t��yyy�gϞ=Q�vm�ٳ�F`` F��N�:���{��E�V�0q�D�q|YYYx��Ǝ��{�b���hڴ)�w�͛7��]�p�J�B���e���-�>>>�����-[�k�.hkk���=֮]�]�va�ҥ 
�?zyy�Y�f�9s�g]/��8			 �w�ޟU���� aĈr���� @�<y�����M  ���ru�W�.�k�N��0r�H��3f�ճi�&�-� �ڵK  DFF�c� �3f�^���[���bcc��yxx�����ׯA�S�N	 ���������_ ��}�]ǢE����ԩ# ���#+���ʕ+'t��������rrr�nݺr��|p̻{ֿ�O�{w?�>|����#\�pAhٲ�P�\9��ӧ�x�DT���p��) ����\yÆQ�Z5�8qB�\*��aÆre�j���ǏS�:u����o�����x���gw��I��������7o�|�٩S'�׵j��ϾOOO��ժU�D"�����LKKU�T���;w�D�&M`hh---hkkcÆ�}��g��;ݻw��K�,A�5вeK�>}!!!������#��	 Q	T�lY���#::���|� P�|��YZZ���cff�A=]]]dff��h?�r��8~�8���1r�HT�\�+W��?���ǽ|����n�_��Z�͘��k155�{���}}}���}P��s�ٳ={�D�
�. <<��۷o?�����~���.������[ԩSm۶���"���	 Q	���	wwwDDD|0��c�%A����{��ʖ-����%FYYYr�O� �Y�f8p� RRRp��E������;v�������>y z-�;;;���/�ҥ7n����p_>Ǘ���������ѠA\�r�/���#��	 Q	���A0dȐ�N������  �Z� �$�w���q��m���+,.[[[ T�����?y���&5j�+V  �\��ɺ���8y��I��͛����ƍ���K"�@GGG.yKHH�`0���Ռ�������8u�F��~��.]*����x�; "R�Z�
#F����3��5j ''W�^�ڵk��䄎;����~�-�-[xxx &&ӦM���5Ǝ���ڷoSSS���`֬Y���BPP�����^�'O�D���acc��o�b�ƍ �֭[��3f�����-Z`���055�֭[q��A,\�&&&
������Ğ={0b����qqq�={6ʗ/�����խY�&���p���/_FFFptt���6lbccq��%`ѢE�p�z�ꅫW��t��
�:"RuL �J�!C��aÆX�d	,X���hkk���^^^5j���U�P�relذ+V����	���+̛7�c��+ccc9r����ׯJ�.��������իS��;$$$���NNNؿ�?�[stt����1y�d�9����V�6m���$18IIIX�z56n܈J�*�~��'O>X����ȑ#ѻwo�y�nnn���[�~=BBB�i�&ԨQ@���_~���������T"R-|���@""""5�����H�0$"""R3L �����@""""5�����H�p@5����gϞ����!EDDœ HKK���%44�����۷}������3��R;qqq nܸq�f[\\\�}�dff
��TX�R�T���������~�����F�����섙3g
yyy�:���3����zzz����p����}�V5j�`ff&���;v,�{�LlTCFFF�hV��7�b����b�P�hJ��J0A�;�b%55+6|���dgg�y@�*�V![s�p����?�0   �֭Cpp0jԨ�˗/c���011��1c  .Ē%K�i�&888`Μ9hӦ�޽+�O���8p� v��333�?��������f���d��dݾZ�L ?�����!;��3R&���(�~�5
�]�a����s���С ���۷o��˗ �  00�'OF�n�  ��������m�0t�P���`Æزe����!!!�������Ѯ]��]��8	��������T�-++��<==q��	ܻw p��5�;w�۷ DGG#!!A�⺺�pss���� ��ɑ�cii	'''Y��-�DDD�<I�V�s ����+�1c���?�>l�0<~�������B^^�Ν�>}�    r�YXX���ǲ::::(S��u�_�1$"""����p?z qqq066����~��ҥK�i�&l߾5j�@dd$|}}aii�2���	 ���r	�̝;ӦMC�޽ 5k���Ǐ1o�<0 R� ���KKK�q����}R����HNN�kLLL����"/KHDDDJ$y��_�/lB��χ��|���&��&���A*�"44T�?;;�O����+ ������ru���q��Y��-�DDD�<
��\]�t��9s`mm�5j��իX�x1Tp:�������=��� }}}xyy LLL�������������0aj֬)�\�1$""�eɒ%066�ȑ#eݼC�����eu&M����L�1���hԨ�;&�Vb`` ���гgOdff���AAA�~@ �� ����T��� -��g�=�O���H��u �LjjJ�������C�����wM�j�v!�kr򀣷�4���-�DDD�<"tӿ�$""""5�@"""R.M��������]�*�]�DDDDj�-�DDD�<lTIL ���Hy8P%�����HͰ������]�*�	 )��U�������[ ���Hy���� ��X%�����HͰ�����G�onbs��1$"""��@�Ĝ����H�0$����B������7��a�QL�;�����H<��7��ԏA�nSEny�r���B$�r��/�����ެmQ^�h��[�����q]��pA��#p���ݾ�]:�iig�6�W����}&BĪ�̙���q ,-�A"��}����J���2�����a-��/_O�TǻI ��H�� �RM��-�x����Qm�'&�_��_���}eu&����n�1j�4�	�/:=K���l�a!�XVD��#Q��.�}�~��u*W㲊ԙ3�0|D_�q�W9�	��y�h烌�7�:�­�+�ĩ-��S������������7�]�:V��+v(��ח�Q�ϞEȶ��C�IuH��Bq )U'�v�$]: x��}Z�G}'Y߮��}-��q 0�G?$�r^�<����q���Kg"�n `�5�m ��WC���E|UE���r�7l���.������  �M]��ͱ`�$Y�J���4NU���
�������rZZZ�J���賱����?OýNc�W� Ԫ䈦N�d	���
����X�y�1�998}�2\����y����e�L �HЫ�t�uv-�H�G��� LMM  ���8t0�������-\���k��︘a�����Q��3*UrE�>#���c�CR�VIl$�Zs��XX�ަ���́��&���a�  RӲ ���r�%�~sK��A��b��x��rrs�&�-���G�qEw1*@�7M�:��� �����o�p�:̚�y�'�葳��}���7��"GMT�5jT���pp�Cb�̝�M�tō'`fVF��T�7���

���/^�~ ���Ǿ}�)j\�ct�~ض+z��͘�S�*�����$l��������}� �HMFWw윺��}�1��x��Q�p#�N��&+��� t��߱� �:u��+X�f@"%��h)�w͚���3�Ti����7�[##�4v+��իadd���\YYzz:���ѴiS���Ν�D"�ݻw�:�"5�Ϸ��m5~	;�1�r� ��	�_�! ��W-e��gQ�,��*Xպ�5m��Sq2�"�?��Y!+q��M���U�$�1�g����8~2VVRYyٲe����j�*�կZ�2g���@5kV����b������ *P˖-����˗/��Ξ=�T�˗/�͛��6O�:KKK8::�j�ѐh 7/O�,/?����'��m���kkiíV}��Yp����'%��� �n�,��{�'�ag'?�CGG��Ľ{�_6����ƦBQ�JD ���p��}�/o!v(����U@rttD���&+C�ΝQ�re�?^��e˂n�ŋ�f͚000���5F������~���hT�RÇ�u��}�O`��P�o�6������������1�Ϸ���5l� h�\��z�m' ܍���؇X�;k�Ryk���6�\�b]Z�=r&�nݏ-[��� 		ϑ����oeu&L������u�����X�<�8�a�����HO�@d�DF�  DG�"2�bc���j���2&�����?��������t�C�Ј>�c �e˖8u�~�� -}�&MB~~>N�:�֭[#++.\��e� �p-]�vvvx��F��I�&a�ʕ��~QQQh׮�����:YYY��ʒ�NMMU��~��� �MV���2fx�2	k��Y!�du���t��r�t�12Ɵw����`�g������s�,<�g����><�ŀ�p8�L�]�XV�� po��\����0�� �K�6X��毅�9pt���]KѴi�"�W]�|-[~-{=n�L ��_#((P��T�חy�4^^���E2ʕ3E���p��o���;4ՠ�.\v+�DA� J�����&ldff���Ϟ=�ɓ'�t�R���8s�������CT�T�s�ܹÇǋc�>5	dժU��􄟟&L��ɘ���1s��w�t �4r�%]��}b�P�hJ��J0AP��U����ҥ�#%%���E���011��:������]-��K:v+X�-������p�={(W����p��eddd ,,+V�%�N�B�6mP�B���x��ܘ�����E�֭1u��L� ���)))�-.N��N!"""yL �J�*���©S�p��)��� �R)lmm���ԩShժ`��Ǐ�}��prr��ݻ�+V  ���?�>�ʕCÆ�cǎ����Յ����FDDT$4��B�*A˖-���0�h�BV��憣G���ŋ�	 �/_F^^-Z�ƍ���Ϟ����J����===�k�iiiʺ""������$&�JвeK�;w����@� \�n޾}+K +W����,[��=-[�z���z<xZZZ�������DDD��� *A˖-����*U�����:PnnnHKKC�ʕam]��[�:u�x�b,X� NNNغu+�͛���ehh�ÇCt��
�""���� �$�VC��Y���8��q0)gQgpV�,���V � �.MDDDʣ�.\v+[ ���Hy$P�,�/{K[[[H$���#G(xƺ��?,--Q�T)�h�7oޔ;GVVF���e���� �:u'OtS������J���p���˶��P ��_<�p�X�d	�/_���pH�R�i�FnI5___�ݻ;v���s琞�OOO���rM�������G�Y��ʕ�T*�m���;*W�777����@L�<ݺu���������l۶ ����6`ѢEhݺ5�֭���DEE���ㅿ'*�	 )�"ր����*�eee���ggg#$$��D"Att4жm[Y]]]������� �������ձ�������Nq�������cc��� ���011�m��n�}����kx{{  @n��w���KHH���ʔ)��:�gQ�'������a�xxx���R��;l$"""�Q䣀����K ?~��Ǐc����2�T
 HLL�����(�'�J�������O�)� ��hH$
���M�6���:t�����A*��f�O�>WWW ���3���������ƍ�:�����������ǦM�0`� hi�Ow$	|}} {{{���#   ������ �������Ǐ���LMM1a�ԬY�[���	 )�_�p��I����Ǐ#66��`ߤI�����#F 99�5±c�`dd$�---������pwwGPP45�\c!A;*Z�t�t �J�����'vŎ��P���|�C(VRS�P�tu����M�P�{|��m �N�ڛ��\d�	/��K:�$"""R3�&"""����:~�9)@"""R�� �?c0��a )[ U@"""R��>˷'QL0$�.`""""5�@"""Rv�&&�DDD�4�VM�&"""R3l$"""�a�jbHDDDJ����o�"!9�&"""R3l$"""��$��������c U�������[ ���Hy�(�P� ��(b`���� ���{����H�0��#�������bW�C�L"��/��E������F�@� �xL ���Hi����&LDDD�f�HDDDJ�.`�������F�¯���O��LDDD�f�HDDDJ�I ��	 )� �&v�� ��hH
�Ba��1$"""��@��.`""""5�@"""R	0	D!��_1$"""�a�jb0��a )�TML ���Hi�,`��.`"""*q�>}�~���������S�"""d�A���?,--Q�T)�h�7oޔ;GVVF���e���� �:u'O��R��	 )ͻI �ݾDrr2�4immm>|�n�¢E�P�tiY��bɒ%X�|9���!�JѦM��������b�޽رcΝ;���txzz"//OQ�G4�&"""�c��`mm�M�6��lmme�����<y2�u� ����mۆ�C�"%%6l��-[кuk @HH���q��q�k׮p%2� Q���~ԯ__�5���Q�n]�[�N�?::			h۶��LWWnnn8�<  ""999ru,--���$�S�1$"""��hH�@jj�ܖ�����|��V�Z{{{=z��$    IDATÇ�w�}���` @BB ���B�8پ�����L�2��S�1$"""�Q�����[[[���D�͛7�o����z��!   u��ŷ�~�!C�`���Ey�*�c ����X�������쵮��G�/_իW�+�V�v�� �J� ���DXZZ��$&&��I�Rdgg#99Y�011...�� ������F�������O%�M�4�ݻw���ݻ ����R)BCCe����q��i��� ������-W'>>7nܐ�)��HDDDJ#�,�c�����ٳ'.]���k�b�ڵ�?�������=��� }}}xyy LLL�������������0aj֬)�\�1$""��A�ػw/���0k�,���!00}���ՙ4i2331b�$''�Q�F8v����du�����={"33���

������PA�������
�~}��F�~ �H�>�C(v���U���H8��K����Ĥ*RRR���)�=�k�g�CS�p�Myosq��"����'�D���?����s���=*Wn�Y����/vhE��0��O�^c"�8OƬ!���a�\�7i�X>�|�h�8��P��8��\�����������D����)o��2T�ӧ����P�lMأn�v���.vX����+��a__�CQY��-C��ad� s�Z��e��} vX*C������D���oGP�bԨ�˗�cР	0111>b�W�n�� ��4�C����G�O�cJ��X:z���W�܃��11�?,�Lq��],��+̤�ѸMM @Vf6�ݪ�٭��D���M�vC˖.8th3�������(]���&<<k�nE�Z��E��>}#G@�u����)S�m[/ܺ}��#�(&�$�˗��S����� `kk�;��e���#�^���}���~Tj6� �=�����= �>�8���]��%�]|Z �_�_�ѫ�V�ں<6n\,+���1��!==��}��k`�ܥb��Ҏ�*�zӦ%07�����h޼�HQ���I �k�+�$���It:����?p��# ��k�p�\8<<Z���2�� �J�oEhت�<~/^C\;O��ùyU��Ti��ٹz��:�W�+�[�M�TިQSѾ}+�n�L�P����T ��iiqQ�\��-�%@LL���p��UԩSG�p�ذa���㧨Z��������9s&�O��b�&*A�v��hP	���*0���nA��ӡ����c��A��E�Vu=z�իC0v�`���Bx�5�3::�0�k��SI;v��+W�p���b�R����c�ѴiC89�2R]L ����$L�6�Fbb"ʔ)�ڵk���...�H$ػw/�t�"v�*o�ҍ
ډmۖ�FDF��ر3aii��_�+��D̝x��k�\����p�z,f��
��q�!VN�	Ss�m�(R��+??���B@� ��u�p��]�Y��?_�����8vl������5j
����ܹ�b��2т�@�c�u��999F�J�����'N�իWb�V�,�ԩߡw�N ��5���㧘?��~A��������A���A��M�o:�i놠�� ��Z<��{֞`��˛�Z5{��jժ`ϞC"E��""�#)���=deyyy8s�O�X��o��5Дa��ؿ�Μ�++�?@MH4
�����?x��5Ν;��e˖���AÆ���:��� еkWH$���s�ΰ�����!4h��Ǐ˝���4h���P�bE����\�t	u�օ���ׯ��W��e+M~~>���P455�rA�r�N�?r󶍂���o�6MM�������G5iR��=�+�w�ll�D�H���7��롸z��l�_���튫W�0��A0j���s'O�
;��b�D��� ����044ľ}���������p ��M�/{��������ĉ�z�*ڵk��;"66V��E���#F`���g���������������?&L���+V�Ν�b��e8x�bb�w�,Y�]��;�"�r�N��{�~�Rzx���WI��z� (e���M�!�7\�p	q/��O��׶�d�y����7����� ����xx�	�^g�r]b�����" `<�ƶm{�n�6�1@��T���!����m�05-�1m�0r�d�����m�add���$$$$!33S��T�
�v+���޽C�Aff&�ի777����j|	��5j`���5j���f͚a˖- 
���J��9s&���k����qqq��/��z�j>���@�������TX[[�̓@��30c�b��s���aii�޽;c��1���;< E�$����}�|�}���F ��/��� "��FZ��W(���\�է�l�LȒC���<���ғ@~��8&O����c`gg��c�`�/��*6Z���k�@`��ءȨғ@$�
-ߴi1��{q4'�@�~�Rڅ:W^fnL:�'�(� �GݻwG�p��Y\�pG���~�zx{{����̜9���;�={���\dff~��.�
I�T���$ ��۷Q�vmY� ...��y�0s���x��ghh�E��aѢib�"�C1���Z�F�]���K����ol{E�U�yz���g�x�XN��)v*M���S�?��!===�i�ӧO�������3f|��ĉ�{�n̝;gϞEdd$j֬���l�z���)I$�B�����CJJ�l������"""�"+A~#�b�@իWGFF�+mmm�����?{�,���ѵkWԬYR�111_�ժUõk��Ɩ\�x����Յ����FDDT��jb��|��Z�BHH�_����h�ܹ.D�������ĉHHH@rr2 �J�*سg"##q��5xyy}q˞��444���[�n�СC�駟~�DDDTr1�ѨQ#,Y�͛7����M��!C�`��� 
f򆆆���u�� ,Y�eʔ���+:v�v�ڡ^�z_����[�P�n]L�2,P�5)»u ��bq�z73KUfE5�$Q�Y�T��,��@�Y���tP�,�kcr��DDDD�f�)���� ��0TML ���Hi$H4
� �x�� �� �ҰX51$"""�Qēܘ�)�������[ ���Hi���� ��0TM�&"""R3l$"""��:���	 )��U�������[ ���Hi$(�:�l�S<&�DDD�4�VM�&"""R3l$"""��,`��������]���]�DDDDj�	 )ϻi��ھ�-���e-��6�T*�/���aii�R�J�E��y��9���0z�h�-[�ԩ�<y���� ��:����edjԨ���x�%۷p�B,Y�˗/Gxx8�R)ڴi���4Y___�ݻ;v���s琞�OOO���)ⶈ�c �������Ғk�{Gb����֭  88ضm����lذ[�lA�֭ !!!�������Ѯ]�"�e` )��D���Kݿ������C�޽���# @tt4жm[Y]]]]������� �������ձ�������Nq�@"""RE�NMM�+��Յ����5j�͛7�������3g\]]q��M$$$  ,,,䎱����Ǐ 			���A�2e>�����-�DDDT,X[[���D�͛7��<<<н{wԬY�[����t�R� ����.ܿ� ���`ll,+�X����f͚��>�t� HLL�����Nbb�l̠T*Evv6����Z���R�kQl$"""�Q�@ccc��s���,ܾ}�˗����R)BCCe����q��i��� ������-W'>>7nܐ�)��HDDD%ʄ	бcGT�XIII�3gRSS1`� H$���"   ������G@@ ������ 011���Ə333���b	�.咀	 )�D]�_:��ɓ'�ӧ^�x�r�ʡq�Ƹx�"lll  �&MBff&F����d4j�ǎ�����������BϞ=���	wwwASS�Pע*$� bAE+55&&&x��&����� �}b�P�|U���!P	&�pӗHMM��IU���ȍ�S�{|״����:�:WΛl�z�i�%?ADDDDj�]�D����V�������Xia�Y���Bh|��|?rR,&�DDD�4����� �b0��a )�"G�������F�O!�)�	�ڵk?���~��H����TG�N g̘�Y�$	@"""%`�j*�	`||��!�5&��I�f����������;""""Q�M���[�9�J�B�ʕ���c ��q�x�b��#""*�$
��Km��S��?���C����'+o޼9�n�*bdDDD%׻.��n�X%z�_�ڵ[�nE�&M���Q�<x bdDDDDEKm���$XZZ~P���	AD������Ӏ&��X�Ԧ�^�z8r���AAAhԨ��|t��I ��6-��С�ݻ���<�Y��n�����&vxDDDDEFmZ �7o���0<{����عs'tuu������HI$�l�Xj� �����_����Hmp!hդV	� 8x� n߾ �^�:<<<���6�DDDD� ޹s]�vEtt4*U� x��lmm�w�^T�VM����J� �&�i���񁭭-bccq��-ܺu������Ð!C�����D�@O!ES��+W� <<���2sss,\�612"""���6	`�*U�����_�z%�&"""�b�j*�]���ٲ���~1c����ŋx��~��w�;�/;T""���VM%�POOOn�pAЩS��ڷo���<1B$"""*r%:<|���!�5v��� �k�N����ԚD��峀�D'�����'O� ;;[����A���������$�/_��СC��o�!??���HDD�x����� �*ѳ��jܸq����ɓ'Q�T)���oX�f*U���{����KKK���?lm]��o�&M�"<���a������6���a��	�_�ۿ`�64�5֦_����zLC��w>z.Aг��G�SRl]v���G{���Zk�Z����{|?S�W���8�w������l���s���ǡe�HOyS���RΜ����Ҳ$�
ط��!+W�ή1��*���+�=���!�H��b�M���͚5���1x�`,X������!C&���ؼ9ׯ��M�fh��O�&�Z�{�&5j�aa�Џ�l_�����p��T�1G�x�<僺���/�cg�]|�.ܰ��D���;���c��2d�ɒ�y��uY�*X�k,ևN�7���Ֆ����F���w4�gd�A��ձb�\�C)6~��7���cʔ�p��Q4k���T�Ј>Jm�����P�|y @�2e���s��ۣ^�z�t��ѩ��̷ؽ�0��[��� ����ߎaժ-�3g�����Ѻ��'����&�z�B���fT�ZՖ�G]{�?�É?��� ��+��[Gɽ�~�7�Z�{ܻ�ڍ� �G�V50lj7Y=K��r���
 y���#V}����J�0��ŋ��ǧ7� ��ѣ��j�f̛�'rt�(`pI�CVj�������� jժ��7��˗ظq#,,,D�N����"//zz�r�J��?�E��x�����GaV����͛7Y�v�",
i#,z��  �� ���|\<qV��1�k�֚��q�H��QRI�������h�V����m�p��e��R�fv#�R�pԨQ��� L�>{��9,X�Y�f���322���3��Y�g�������=��ϫ��O;<�t�P8*����I,_���NGS#���ףa�h߱��Q=A��j6���� ��/Ґ����+��a���q�h4��6�^��l���{�����`a!ߪlaQ		�F�Im��(�w��7n������"F�e$	��݋.]���Bm޼>>ae�����W�	^^]%vh*��[M�]
���ز��v��,F�r8���8v��;�"��_���3,�;^V��/  \����ߺ �8Y���G8���p	("e�BЪImZ �������*��%%%a�С�X�"tuu!�JѮ];\�p  @LL$	"##E�X1*W�EX�N���Al�E������R��b����P��%4���k����>��� ����G	�d��]`n� ��{>:��,b�ʵt�/8�:���E9����&����Ҁ�}y����r������eM������r剉/ �����ೀUS�n�<�������޽;rrr�J�*!11'N���W_TR�T����@�HN~��G�`��<��A@~^���c&��7���o�<s~��W���R	���SŹ#�X�s,�W�����B��6�{(�4̓GI��2-�P����с�s-���A׮����3�ܙ���6�|���a̘1P�{c�̙X�v-���ѨQ#�X�5jԐ����	&`�����̄��;V�\	+++��D�Jtx�ԩϪ�*�K_�~�s��!,,nn��mllаaCY��v��� �֭ pssCXXX�ǭG��� pt��b0iR �V���{�Z�KO�D��x��ؘDD]{�2e�P�������<B*5ūWiظ��=}�Nݛ  ,�e>:��ʺl�J���w�ľ˘�q(�u�*�`9�R�-� �5�f߀Z�����.�����(���UR
^%��i�s ��;Ϡo��
�0.cP��%���<x-{���05-��+���7n��f�ׯg�]��ا6��C����ǚ5kP�V-��bɒ%شi0g��i�w�ޅ�Q�xj___8p ;v쀙�ƏOOODDD@SS�Pף
Jt��۴�044���!��ۇƍCWW��_�t	6����Q�F���|�^VV��ޯ�����и!%%�'/��'	055A�n�1w�Dhkk���%Ld�tn7E�z� ���Za����	v�9�W/RQ��u�����Q��zv���| 0�G�\�����W�\  �<�`��>ض�(�M�	�J��nj6���<[�"x�!��1�puq��5�l�����q3 |���@��Rm�zu�˗ɘ5k	�����C���Ʀd��Xc ���ѷo_�[�s�̑�����@L�<ݺ,l۶C�EJJ
6l؀-[��u�� ���X[[����h׮���JA���۽{7����LԫWnnn�ݻ�쯗�� ������W�^E�:u>yN̜9��ׯo����#G��%gG�B�s���C(VZXv;*�RS�`bR)))066.��L���	|C�A���4�MVF�,��� SSS,Y�-Z�@�:u�G��r�ʸr劬 :w�ҥK#88'O����;^�z�2e���Ԯ]]�t��wjq���@TU������3�߿�ڵCXX�ի�����|N???���ȶ��8�LDD�4�I�_���n�Վ;p��̛7�}		O�����}			��ёK��^��c�����ЦML�>�ϟ���7f̘�ϧ��ccc�����((r!hkkk���ȶ�%xqqq3f�n�
==����b�	`1P�zuddd|P�n�_^^^Q�DDDT�����z���>\)"""III�W��������ӧOc�ҥ��Ғ��%&ʯ���([iC*�";;��ɟ�S�1T!/_�D�V���ׯ#::;w��ѹ�����Q�T)9r���HII!j""�OS�:�����dIwwwDEE!22R�կ_}��Edd$*U��T���P�1���8}�4\]] �����֖��7n��w%z���ܹ�W�Ftt4���P�bE�X�vvvh߾�������5%K���C������C��蚆ZZZX�t)f͚��ӧ�Y�f�v""*�4 hr��/i�222����\������d徾����=��� }n�wd    IDAT}}xyy LLL�������������0aj֬)�\ܩM�~�zL�4	#G�ą��� (U�-Z�	���.�͛��1��}�����1x�`e�FDD��H��_aϡH�&MBff&F�![�رc�5  00ZZZ�ٳ�l!蠠�� �F��ԨQ�f�B���add�k׮�R�J���B�V����s�C,2��s���e`����2\��I�e`~81���\&=��,��K:�i|���ׯ�A������E������(`!hUybWI�6�@lll�Ayhh(�V�*BDDDD%��D1)�ڴ �;�F��-�r��5�ݻ�f�����E�������M8t�Pdggc�С���@���Q�lY��o��n"""eP�I �F	  �=�G�Ɠ'O���kkk�+ ""R"�,���!�J ߱��;""""ѨMX�Z�l��u�VFCDD�1���@Om@ooo��999�z�*N�:___Qb"""*�$��/��`�S�����hy`` n޼Y���Gm����;��_;""�I�l�Xj��)�����a�H���&tqq�� ������Y�Ȉ������$�-Z��{����r�ʡe˖�]��8A�p�D�@��hj� ���N�:hٲ%�������Hm(b� *�ZL��҂��7233�����Htj� @�p��u�� ""R+��� �uc��E0 �;&L@bb"���a`` ����A�Ȉ��J.�TMj� v�� ����x��$$	���D������(�Mx��m�C ""R;	4
قW���C%>4h~��g8::�
���@�'�̈́�"T��ipp0g��E�oA�����'���� ��!""� �&�H �5	|��UECDDD$.�H gΜ	�� ""R;�s.�9H��"�ݻ7�LDD$���+�9H�J�,`��#"""�W�[ 9���H<�VM%>���;"""��Y����w���H�&�hB"�;�b!37K���:f�b�P�\{yT����f���>'��&&�DDD�4�d������HͰ�������@T@"""R	$�^��c �]�DDDDj�-�DDD�4(|k[������������n_bժU�U����all>|X�_�������J�B�-p��M�sdeea���([�,ЩS'<y�D!�D0$""����
���GDD._��V�Z�S�N�q� `�X�d	�/_���pH�R�i�iii�s���b�޽رcΝ;���txzz"//O��R(&�DDD�4�D!ۗ�ر#ڷo{{{888`�ܹ022�	A�ɓ'�[�nprrBpp0޼y�m۶ RRR�a�,Z��[�Fݺu���(?~\���1$"""��P� ���r[Vֿ?�)//;v��۷oѬY3DGG#!!m۶����Յ��Ο? ���@NN�\KKK899��wL ����X�������l�7o�'�FEE������2d~��W888 !! `aa!W���B�/!!:::(S��'�w�LDDDJ�_�p?v ��������\WW���8::"22)))صk��郰��B�Q�0$"""�Q�@����:::�R�
 ������X�j&O� HLL�����~bb"�R) @*�";;���r�����pqq)Ե�
vQ�'���`gg�T���Pپ��l�>}��� 
Fmmm�:���qㆬNq�@"""R����K������+VDZZv�؁��0=z�������=��� }}}xyy LLL�������������0aj֬�֭[�jT@"""RE��\III�߿?���abb�Z�j�ȑ#��mҤI���Ĉ#����F��رc022��#00ZZZ�ٳ'233��� hjj�ZT�DA� �h������))w`ll���f��i�E�@KO��������X�m�N�����4��TEJJ�g��+�{|��� �F��}�&�-z���H�/��HDDDJS��_!'�(&�&�DDD�4I�V�s�b1�&"""R3l$"""��(`��N"�1$"""�������]�DDDDj�-�DDD�4���� ��h@��e`�*�������[ ���Hi�x�;� ��V�
F�Z�al�ccG��t���'�K4�݂����\�[X~�#.}P���'�s>�Y����tl9O��%��]�l�{�~�^a z|5��YEuE揳7л�,T��2zqp�پ��\̘W�Q�`���`ؠň�RV'6&e�:~t۷����T«�,��G˪c���2�܏�����E�̝��lW�65�b�uH�O�;��q���=YEK����o)��ru�re��CO�����ٳ��Jx7��)@��Uy̟�ø|�Z�j�N��ƍ;b�&�7o�P��s�|t̣tm;�*`��8v�'�~���:�:�E��s�ܽ6~���Oχ�Я��Q�>�o޼�SM;�8�#��p��CL�녰��ؼ�<�W�9�:���N�f��o�K�u;碼�"q�.z����S�z�x���a�׋��Q�����lܾ�C�wĎ3�(x?L�o��r�V�3��s����`h�����/�e��_~���?�L�W�E�f�����O���$� bAE���SR����H�p>�Դ~�q*||��
 �i�yQ����k��>_ul(+1`	��5�t�w�<�c��h޲&N�]a~��V���_��눐_'�C'�Oֹr�ܛ���{`]���u�7��u*cٚO�cE{����+)���hU��gWǏֹq%������Q���u�݌CO�8>�v���T۬����K4j�z���j�|YY�jn���+̛�'bdRS�`bR)))066.��,��9xg1�J�\i��Pu\��_ҕ�� *����c�ox��-�5k������|�8z��X�o�9�m��~r��/�Rp5�>�ʙ���Ա��������"F�:RS�@"�����G�G^y��k��ϻMG&���L �I�O�I+�gF&�ݟ���߶��MeH�L��*���FD�u�m�&W޶�Ο�,RT�����n�XL I%DE݆��=tu�0d�D���j88T;,���y
2��b��}hѦ�퟊�:6���p��M ��D ��y��˻5B�MA�:���s=P�1Zo�fc�`���c�'3[�����5�T+�芞 �i��ml�*լ>Z'�m�����`��V�_6����p��Ǚcװ|�Xhk����/^!//e��-,�"!!I������}RI%9:VFd�1���a׮���g$��v�ٹ�ء�����m;�ǐQ� �����]�l�K��_�ߠ6��MK �Sm;���/[N�of_q�YNN.|�Y���|��t�G�dffa�/g0ѯWG'�y߇���':��.ʜ�\|?d5��L��������j�E�kl^q#{-���3�oX�����4P��&�V)�x{{C"�`ذ��9����EX1����*U���\����Q��X�j��a�S3#hii¡��\yG+<}� `.- ��*ߢc�XO�^I��&''�.��D�=8�������o�лo�"�����a+N���}�`a�a�mNN.&��³��X�{�� `d���pvu�O�F )!�]-��UJٲ����Db���+1��R叇Tu�VML ���;v�@ff����۷ضm*V�(bdœ ���;������Ε�����G����uA����9,ʗ��{���<��U�rE��x��=|��́�٧����ó!ʖ3)��� ��}N���{'��͇?�GIX�{J�~|��'���o�����k!4�\yh����)*��PA�իkkk�ٳGV�g�T�Xu�֕�eee�ﾃ��9���дiS���(�oee�U�Vɝ��ի�H$x��  %%�~�-���all�V�Z�ڵkEp����7g�\DLL��ncʔ����}���(2�3q�z4n^� �=N���Ѳu������c��~�M����R0+R"�`�ogl\}�｀���q�<����������LD]{��k���1����q�I���ŀ>�q5��M@^^>���������<z�����7ۊqE&`Ryk���P/S�"1o3�$�ĭ����<AV''; �$&	�Vd⟼ĵ���
�z�hں���'�q�`���ظqn߾��cg 6�)����\ݰP5q�4�6mB߾c�6n܈�",,LVgҤIؽ{7���acc���]�vx��LMMѧOl۶Ç���u�V����R�J:t���):�Y����w�LM?�����BV���SSS�w��������`bb�Z���ȑ�hݺ�ء��ڕG���_�z�� ����aɚQ����~�����Q��k�N@C�����P0��`�NNG��6ؾl+I��r�.2�:��,{=e� @�~���T/��`1���t9p4 M��'+!A�aY��Z�EI�s�) �����g.��}�")�5D z�𗫳n�$4hZ:�ڸr���	E����3F=G�Ӳ�DG�^���e2f�Z���$899�С-�����u�1���� *���7^�~�u�����w�� T�Zqqq<x0J�.�+V�L�2

���  ''�������ĉ�z��!&&+VD~~>*V�???�9'O�D׮]���]]]YU�T��I�����~���?fΜ�A�*��j�Z�8c��L�u �+U[PՉ����?+d���c��1�V�r�ʡC�

¦M�СC�-�~Y��"''M�4��ikk�aÆ�}�`��:u�Z�jضm ����HLLD�^�#""���333ʶ��h<|��q���!%%E����)��a�jb��4�F� �X��?��o߾ضm~��l۶�ڵ�%����(_��\��;�K����tuu�Z�����6�c�hlT����
������A�v�]�+W������YYNN���Q�zuY���nܸ����ڵ������W�����*U��mmm$""R�l�XlT0MMMYw�����>>'N���)*V����͛7���ճ�����+|||����N�:���n�...�ҥ,X GGG<{��B�.]P�>� ""��P	�i����󑟟�o��iii�_�>�=�2e����۷/F�����C_������)S0h� <�R�͛7����Ү���运(���]��w�������,���Y�_����g��2b�>�h�G�$�%��2Ѽ�H�V �$"""R3�&"""��H
���	 )� �&v�� ��h@�B��j�P� ��HP��x0�S<v�� ��hH��i�
������F"�@R����Ӈ�LDDD%ʼy�РA���]�t�ݻw��� XZZ�T�RhѢn޼)W'++�G�Fٲea``�N�:�ɓ'Ey)J������F���K�>}#G��ŋ���\�m��:.Ē%K�|�r���C*��M�6HKK������޽{�c��;w������D^^��*�]�DDD�4b�<r���M�6���h޼9A@`` &O��nݺ ���aaa�m۶a�СHII���e��n� kkk?~����Q���������T�-++볎KII ��� ��������m����������ϟ DDD ''G����%���du�3&�DDD�4]�������5LLLdۼy����A�رcѴiS899   ru-,,d�����2e�|�Nq�.`"""R	P�I��?>..��Ʋb]]�=tԨQ���¹s�
D��@"""*����K G������ԩS�����K�R @bb�\���D�>�T���l$''�Nq��������ݿ��<� `ԨQسgN�<	;;;��vvv�J���egg����puu 8;;C[[[�N||<nܸ!�S��������$p;Ǘ9r$�mۆ�~�FFF�1{&&&(U�$	|}} {{{���#   ��������������affSSSL�05k֔�
.Θ Q��j�* @�-��7m�ooo ��I�����#F 99�5±c�`dd$�---������pwwGPP455��R�F"� vT�RSSabb���;066��O3�����f��'v�����b�P��6+�k����4��TEJJ��$
�g�wM��02�/Թ�Rߠ���"���c )�]���8	����HͰ�����F"�@Rȅ {<}�	 )�(t.�?�c0��a )'��&&�DDD�4���LDDD�f�HDDDJ�I ��	 )� �&v�� ����g��7;�b!='S��������X�m�#v�ʭ��b�P�����;��@T@"""R�TM�&"""R3l$"""�a�jbHDDDJ�Y���]�DDDDj�-�DDD�4lTML ���Hi$����� �b0��a )��U@"""R&���]�DDDDj�-�DDD�<
��@�cHDDDJħ�"v�� ���Y���	 );�U�������[ ���Hi��jbHDDDJ�1���]�DDDDj�-�DDD�4���� ��p�jb0��a )�DR�I��xl$"""��(�/q��t������H$طo��~A���KKK�*U
-Z��͛7��deea���([�,ЩS'<y򤰷Ce0$""�%##�k�Ɗ+>��X�d	�/_���pH�R�i�iii�:���ػw/v�؁s��!==������+��P*v��t�_����'1y�dt�� l۶C�EJJ
6l؀-[��u�� ���X[[����h׮]��G������F�.�����m�VV���777�? ����:���prr��)� R�{�4	��i��01l�����J�m�~A0k�Z�X���A3�n57o>1⢳f��h5������H���G������e���Y��P��������ǥ�NN��I��U�I�]�-�|1��-HKySԗ#�3g�D�N>����vط����'@S�Nnsu�*R��k�� ��5��^%8;��g�;�"�n��t�G��C��aF����L���!�µ��0��Q�?8ϯA���qV�����QT�P�����mYYY_|��� ����\����l_BBtttP�L�O�)� R�JNNE�fC����Ƶ�`�c`R�HV�7ci�v.�������ۍFZZ���~����/Ǧc������M��_r�>���G1m�7�yb&ʙ�`P��HO� $ſFR�kL���`��oq��uL�n�X�U�222Q�V5,[>�u�}冧�.ɶ�n*�U�/��__L���^=�f��ãbc��Z�
��.���c��iX�g�r�0���r���7Y����f���y�ff��{Mױ(�V1m���5LLLdۼy��RJ��"���Ͱ�6����ee������e?���~��ڭ% `c�X��
;�Ő�݊<梴~�D���V���(܌�F�&U!6�>���:�m� ����E���}���
խ�l�w�sT���ة_c�����̓��f�^�<<Z�ã�?���ՁTZ�h*�/^��<� 8G��ƪU�1o�������&Ƚ��|0�9�ƭkѨ�Z ЩW ����<O��c�.����:%�
��Tp|\\���e����_|&�T
 HLL�������D�>�T���l$''˵&&&����?]��a �����s5���*Hۡ�s?lX�O�?:�^�u�Ʋ2]]4k^.\!bq���Ꙕ1 <y��SФ������64q��K���<o`hTJ-���u:�"��Qձ%����^������FD�u�m�&P
    IDATW޶�Ο�,RT�A��,m(r$����Xn�/	����R)BCCee���8}�4\]] �����֖��7n��wl�"��)֬ރ1c���@D���X�E���B��HLx	 ��0�;������?v�K̛�΍�P�
 �<1 `V�D����	��}<�I~��U?��^�-�p1��W-УG{��T@ttfL_���}~y��B)i^�x���<XX��+��(���$��� X0e�5v���?����x\zz:<x {���H����bŊ���E@@ ���aoo��� ����˫�����>>>?~<���`jj�	&�f͚�Y��@�����ׯ?Xp����χs�j�3w �n]Gܼ����A��"G�ZfO܌{��`����驙�k1*;V���(.�b�W��?kNN��_�*�5����Э�W"FF�lΤ-��	��"v(���1|�;��|�2Z�|�G�q�  @PP&M����L�1���hԨ�;#���������={"33���

��f��IaHE�|���V�N��jU[��s
 `!5 $&������Y���d����I�q��U��i�����,
Z�^>O�E���R^&����|�`zZ&����X����q�����Q��%܏;�P��)455��(ߪ���R��HQ�k��[v�*�N��L�jjѢA��~�D�������.�-[�e˖)!B�q�����"00P��N�:r?l�k֬���'���Q�jU�;w��݃����ꊇ�/����:u�`͚5������>���k�~��h.��\\k��=�%�ߏEE����vv��J�p����&��sp�����*�X� fM܌��#��X��OR��)�r&8��#���s��]�mh/+KÖ́O�������mc���Sd�P�x�
qq� -�I! ���g�Z=#Wz���E�J� `Τ�8��el���>��94��"�l����߿?"##Q�Z5|��72d&O��˗/C�5J���_Łp��DFFb�ȑ"]����?/���y���A�o;����a��$ţ��ƂyAط�n�x��3�����^���3kB0�z?�C=<O|�牯�63@���?��,>���/�ޭ'��z�:��Q03�]򗙑���|���);O^^���Wd��3y���  1�q�����اHO���	sq����<AX�Et�<e˚�kג�3��ƍ���c���}�>Ǝ���ا6��C+R�'n��^�µá���$P������xx�  �~nG=���ײ:�_�v�c�>J ܿ����urz�^�(�=
��)��D6p�@��Y�v���L�6M���1c�`���rǼ}��7oF�
  ˖-C��h�"�����ʒ[,355UY���7����bꔕ�;{l�,�h�8x�}?�j�������w�"99���#�`dd Z�Ee�Ɠ ���r�+���W3 ��1��m6fMF��7��\	vO��Q) ���1�v��ոm=�ee�_[��%����(���#{=~� @�ݱr�Dݸ�-[����T�/_-Z�`��e02���wz�ꌗ/�1k���'����m���zM~����I�����Y>]���<u�*��Z/�7a�J ��I]0���t
+����!���%&�"�U�}��U�k֬)W���[�����>�X��,� �����ݻM �͛��3g*��X�f����_x��g|��3�-¨TÝ���ZG"�`��0�����بi��:OI֢Ec��Gr��#�}>׈�1�[�0Du�U������_���?t�%��F�Y�����$@����������ߒ�7q�,?��w����!%%E������s}�TElT�r��!>���u������t�ė���ųg�d+�_�xptt�h}]]]�mFDD")�e`��1�V�V�Za˖-8{�,nܸ�(l� ===0 ׮]�ٳg��wߡgϞ��%"""�;� *P~~>��
n���������	̞=[a-�U�TA�n�о}{�z�
�۷�ʕ+rn"""�Rܳ�Iq� *PRR�T���y�;v��?`� ��#hkk�A٧�>|8����������@TSjHNN��������@"""*��� �Bxx8Ə�Ν;���a��a� {��-����g��TE��DDDDj�-�DDD�D\P1$"""��@�Bv8�x��(��a )��U@"""R��`+�9H��LDDD�f�HDDDJ�u U@"""R>X51�&"""R3l$"""%�,`U�������c U�(��a )��U@"""R>X5��� ��X1$"""%c�j�LDDD�f�HDDDJ�u U@"""R�D����.���!��DDDDj�-�DDD�D���� �q�*�%""�i�ʕ���������q��Y�CRL ���H�$
ھ�/��___L�2W�^E�f��������_R	�������ݳ��}�ŋ����F�j�kkk�Z�J	WY�0$""�%;;h۶�\y۶mq��y��R-���A  ��f�I񑞚)v�N�$M�����!+�L~��������QJMMGag��HMM�+��Յ����_�x���<XXXȕ[XX !!�P��L �PZZ���MG�#!"������"y/H�RX[7P��amm-W6c����+���	����D\\���Tju���TX[[#..���b�S,�}ޯ/���eT�~	����4XZZ�{���!::���
9� |g}�� ʖ-MMM$&&ʕ'&&B*�*$��	���Ѐ����a|�������Tu�g_�����~}U�_E���Wzzz���+����с��3BCCѵkWYyhh(:w�\��"&�DDDT�7�|��ׯ�]����6l�ء�&�DDDT����/_�ĬY�''':t666b��� �����Ō3>9��>�{�ex���ח��R=#F���#�C%I1��h�4��aHDDD�f� �&�DDDDj�	 Q	�� ??_�0��H�0$*��]��f͚a���j���~A�nი�#"u��� �1$���_�W�\A\\��O�k�ƚ5k��s�"//O�D����{��a֬Y�H$عs'������C�C�,??_�G�šC�p�� P�簓z�BФ�^�~�2e� ??����C�N�p��q�C+޵t2:::4h1j�(hkk��(nܸ��3g"**
{����M�P�re��*qA`r����.�;v,6o��y����Ê+`oo/r����H*gǎhڴ)�ݻH$����t��([�,�N>û/��G�"==���?~<-Z�V��s���ŋ ݺuðaðg�t��9����g����ؽ{7���3dee1��|��'O�رc�ɓ'���c���III >�ML�LL Ie��R���D���1d�ܻw ������\�����3hhh�����ҥ2331i�$�9S�LA@@@���y��t��I��˕+�����ȑ#����#V�ŚD"��ݻѾ}{�_��V�B��l�2�S���ݰa�L���5k�nݺ�T�jժ�K�.!!!&L  YK!Q��T����e�޿�жm[���U�{��p��M�B�
Brr�x#���B��݅�����+V������l�T�����N�8!\�xQ�z۶m����0i�$!77WV~��բ�ػz��P�|yaӦM� �ӧO�D"�����B^�z%>\�J�B���e噙�� ¦M�GGG�ٳgB~~�Xa������;	%:�<�6m
mmm4k����011��kװo�>T�P			����˗/q��}ܾ}W�^ENNʗ//�%����<�\�*T���'A@~~>4h���8BKKM�6-Q-��������_� ���D�^��r�J4i�*T@�Z��������Z�*~��g̘1^^^(U��ؗ��.^���bɒ%�s����ѽ{w̟? ��������n��;�J����=rss�{�n���I�&��*z��]�8q>>>044+lRc�B*�r�ʘ5k-ZMMM|����ԩA��5k0j�(�~������>444����D�}��������F�v��n�:ܺuիW������C�
�aÆ�lٲb����(���5jԀ�� �ҥKhڴ)��� ����w�������_�#G��ɓ'�����W�z��M�x����������m�����V� <xW�\����\�{I��پ��݃��.��������C�"??˗/Gvv6����lܸ���(W���ѓ���轤�$a�ܹ����0�|Y��}��nݺ	u��=z$� �}�VAHII%VU���ɓ'��۷e]�׮]ڴi#t��Y�u떬��q�M�6	����īL�o�ʔ)#����ӧO?����*���gϞ�u�޾}[ؿ�[��+�,X B�=333������N�ޘ1c�N�:��g���O�J�����ЪU+!==]A�w�0v�XACCC(W�����-t��A���aD���Tʧ��={��[�Z�h!\�vMVα3v��%T�RE077Z�h!>|XA8p��ЦMA*�
}��<<<###��͛"G�xo޼z��!�9R�<;;[x�葐��$� |��WBŊ�s�Ε�q��(�-[V�GƬY�+++a֬Y� ����???�L�27D���:tH�T��p��a���B��ʕ+			� �Ç����U�V�M�&;���DE�	 ��gϞ	���� +?p��иqc�]�v������(�J�*?�(9rDh޼�P�~}a۶m� �>���OB�Ν!**J䈕#;;[hڴ��l�2Yّ#G___���X���z��!BAhbb"\�pA�p���ׯ4֯_/B��լY�ccc���Rprr�U�&\�rE�H���[�BCC��~�I��?L�ݻ'����*U���A�;w�&L�+Vy�D%.�F��?��Ν;�e^Z�j ���GPP�ϟ�~�~~~ �#G��z��X�������ׯ��ɓx��	~��' @ff&��틸�8����gϞ��Z���%v`~jj*5j�f͚aܸqػw/�������͛����f�����1u�T�n��W�F�*U�]��l
�7z�h8p w�܁��޾}���$�;w������S�	Y�/K�,��G�p��Y4m�˖-����>������s�������=5k�i�&̞=C��RH���}��z�u�g��J�*B�J�'''�e˖BVV� �/fffr]&Tp����gggA"��۷�۟��&t��EpuuV�Z�6-�'N�������HX�z�p��}A
Z۶m+���G�(��S�N	�V��޽++KHH�ԩ#k�RǱk��s�
FFFB׮]�Z�j	R�T�������}�J�*��gA(h=�6m�����"���� R�x�K�ݠgA�cǎ	���5k���ta߾}�D"\\\d��ㅩS�
+V^�x�1��� ���U�T���/�唞�.�j�Jhݺ����k�,Z�����˗��ϟ˕���	_��0u�T!//O-��/1l�0�N�:�����t�Rٸ�Q�F	mڴ��S��dtt�0p�@���ϟ?Z�l)���	ׯ_��'�ޤ BNNN��J�1L ��<y�D�V�����cA���[6���ӧ����гgO���AhР�,	LLL^�x!Zܪ�S_�			B�F���͛������!���U�*+++K�:u�`ii)ܻwO�p��G�	?���`kk+4n�X�2e�.������b�'�-[��Dppp._�,+OMM�%�k��$�HLHE&>>͚5C�F��e��߿666������;7n�+V ((�FժUq��%.���c����p��<z�>>>����T*E||<�t�===L�2mڴ)�c��THH�����/����è[���!��w?_x��!�?�.]��lٲ���ŭ[��3f�lٲ�r�
�u놭[�BWWW������ӵkW���o��)	 iii�֭���ܸq�*U#d��'���(-�����G��J�*���D���q��\�r���(]�4�<y��/_�T�Rpww�¼(x��޽{ѫW/�)S���ؾ};RSSQ�R%X[[���;v��������/<ma��������͛Q�fM�CRI��v�B�=���lݺ�˗/Q�^=XYY�N�:5j���abb������{bmm�޽{�		A�z�P�bEhhh@WW]�t��ׯѫW/>�T��͏Tr�_���K��ׯ_u�֕C4o�<�\�r��~~~¨Q��>�_\�xQ���6l� B�$]]]�r��¸qㄘ�A
�����e��`�:���/�������6l��Ӆ��\a�رB�f̈́ٳgYYYrc&�M�R'w��,,,����rwww���R��>v����&�x�����ѩS'$$$ ##&&&X�nΝ;'{nhϞ=���t��˖-�СCբk�������nݺaРA���F͚51p�@���k֬��+p��}T�PG��������sss�����J�������e����Q�T)�k����Ĝ9sP�n]�ڵ�����Ѐ��C:::b�^d����rttĲe��U�V���� ��Ǐ�z��4h�=���<��455�,f�/�.`R�/^`�ڵ������ב��CCCԭ[)))8p� jժ�Z�j�iӦHNN���9V�Z'''��/r�&���H$hjj�ܹs077���jԨ###0 5k��ڵkѢElܸ���@�f�J�ZTxϞ=C�z�p������V�Zx��	v�څ~�����999���C���1q�DԪUNNNj�s��ZSSSe�֨Qؼy3`mm���ˣ��ؼy3<x ///1�&�lZb@%˻�ҹ������̙3�آٳg�G�8t�>4l�6D^^���Ŭ�����Xt��[�nŵk�ЧO�8q-[�������Űa�  			�W�*W��!C���}ʽ{����+���a׮]���B�.] �H�����QFF�W���eˊuѹp����������@<|��Ǐ���- �{������ȑ#���___ԫWW�^�k�'Ru�&�x�]��� ��V���k���۷���%K�����|�Μ9+++,[�׮]��G���~i�{�I�6m���6�e˖��������|ܻw>Ě5k��� ???v��giѢ������---�Z�
'O��Ν;q��Q������x���-[���D888�v�����رc1a�$$$���۷o�ڵk���cY����#G�ā0�|ܺu@�q�&RU�&��H$HHH@�:u����������������?�����M�6�ɓ'����7n 66�{�V��r�ZLcbb�m�6��ؠjժX�f���1i�$�+WNvoʕ+��w�b����ŋ�e�Y��?��ʂ��������+x{{����E�J�0f�,]����ؾ};����o�>T�ZU�ЋD�ҥ!�\���g�����7orrrP�J�.] �ׯ_�\�r���u���1*�� �����"''K�.ťK�����ƍ�y��u�N�<�V�Z����5���#�?��� �YN��%QQQ�ԩ���Q�bE���᫯�;w�q�Fԯ_*T�}�xxx�Q�Fhݺ5&O��jժ�|%�����p��IT�VM�2_�ti̞=U�T��I�p��)�;wu����ŋѪU+t��'NT��?���V9;;C���!,,�ƍC�J��p�B�8���X�x1|||����^��A��ݺu3f�@dd$����z�j\�~D�~�кukY]�o�W7w�܁��+��ѣG���Rn�&M���Sl߾6���&v����2  	dIDAT�:@OOO�������Cݺu���+xxx`���S�p�����ؽ{7^�x��S�"99����߿�ء�⯿�����~�zXYYaݺu8z�(f͚��ϟ���:::�����������	 )ūW�p��L�6)))�ٳ'BCC���5kֈ�J���D���aaa��˗��sss����}���s��M��۷ocѢE�w�*W�,b�T<~�=z􀶶6���P�^=���b���(]�4�lق#F����n�1cP�T)lٲEm���T�j�*�x��/_Fjj*|||�����רxcHJ7v�XܹsQQQx��֮]������rrrЪU+����F� =zG��ƍall�ƍc�Ν�ٳ'>|���l߾��3��v��}������G���!�H���?�t������аaC�9s:::�{�.`ee%vآ�k�i�&lܸ*T��y�`gg'��䏊3&��4�%�#G�`�ʕ�t��*�'���hԨ�5k�q��a�޽����7oCCC̚5>>>�6m=z�ҥK��x���޽��c�"//˖-C�
��s�W�^�ׯ�1���I`PP*V��y��}�L%@R���������XĈT�ɓ'Ѯ];T�P�^�?�wwwT�R999��􄹹9�l�"v�T�ݿ_��<}�t4i�D�T�_mܸk׮ŷ�~�A�1a�b�	 ����␔��w���ѻwo8::b֬Y �/*�����ﾃ �:u*�6m*vH*﯉���'����o�>q�"R .XD$2kkk8;;�%��٘1c���ٸ-&TX���X�t)���1q�D\�xQ�T�D"�-�ncc�R�J!;;[䨈
���#R1!!!�/���Ç���^쐨���Ǐ?��iӦ}��}�D"��/p��5�^�:::b�DTh�&R!w��Űa�P�L̝;��<��dgg3��Bo߾���Tb0$R1III���U�u؈�H�� �N!"""R3L �����@""""5�����H�0$"""R3L �����@""""5���T�D"��}�  111�H$���,�8���ѥK�O�

B�ҥ�蜶��,T\����S�N��AD�	 �+oooH$H$hkk�R�J�0a222�����ֈ������g������� -� ��᫯�¦M������g�b���x��V�\�A]A���-������ԄT*-�y���=� �g��ՅT*���5���Я_?Y7mXX$	�=����CWWgϞ 8p �����_{�
���[.������X�,�QX��DI�a�ȥY1��ʂY�%+�hrͥF���H��jJn��{o����~>���;��u��9#� --z�<��^]]�������X__������9*++!���Z�Պ��>���byyYxc��� xxx@}}=����V���[a���Ottt 22111���F���i�Z��j�㐟�����~OOOhhh �qP(�?>>���qqq��d())���Y@�0���@�a~E"����]T��ݍ��AX,(�J�F466���0���A? ��񠦦AAA0�L���@OO��u��qkk���hii�����D]]4�v;�v;
������bp���]��8h4|||  FGG155���I�����tbqq1�����������	���PUU���;Q���a(�JC�ӡ��]H|����p8X]]��l�J�Bii)�Ng@�0����a�GSSi�Z�|xxH���TWWGDD�������D��j5�����H.���h��� ������� -..������""��t���J~�JD499I�����x����w�J�d4��H.���А��v�)11�k�����)""��v"���l���ɤ�hD}��멼����677I&�����IOO'��@DD�������s]�a_�@�a��������<�n7�Z���e^^��l6�qtt$���n��\.����b� ))				B{AA��8NOO�V��w�f����ջ\.X�V<>>�n���F^^^@��///���XYY��f��x{{�z��37��f3���#�������w,�0���a�c||!!!P(�&`aaa����^�GMM�W_�D�8�Ri�c<rss1??����8�����ш��ddd@*����V�f�7V�\.�]�*�O�0���%���%,,�Q�T����q\VV���`�٠P(  &���J����p���&��������caaA�T��\�Ʉ��"  ����;��������� ��	�z��?�>��dBVV����@pp0RRR�^�a& �2��?��    IEND�B`�	zj�