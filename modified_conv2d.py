import sys,os

from keras import backend as K
from keras import initializers,regularizers,constraints,activations

from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras.regularizers import l2

import tensorflow as tf
import numpy as np

class ModifiedConv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 template_tensor=None,
                 clusterid=None,
                 **kwargs
                 ):
        super(ModifiedConv2D, self).__init__(**kwargs)
        self.rank=2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size,2,'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides,2,'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format=conv_utils.normalize_data_format(data_format)
        self.dilation_rate=conv_utils.normalize_tuple(dilation_rate,2,'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.template_tensor = template_tensor
        self.clusterid = clusterid

    def build(self, input_shape):
        if self.data_format=='channels_first':
            channel_axis=1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]

        if self.clusterid.shape != (input_dim,self.filters):
            raise ValueError('clusterid shape error'
                             'should be ' + str(input_dim) + '*' + str(self.filters) + ' '
                             'find' + str(self.clusterid.shape))

        self.kernel_num=input_dim*self.filters

        self.kernels_shape=self.kernel_size+(input_dim,self.filters)

        self.A=self.add_weight(shape=(input_dim,self.filters,),
                               initializer=self.kernel_initializer,
                               name='A',
                               trainable=True)

        for i in range(self.filters):  ##kernel num
            for s in range(input_dim):

                temp_id=self.clusterid[s,i]
                temp=self.template_tensor[temp_id]
                temp=tf.reshape(temp,(3,3,1,1))

                if s == 0:
                    temp_slice=temp
                else:
                    temp_slice = tf.concat([temp_slice,temp],2)
            if i==0:
                self.template = temp_slice
            else:
                self.template=tf.concat([self.template, temp_slice],3)

        self.B=self.add_weight(shape=(input_dim,self.filters,),
                               initializer=self.kernel_initializer,
                               name='B',
                               trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        self.kernel=self.A*self.template+self.B

        outputs=K.conv2d(inputs,
                         self.kernel,
                         strides=self.strides,
                         padding=self.padding,
                         data_format=self.data_format,
                         dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs=K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format
            )


        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)


    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config=super(ModifiedConv2D, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    _IMAGE_DATA_FORMAT='channels_last'

    input_shape=(32,32,3)
    cdata = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2, 2, 2, 2, 2],
                      [3, 3, 3, 3, 3, 3, 3, 3, 3],
                      [4, 4, 4, 4, 4, 4, 4, 4, 4],
                      [5, 5, 5, 5, 5, 5, 5, 5, 5]])

    clusterid = np.random.randint(0, 5, size=(3, 16))


    template = tf.Variable(cdata, name='template')

    layer1=ModifiedConv2D(filters=16,
                          kernel_size=(3,3),
                          strides=(1,1),
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(1e-4),
                          padding="same",
                          data_format=_IMAGE_DATA_FORMAT,
                          template_tensor=template,
                          clusterid=clusterid)
    layer1.build(input_shape)
    print layer1.compute_output_shape(input_shape)
    print layer1.A
    print layer1.template
    print layer1.B
    print layer1.get_config()
    print "########"
    weights = layer1.get_weights()

    print np.array(weights[0]).shape
    print np.array(weights[1]).shape
    print np.array(weights[2]).shape