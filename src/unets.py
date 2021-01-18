from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.utils import conv_utils
from tensorflow.keras import backend as K
from keras.layers.core import Activation, SpatialDropout2D

import tensorflow as tf
from tensorflow.keras.applications import Xception as Xception
from tensorflow.keras.applications import DenseNet169 as Densenet
from tensorflow.keras.applications import ResNet50 as Resnet
from tensorflow.keras.layers import ReLU,LeakyReLU,ZeroPadding2D

def conv_block(
    x,
    filters,
    kernel,
    strides=(1,1),
    padding='same',
    activation=None,
    trainable=True,
):
    '''
    X - Input features,
    filters - Output dimension
    kernel - (2,2) Convolution or (3,3) Convolution
    strides - Number of steps taken for consecutive convolution,
    padding = 'same' --> Same across the left,right,top and bottom. or valid --> No padding,
    activation = Activation Layer to be applied.
    '''
    #Input - (224,224,3)
    #Conv2D 
    #- strides --> It divides the input. So it reduce the size of the input. Since we are hoping.
    #- kernel --> 

    x = Conv2D(filters,
            kernel,
            strides,
            padding)(x)  
    x = BatchNormalization(trainable=trainable)(x)  #BatchNorm does not change the dimension of the input. 

    if activation != None:
        if activation == 'relu':
            x = tf.keras.layers.Activation('relu')(x)
        elif activation == 'leakyrelu':
            x = tf.keras.layers.Activation('leakyrelu')(x)
    return x


def residual_block(
    block_input,
    num_filters,):
  
    block_input1 = BatchNormalization()(block_input) #Just adding a precautionary layer, of getting specialized features.
    '''
    We can pass the normalized features to a RELU layer, or we don't it depend on us.
    Let's not pass it. 
    '''
    x = tf.keras.layers.Activation('relu')(block_input)
    print(x.shape)
    x = BatchNormalization()(x)
    x = conv_block(x,
                 filters=num_filters,
                 kernel=(3,3),
                 strides=(1,1),
                 activation='relu')

    x = conv_block(x,
                 num_filters,
                 (3,3),
                 strides=(1,1),
                 activation=None)
    print(x.shape)

    #Assuming the size of the blockinput1 and x is same we will add them.
    #For sanity purpose we will pass the x with a batchnorm.

    x = BatchNormalization()(x)
    return tf.keras.layers.Add()([x,block_input1])


def convolutional_block(input_block,
          start_neurons,
          multiplying_factor,
          kernel=(3,3),
          strides=(1,1),
          padding='same'):
    
    output_filters = start_neurons * multiplying_factor
    
    output_block = Conv2D(output_filters,kernel,activation=None,strides=strides,padding=padding)(input_block)
    output_block = residual_block(output_block,output_filters)
    output_block = residual_block(output_block,output_filters)
    output_block = LeakyReLU(alpha=0.1)(output_block)
    return output_block


def upconv_block(backbone,
                input_block,
                layer_number,
                start_neurons,
                multiplying_factor,
                kernel=(3,3),
                stride=(2,2),
                padding='same',
                padding_kernel=1,
                padding_flag=True,
                dropout_rate=0.1):
    
    print(f'The input block shape is {input_block.shape}')
    
    output_filters = start_neurons * multiplying_factor
    upconv = Conv2DTranspose(output_filters,kernel,strides=stride,padding='same')(input_block)
    layer_output = backbone.layers[layer_number].output
    print(f'The layer output is {layer_output.shape}')
    print(f'The upconv output is shape {upconv.shape}')
    print(f'The padding flag is set to {padding_flag} and padding is {padding_kernel}')
    
    if padding_flag == True:
        print('Inside zero padding')
        layer_output = ZeroPadding2D(((int(padding_kernel),0),(int(padding_kernel),0)))(layer_output)
        print(f'Updated layer_output shape is {layer_output}')
    
    output = concatenate([upconv,layer_output])
    dropout = Dropout(dropout_rate)(output)
    
    return dropout


def Unets(backbone_encoder,input_shape=(None, None, 3),dropout_rate=0.5,start_neurons=16):
    
    if backbone_encoder=='Xceptionnet':
        backbone = Xception(input_shape=input_shape,weights='imagenet',include_top=False)
    elif backbone_encoder=='Densenet':
        backbone = Densenet(input_shape=input_shape,weights='imagenet',include_top=False)
    
    input = backbone.input
    
    if backbone_encoder=='Xceptionnet':
        conv4 = backbone.layers[121].output
    elif backbone_encoder== 'Densenet':
        conv4 = backbone.layers[361].output 
    
    conv4 = LeakyReLU(alpha=0.1)(conv4) 
    pool4 = MaxPooling2D((2, 2))(conv4) 
    pool4 = Dropout(dropout_rate)(pool4)
    
    #Middle 
    convm = convolutional_block(pool4,start_neurons=16,multiplying_factor=32)    
    
    #Decoder
    if backbone_encoder == 'Xceptionnet':
        uconv4 = upconv_block(backbone,convm,layer_number=121,start_neurons=16,multiplying_factor=16,dropout_rate=0.5,padding_flag=False)
    elif backbone_encoder=='Densenet':
        uconv4 = upconv_block(backbone,convm,layer_number=361,start_neurons=16,multiplying_factor=16,dropout_rate=0.5,padding_flag=False)
    uconv4 = convolutional_block(uconv4,start_neurons=16,multiplying_factor=16)
    
    if backbone_encoder=='Xceptionnet':
        uconv3 = upconv_block(backbone,uconv4,layer_number=31,start_neurons=16,multiplying_factor=8,padding_flag=False,dropout_rate=0.5)
    elif backbone_encoder=='Densenet':
        uconv3 = upconv_block(backbone,uconv4,layer_number=133,start_neurons=16,multiplying_factor=8,padding_flag=False,dropout_rate=0.5)
    uconv3 = convolutional_block(uconv3,start_neurons=16,multiplying_factor=8)
    

    if backbone_encoder=='Xceptionnet':
        uconv2 = upconv_block(backbone,uconv3,layer_number=21,start_neurons=16,multiplying_factor=4,padding_kernel=1,padding_flag=True)
    elif backbone_encoder=='Densenet':
        uconv2 = upconv_block(backbone,uconv3,layer_number=45,start_neurons=16,multiplying_factor=4,padding_kernel=1,padding_flag=False)
    uconv2 = convolutional_block(uconv2,start_neurons=16,multiplying_factor=4)
    
    if backbone_encoder=='Xceptionnet':
        uconv1 = upconv_block(backbone,uconv2,layer_number=11,start_neurons=16,multiplying_factor=2,padding_kernel=3,padding_flag=True)
    elif backbone_encoder=='Densenet':
        uconv1 = upconv_block(backbone,uconv2,layer_number=3,start_neurons=16,multiplying_factor=2,padding_kernel=3,padding_flag=False)
    uconv1 = convolutional_block(uconv1,start_neurons=16,multiplying_factor=2)
    
    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Dropout(dropout_rate)(uconv0) 
    uconv0 = convolutional_block(uconv0,start_neurons=16,multiplying_factor=1,strides=(1,1))
    
    uconv0 = Dropout(dropout_rate/2)(uconv0) 
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv0)  
    
    model = Model(input, output_layer)
    
    return model