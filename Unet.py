from keras.layers import merge, MaxPooling2D, UpSampling2D, core, GRU, LSTM
from keras.layers.wrappers import Bidirectional

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout, RepeatVector, Flatten, Reshape
from keras.layers.merge import concatenate
from keras.layers.merge import multiply, add, average, subtract, maximum

from keras.layers.convolutional import ZeroPadding2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation

from keras.applications.vgg16 import VGG16

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Lambda

import CNN, Deeplab
import os
    

def UNet(n_classes, input_height=256, input_width=512, nChannels=1,
         trainable=False, sed_model=None, num_layer=None, aux=False,
         mask=False, RNN=0, freq_pool=False, enc=False, mul=True):

    if freq_pool == True:
        stride = (2, 1)
    else:
        stride = (2, 2)


    inputs = Input((input_height, input_width, nChannels))
    if nChannels > 1:
        inputs2 = Input((input_height, input_width, 1))
    
    
    if mask == True:
        x = sed_model.layers[1](inputs)
        sed_model.layers[1].trainable = trainable # fixed weight
        
        for i in range(2, num_layer):
            x = sed_model.layers[i](x)
            sed_model.layers[i].trainable = trainable # fixed weight or fine-tuning           
        sed = x
        
        x = Flatten()(x)
        x = RepeatVector(256)(x)
        x = Reshape((256, input_width, n_classes))(x)
        
        e1 = concatenate([x, inputs], axis=-1)
        
        e1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(e1)
        e1 = BatchNormalization()(e1)
        e1 = LeakyReLU(0.2)(e1)

        e1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(e1)
        e1 = BatchNormalization()(e1)
        e1 = LeakyReLU(0.2)(e1)

    else:
        e1 = inputs
        
    
    e1 = Conv2D(64, (3, 3), strides=stride, padding='same')(e1)
    e1 = BatchNormalization()(e1)
    e1 = LeakyReLU(0.2)(e1)

    e2 = Conv2D(128, (3, 3), strides=stride, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(0.2)(e2)
    
    e3 = Conv2D(256, (3, 3), strides=stride, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(0.2)(e3)
    
    e4 = Conv2D(512, (3, 3), strides=stride, padding='same')(e3)
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU(0.2)(e4)
    
    e5 = Conv2D(512, (3, 3), strides=stride, padding='same')(e4)
    e5 = BatchNormalization()(e5)
    e5 = LeakyReLU(0.2)(e5)
    
    e6 = Conv2D(512, (3, 3), strides=stride, padding='same')(e5)
    e6 = BatchNormalization()(e6)    
    e6 = LeakyReLU(0.2)(e6)  # 4
    
    
    if RNN > 0:
        e6 = Reshape((-1, 512))(e6)
        
        for i in range(RNN):
            e6 = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
                    return_sequences=True, stateful=False)(e6) 
            #e6 = BatchNormalization()(e6)
        
        e6 = Reshape((4, -1, 512))(e6)
    
    if enc == True:
        enc = Conv2D(n_classes, (1, 1), activation='sigmoid')(e6)
        sed = MaxPooling2D((4, 1), strides=(4, 1))(enc)
        sed = UpSampling2D(size=(1, 64))(sed)
        e6 = concatenate([enc, e6], axis=-1)
    
    d5 = Conv2DTranspose(512, (3, 3), strides=stride, use_bias=False, 
                         kernel_initializer='he_uniform', padding='same')(e6)
    d5 = BatchNormalization()(d5)
    d5 = Activation('relu')(d5)
    d5 = Dropout(0.5)(d5)
    d5 = concatenate([d5, e5], axis=-1)
    
    d4 = Conv2DTranspose(512, (3, 3), strides=stride, use_bias=False, 
                        kernel_initializer='he_uniform', padding='same')(d5)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)
    d4 = Dropout(0.5)(d4)
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Conv2DTranspose(256, (3, 3), strides=stride, use_bias=False, 
                        kernel_initializer='he_uniform', padding='same')(d4)
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    d3 = Dropout(0.5)(d3)
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Conv2DTranspose(128, (3, 3), strides=stride, use_bias=False, 
                        kernel_initializer='he_uniform', padding='same')(d3)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Conv2DTranspose(64, (3, 3), strides=stride, use_bias=False, 
                        kernel_initializer='he_uniform', padding='same')(d2)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = concatenate([d1, e1], axis=-1)
    
    d0 = Conv2DTranspose(n_classes, (3, 3), strides=stride, use_bias=False, 
                         activation='sigmoid',
                         kernel_initializer='he_uniform', padding='same')(d1)
        
    if mul == False:
        model = Model(input=inputs, output=d0)
        
        return model

    if nChannels > 1:
        d0 = multiply([inputs2, d0])
        if aux == True:
            model = Model(input=[inputs, inputs2], output=[sed, d0])
        else:
            model = Model(input=[inputs, inputs2], output=d0)
    else:
        d0 = multiply([inputs, d0])
        
        if aux == True:
            model = Model(input=inputs, output=[sed, d0])
        else:
            model = Model(input=inputs, output=d0)
                        
    return model



def WNet(n_classes, input_height=256, input_width=512, nChannels=1,
         trainable=False, sed_model=None, num_layer=None, aux=False,
         mask=False, RNN=0, freq_pool=False, enc=False, ang_reso=8):

    sss_model = UNet(n_classes = ang_reso, # direction resolution
                     input_height=256, input_width=input_width, 
                     nChannels=nChannels,  # input feature channels
                     trainable=trainable, 
                     sed_model=sed_model, num_layer=num_layer, aux=aux,
                     mask=mask, RNN=RNN, freq_pool=freq_pool)
    
    # pretrained SSS U-Net
    sss_model.load_weights(os.getcwd()+"/model_results/iros2020/UNet_1class_8direction_8ch_cinTrue_ipdTrue_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/UNet_1class_8direction_8ch_cinTrue_ipdTrue_vonMisesFalse_weights.hdf5")
    
    for i in range(0, len(sss_model.layers)):
        sss_model.layers[i].trainable = trainable # fixed weight
    
    x = sss_model.output

    unet = UNet(n_classes=n_classes, input_height=256, 
                              input_width=input_width, nChannels=1,
                              trainable=True, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=0, freq_pool=False, mul=False)
    unet.load_weights(os.getcwd()+"/model_results/iros2020/UNet_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/UNet_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")

    netlist = []
    for i in range(8):
        o = Lambda(lambda y: y[:,:,:, i:i+1])(x)                # select 1ch
        o = unet(o)
        o = multiply([sss_model.input[1], o])
        netlist.append(o)

    out = add(netlist)

    model = Model(inputs=[sss_model.input[0], sss_model.input[1]], outputs=out)    
                        
    return model



def UNet_Deeplab(n_classes, input_height=256, input_width=512, nChannels=1,
         trainable=False, sed_model=None, num_layer=None, aux=False,
         mask=False, RNN=0, freq_pool=False, enc=False, ang_reso=8):

    sss_model = UNet(n_classes = ang_reso,  # direction resolution
                     input_height=256, input_width=input_width, 
                     nChannels=nChannels,   # input feature channels
                     trainable=trainable, 
                     sed_model=sed_model, num_layer=num_layer, aux=aux,
                     mask=mask, RNN=RNN, freq_pool=freq_pool)

    # pretrained SSS U-Net
    sss_model.load_weights(os.getcwd()+"/model_results/iros2020/UNet_1class_8direction_8ch_cinTrue_ipdTrue_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/UNet_1class_8direction_8ch_cinTrue_ipdTrue_vonMisesFalse_weights.hdf5")
    
    for i in range(0, len(sss_model.layers)):
        sss_model.layers[i].trainable = trainable # fixed weight
    
    x = sss_model.output
    
    deeplab = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                input_shape=(256, input_width, 1), # + 1), # number of direction resoluton
                                classes=n_classes,                            # number of classes
                                OS=16, RNN=0, mask=mask, trainable=trainable, 
                                sed_model=sed_model, num_layer=num_layer, aux=aux, mul=False)
    deeplab.load_weights(os.getcwd()+"/model_results/iros2020/Deeplab_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/Deeplab_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")

    netlist = []
    for i in range(8):
        o = Lambda(lambda y: y[:,:,:, i:i+1])(x)
        o = deeplab(o)
        o = multiply([sss_model.input[1], o])
        netlist.append(o)

    out = add(netlist)

    model = Model(inputs=[sss_model.input[0], sss_model.input[1]], outputs=out)    
                        
    return model



def UNet_CNN(n_classes, input_height=256, input_width=512, nChannels=1,
         trainable=False, sed_model=None, num_layer=None, aux=False,
         mask=False, RNN=0, freq_pool=False, enc=False, ang_reso=8):

    sss_model = UNet(n_classes = ang_reso, # direction resolution
                     input_height=256, input_width=input_width, 
                     nChannels=nChannels,  # input feature channels
                     trainable=trainable, 
                     sed_model=sed_model, num_layer=num_layer, aux=aux,
                     mask=mask, RNN=RNN, freq_pool=freq_pool)
    
    # pretrained SSS U-Net
    sss_model.load_weights(os.getcwd()+"/model_results/iros2020/UNet_1class_8direction_8ch_cinTrue_ipdTrue_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/UNet_1class_8direction_8ch_cinTrue_ipdTrue_vonMisesFalse_weights.hdf5")
    
    for i in range(0, len(sss_model.layers)):
        sss_model.layers[i].trainable = trainable # fixed weight
    
    x = sss_model.output
#    x = concatenate([sss_model.input[1], x], axis=-1) # concatenate mixes spectrogram

    cnn = CNN.CNNtag(n_classes, input_height=256, input_width=input_width, nChannels=1, 
                       filter_list=[64, 64, 128, 128, 256, 256, 512, 512])
    cnn.load_weights(os.getcwd()+"/model_results/iros2020/Cascade_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/Cascade_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")
    for i in range(0, len(cnn.layers)):
        cnn.layers[i].trainable = False # fixed weight
        
    netlist = []
    for i in range(8):
        s = Lambda(lambda y: y[:,:,:, i:i+1])(x)                # select 1ch
        o = cnn(s)        
        o = multiply([s, o])
        netlist.append(o)

    out = add(netlist)

    model = Model(inputs=[sss_model.input[0], sss_model.input[1]], outputs=out)    
                        
    return model



def VGG_UNet(n_classes, input_height=256, input_width=512, nChannels=3):
    inputs = Input((input_height, input_width, 3))
    inputs2 = Input((input_height, input_width, 1))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)

    x = vgg16.layers[1](inputs)
    x = vgg16.layers[2](x)
    x = vgg16.layers[3](x)
    e1 = x
    
	# Block 2
    x = vgg16.layers[4](x)
    x = vgg16.layers[5](x)
    x = vgg16.layers[6](x)
    e2 = x
    
	# Block 3
    x = vgg16.layers[7](x)
    x = vgg16.layers[8](x)
    x = vgg16.layers[9](x)
    x = vgg16.layers[10](x)
    e3 = x

	# Block 4
    x = vgg16.layers[11](x)
    x = vgg16.layers[12](x)
    x = vgg16.layers[13](x)
    x = vgg16.layers[14](x)
    e4 = x

	# Block 5
    x = vgg16.layers[15](x)
    x = vgg16.layers[16](x)
    x = vgg16.layers[17](x)
    x = vgg16.layers[18](x)
    e5 = x
    
    d4 = Conv2DTranspose(512, (3, 3), strides=2, use_bias=False, 
                        kernel_initializer='he_uniform', padding='same')(e5)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)
    d4 = Dropout(0.5)(d4)
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Conv2DTranspose(256, (3, 3), strides=2, use_bias=False, 
                        kernel_initializer='he_uniform', padding='same')(d4)
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    d3 = Dropout(0.5)(d3)
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Conv2DTranspose(128, (3, 3), strides=2, use_bias=False, 
                        kernel_initializer='he_uniform', padding='same')(d3)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Conv2DTranspose(64, (3, 3), strides=2, use_bias=False, 
                        kernel_initializer='he_uniform', padding='same')(d2)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = concatenate([d1, e1], axis=-1)
    
    d0 = Conv2DTranspose(n_classes, (3, 3), strides=2, use_bias=False, 
                         activation='sigmoid',
                         kernel_initializer='he_uniform', padding='same')(d1)
    
    d0 = multiply([inputs2, d0])
    model = Model(input=[inputs, inputs2], output=d0)
            
    return model