from keras.layers import merge, MaxPooling2D, UpSampling2D, core, GRU, LSTM
from keras.layers.wrappers import Bidirectional

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout, RepeatVector, Flatten, Reshape
from keras.layers.merge import concatenate
from keras.layers.merge import multiply, add, average, subtract

from keras.layers.convolutional import ZeroPadding2D, Conv2DTranspose
from keras.layers import LeakyReLU, BatchNormalization, Activation

from keras.applications.vgg16 import VGG16

import CNN
import os

class UNet_old(object):
    def __init__(self, n_classes, input_height, input_width, nChannels, 
                 soft=True, mul=True):
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 3
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 3
        self.DECONV_STRIDE = 2
        first_filter_count = 64
        
        # (256 x 256 x input_channel_count)
        inputs = Input((input_height, input_width, nChannels))
        if nChannels == 3:
            inputs2 = Input((input_height, input_width, nChannels//3))
        
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(first_filter_count, self.CONV_FILTER_SIZE, 
                      strides=self.CONV_STRIDE)(enc1)
        filter_count = first_filter_count * 2
        enc2 = self._add_encoding_layer(filter_count, enc1)
        filter_count = first_filter_count * 4
        enc3 = self._add_encoding_layer(filter_count, enc2)
        
        filter_count = first_filter_count * 8
        enc4 = self._add_encoding_layer(filter_count, enc3)
        enc5 = self._add_encoding_layer(filter_count, enc4)
        enc6 = self._add_encoding_layer(filter_count, enc5)

        dec3 = self._add_decoding_layer(filter_count, True, enc6)#dec2)
        dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)
        dec4 = self._add_decoding_layer(filter_count, True, dec3)
        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count * 4
        dec5 = self._add_decoding_layer(filter_count, True, dec4)
        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)
        
        filter_count = first_filter_count * 2
        dec6 = self._add_decoding_layer(filter_count, False, dec5)
        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count
        dec7 = self._add_decoding_layer(filter_count, False, dec6)
        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)
        dec8 = Activation(activation='sigmoid')(dec7)
        dec8 = Conv2DTranspose(n_classes, self.DECONV_FILTER_SIZE, 
                               strides=self.DECONV_STRIDE, padding="same")(dec8)
        
        if soft == True:
            dec8 = core.Activation('softmax')(dec8)
        
        if nChannels == 3:
            if mul == True:
                dec8 = multiply([inputs2, dec8])
                self.UNET = Model(input=[inputs, inputs2], output=dec8)
        else:
            if mul == True:
                dec8 = multiply([inputs, dec8])
            self.UNET = Model(input=inputs, output=dec8)


    def _add_encoding_layer(self, filter_count, sequence):
        new = LeakyReLU(0.2)(sequence)
        new = ZeroPadding2D(self.CONV_PADDING)(new)
        new = Conv2D(filter_count, self.CONV_FILTER_SIZE, 
                              strides=self.CONV_STRIDE)(new)
        new = BatchNormalization()(new)
        return new

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new = Activation(activation='relu')(sequence)
        new = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, 
                                       strides=self.DECONV_STRIDE,
                                       padding="same", 
                                       kernel_initializer='he_uniform')(new)
        new = BatchNormalization()(new)
        if add_drop_layer:
            new = Dropout(0.5)(new)
        return new

    def get_model(self):
        return self.UNET
	
    
    
class Complex_UNet2(object):
    def __init__(self, n_classes, input_height, input_width, nChannels):
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 3
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        first_filter_count = 64

        # (256 x 256 x input_channel_count)
        inputs = Input((input_height, input_width, nChannels))
        #inputs2 = Input((self.INPUT_SIZE, self.INPUT_SIZE*2, 1))
        
        # spectrogram
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(first_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)
        filter_count = first_filter_count * 2
        enc2 = self._add_encoding_layer(filter_count, enc1)
        filter_count = first_filter_count * 4
        enc3 = self._add_encoding_layer(filter_count, enc2)
        filter_count = first_filter_count * 8
        enc4 = self._add_encoding_layer(filter_count, enc3)
        enc5 = self._add_encoding_layer(filter_count, enc4)
        enc6 = self._add_encoding_layer(filter_count, enc5)        
        
        dec3 = self._add_decoding_layer(filter_count, True, enc6)
        dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)
        dec4 = self._add_decoding_layer(filter_count, True, dec3)
        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count * 4
        dec5 = self._add_decoding_layer(filter_count, True, dec4)
        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count * 2
        dec6 = self._add_decoding_layer(filter_count, False, dec5)
        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count
        dec7 = self._add_decoding_layer(filter_count, False, dec6)
        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)
        dec8 = Activation(activation='sigmoid')(dec7)
        dec8 = Conv2DTranspose(n_classes, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
    
        
        # real part
        enc1r = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1r = Conv2D(first_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1r)
        filter_count = first_filter_count * 2
        enc2r = self._add_encoding_layer(filter_count, enc1r)
        filter_count = first_filter_count * 4
        enc3r = self._add_encoding_layer(filter_count, enc2r)
        filter_count = first_filter_count * 8
        enc4r = self._add_encoding_layer(filter_count, enc3r)
        enc5r = self._add_encoding_layer(filter_count, enc4r)
        enc6r = self._add_encoding_layer(filter_count, enc5r)

        dec3r = self._add_decoding_layer(filter_count, True, enc6r)
        dec3r = concatenate([dec3r, enc5r], axis=self.CONCATENATE_AXIS)
        dec4r = self._add_decoding_layer(filter_count, True, dec3r)
        dec4r = concatenate([dec4r, enc4r], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count * 4
        dec5r = self._add_decoding_layer(filter_count, True, dec4r)
        dec5r = concatenate([dec5r, enc3r], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count * 2
        dec6r = self._add_decoding_layer(filter_count, False, dec5r)
        dec6r = concatenate([dec6r, enc2r], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count
        dec7r = self._add_decoding_layer(filter_count, False, dec6r)
        dec7r = concatenate([dec7r, enc1r], axis=self.CONCATENATE_AXIS)
        dec8r = Activation(activation='sigmoid')(dec7r)
        dec8r = Conv2DTranspose(n_classes, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8r)   
        
        # imaginary part
        enc1i = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1i = Conv2D(first_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1i)
        filter_count = first_filter_count * 2
        enc2i = self._add_encoding_layer(filter_count, enc1i)
        filter_count = first_filter_count * 4
        enc3i = self._add_encoding_layer(filter_count, enc2i)
        filter_count = first_filter_count * 8
        enc4i = self._add_encoding_layer(filter_count, enc3i)
        enc5i = self._add_encoding_layer(filter_count, enc4i)
        enc6i = self._add_encoding_layer(filter_count, enc5i)

        dec3i = self._add_decoding_layer(filter_count, True, enc6i)
        dec3i = concatenate([dec3i, enc5i], axis=self.CONCATENATE_AXIS)
        dec4i = self._add_decoding_layer(filter_count, True, dec3i)
        dec4i = concatenate([dec4i, enc4i], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count * 4
        dec5i = self._add_decoding_layer(filter_count, True, dec4i)
        dec5i = concatenate([dec5i, enc3i], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count * 2
        dec6i = self._add_decoding_layer(filter_count, False, dec5i)
        dec6i = concatenate([dec6i, enc2i], axis=self.CONCATENATE_AXIS)
        filter_count = first_filter_count
        dec7i = self._add_decoding_layer(filter_count, False, dec6i)
        dec7i = concatenate([dec7i, enc1i], axis=self.CONCATENATE_AXIS)
        dec8i = Activation(activation='sigmoid')(dec7i)
        dec8i = Conv2DTranspose(n_classes, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8i)
                         
        #dec8ri = concatenate([dec8r, dec8i], axis=self.CONCATENATE_AXIS)
        #dec8ri = Conv2D(n_classes, (1,1), activation='sigmoid')(dec8ri)
        
        #dec8ri = concatenate([dec8, dec8ri], axis=self.CONCATENATE_AXIS)
        #dec8ri = Conv2D(n_classes, (1,1), activation='sigmoid')(dec8ri)

        self.UNET = Model(input=inputs, output=[dec8, dec8r, dec8i])


    def _add_encoding_layer(self, filter_count, sequence):
        new = LeakyReLU(0.2)(sequence)
        new = ZeroPadding2D(self.CONV_PADDING)(new)
        new = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new)
        new = BatchNormalization()(new)
        return new

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new = Activation(activation='relu')(sequence)
        new = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new)
        new = BatchNormalization()(new)
        if add_drop_layer:
            new = Dropout(0.5)(new)
        return new

    def get_model(self):
        return self.UNET
    
    

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

    d4 = Activation(activation='relu')(e5)#d5)
    d4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Dropout(0.5)(d4)  ###
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Activation(activation='relu')(d4)
    d3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.5)(d3) ###
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Activation(activation='relu')(d3)
    d2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform')(d2)
    d2 = BatchNormalization()(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Activation(activation='relu')(d2)
    d1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform')(d1)
    d1 = BatchNormalization()(d1)
    d1 = concatenate([d1, e1], axis=-1)
    
    d0 = Activation(activation='sigmoid')(d1)
    d0 = Conv2DTranspose(n_classes, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform')(d0)
            
    d0 = core.Activation('softmax')(d0)
    
    d0 = multiply([inputs2, d0])
    model = Model(input=[inputs, inputs2], output=d0)
            
    return model

  
    

def Mask_UNet(n_classes, input_height=256, input_width=512, nChannels=3,
              soft=True, mul=True, trainable=False):
    inputs = Input((input_height, input_width, nChannels))
    if nChannels == 3:
        inputs2 = Input((input_height, input_width, nChannels//3))
    
    #pre_cnn = CNN.CNN4(n_classes=75, input_height=256, input_width=input_width, nChannels=1)
    #pre_cnn.load_weights(os.getcwd()+"/model_results/2018_1118/CNN_75_class/CNN_75_class_weights.hdf5")
#    pre_cnn = CNN.CRNN8(n_classes=9, input_height=256, input_width=input_width, nChannels=1)
#    pre_cnn.load_weights(os.getcwd()+"/model_results/2019_0226/CRNN8_9_class/CRNN8_9_class_weights.hdf5")
    pre_cnn = CNN.CNN8(n_classes=75, input_height=256, input_width=input_width, nChannels=1)
    pre_cnn.load_weights(os.getcwd()+"/model_results/2019_0122/CNN8_75_class/CNN8_75_class_weights.hdf5")
    #pre_cnn = CNN.CRNN8(n_classes=75, input_height=256, input_width=input_width, nChannels=1)
    #pre_cnn.load_weights(os.getcwd()+"/model_results/2019_0122/CRNN8_75_class/CRNN8_75_class_weights.hdf5")
    x = pre_cnn.layers[1](inputs)
    pre_cnn.layers[1].trainable = trainable # fixed weight
    
    for i in range(2, 34): #CRNN8: 42 CNN8:34 CNN4:18
        x = pre_cnn.layers[i](x)
        pre_cnn.layers[i].trainable = trainable # fixed weight or fine-tuning    
    
    x = Flatten()(x)
    x = RepeatVector(256)(x)
    x = Reshape((256, input_width, n_classes))(x)

    """
    # GLU
    a = Conv2D(n_classes, (3, 3), padding='same', dilation_rate=1)(x)
    a = BatchNormalization()(a)   
    g = Conv2D(n_classes, (3, 3), activation='sigmoid', 
               padding='same', dilation_rate=1)(x)
    g = BatchNormalization()(g)
    x = multiply([a, g])
    """
    
    e1 = concatenate([x, inputs], axis=-1)
    
    e1 = ZeroPadding2D((1, 1))(e1)
    e1 = Conv2D(64, (3, 3), strides=2)(e1)

    e2 = LeakyReLU(0.2)(e1)
    e2 = ZeroPadding2D((1, 1))(e2)
    e2 = Conv2D(128, (3, 3), strides=2)(e2)
    e2 = BatchNormalization()(e2)
    
    e3 = LeakyReLU(0.2)(e2)
    e3 = ZeroPadding2D((1, 1))(e3)
    e3 = Conv2D(256, (3, 3), strides=2)(e3)
    e3 = BatchNormalization()(e3)
    
    e4 = LeakyReLU(0.2)(e3)
    e4 = ZeroPadding2D((1, 1))(e4)
    e4 = Conv2D(512, (3, 3), strides=2)(e4)
    e4 = BatchNormalization()(e4)
    
    e5 = LeakyReLU(0.2)(e4)
    e5 = ZeroPadding2D((1, 1))(e5)
    e5 = Conv2D(512, (3, 3), strides=2)(e5)
    e5 = BatchNormalization()(e5)
    
    e6 = LeakyReLU(0.2)(e5)
    e6 = ZeroPadding2D((1, 1))(e6)
    e6 = Conv2D(512, (3, 3), strides=2)(e6)
    e6 = BatchNormalization()(e6)    
    
    d5 = Activation(activation='relu')(e6)
    d5 = Conv2DTranspose(512, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Dropout(0.5)(d5)
    d5 = concatenate([d5, e5], axis=-1)
    
    d4 = Activation(activation='relu')(d5)
    d4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Dropout(0.5)(d4)
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Activation(activation='relu')(d4)
    d3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.5)(d3)
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Activation(activation='relu')(d3)
    d2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d2)
    d2 = BatchNormalization()(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Activation(activation='relu')(d2)
    d1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d1)
    d1 = BatchNormalization()(d1)
    d1 = concatenate([d1, e1], axis=-1)
    
    d0 = Activation(activation='sigmoid')(d1)
    d0 = Conv2DTranspose(n_classes, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d0)

    if soft == True:
        d0 = core.Activation('softmax')(d0)
        
    if nChannels == 3:
        if mul == True:
            d0 = multiply([inputs2, d0])
            model = Model(input=[inputs, inputs2], output=d0)
    else:
        if mul == True:
            d0 = multiply([inputs, d0])
        model = Model(input=inputs, output=d0)
                        
    return model


def UNet(n_classes, input_height=256, input_width=512, nChannels=3):
    inputs = Input((input_height, input_width, nChannels))
    if nChannels > 1:
        inputs2 = Input((input_height, input_width, 1))

    
    e1 = Conv2D(64, (3, 3), strides=2, padding='same')(inputs)

    e2 = LeakyReLU(0.2)(e1)
    e2 = Conv2D(128, (3, 3), strides=2, padding='same')(e2)
    e2 = BatchNormalization()(e2)
    
    e3 = LeakyReLU(0.2)(e2)
    e3 = Conv2D(256, (3, 3), strides=2, padding='same')(e3)
    e3 = BatchNormalization()(e3)
    
    e4 = LeakyReLU(0.2)(e3)
    e4 = Conv2D(512, (3, 3), strides=2, padding='same')(e4)
    e4 = BatchNormalization()(e4)
    
    e5 = LeakyReLU(0.2)(e4)
    e5 = Conv2D(512, (3, 3), strides=2, padding='same')(e5)
    e5 = BatchNormalization()(e5)
    
    e6 = LeakyReLU(0.2)(e5)
    e6 = Conv2D(512, (3, 3), strides=2, padding='same')(e6)
    e6 = BatchNormalization()(e6)    
    
    d5 = Activation(activation='relu')(e6)
    d5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), use_bias=False, 
                         kernel_initializer='he_uniform', padding='same')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Dropout(0.5)(d5)
    d5 = concatenate([d5, e5], axis=-1)
    
    d4 = Activation(activation='relu')(d5)
    d4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Dropout(0.5)(d4)
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Activation(activation='relu')(d4)
    d3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.5)(d3)
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Activation(activation='relu')(d3)
    d2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Activation(activation='relu')(d2)
    d1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = concatenate([d1, e1], axis=-1)
    
    d0 = Activation(activation='sigmoid')(d1)
    d0 = Conv2DTranspose(n_classes, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d0)
                    
    if nChannels > 1:
        d0 = multiply([inputs2, d0])
        model = Model(input=[inputs, inputs2], output=d0)
    else:
        d0 = multiply([inputs, d0])
        model = Model(input=inputs, output=d0)
    
    return model


def RNN_UNet(n_classes, input_height=256, input_width=512, nChannels=3):
    inputs = Input((input_height, input_width, nChannels))
    if nChannels > 1:
        inputs2 = Input((input_height, input_width, 1))

    
    e1 = Conv2D(64, (3, 3), strides=2, padding='same')(inputs)

    e2 = LeakyReLU(0.2)(e1)
    e2 = Conv2D(128, (3, 3), strides=2, padding='same')(e2)
    e2 = BatchNormalization()(e2)
    
    e3 = LeakyReLU(0.2)(e2)
    e3 = Conv2D(256, (3, 3), strides=2, padding='same')(e3)
    e3 = BatchNormalization()(e3)
    
    e4 = LeakyReLU(0.2)(e3)
    e4 = Conv2D(512, (3, 3), strides=2, padding='same')(e4)
    e4 = BatchNormalization()(e4)
    
    e5 = LeakyReLU(0.2)(e4)
    e5 = Conv2D(512, (3, 3), strides=2, padding='same')(e5)
    e5 = BatchNormalization()(e5)
    
    e6 = LeakyReLU(0.2)(e5)
    e6 = Conv2D(512, (3, 3), strides=2, padding='same')(e6)
    e6 = BatchNormalization()(e6)    

    e6 = Reshape((-1, 512))(e6)
        
    e6 = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False)(e6) 
    e6 = BatchNormalization()(e6)

    e6 = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False)(e6) 
    e6 = BatchNormalization()(e6)

    e6 = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False)(e6) 
    e6 = BatchNormalization()(e6)

    e6 = Reshape((4, 4, 512))(e6)

    
    d5 = Activation(activation='relu')(e6)
    d5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), use_bias=False, 
                         kernel_initializer='he_uniform', padding='same')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Dropout(0.5)(d5)
    d5 = concatenate([d5, e5], axis=-1)
    
    d4 = Activation(activation='relu')(d5)
    d4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Dropout(0.5)(d4)
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Activation(activation='relu')(d4)
    d3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.5)(d3)
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Activation(activation='relu')(d3)
    d2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Activation(activation='relu')(d2)
    d1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = concatenate([d1, e1], axis=-1)
    
    d0 = Activation(activation='sigmoid')(d1)
    d0 = Conv2DTranspose(n_classes, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d0)
                
    if nChannels > 1:
        d0 = multiply([inputs2, d0])
        model = Model(input=[inputs, inputs2], output=d0)
    else:
        d0 = multiply([inputs, d0])
        model = Model(input=inputs, output=d0)
    
    return model


def CR_UNet(n_classes, input_height=256, input_width=512, nChannels=3):
    inputs = Input((input_height, input_width, nChannels))
    if nChannels > 1:
        inputs2 = Input((input_height, input_width, 1))

    
    e1 = Conv2D(64, (3, 3), strides=(2, 1), padding='same')(inputs)

    e2 = LeakyReLU(0.2)(e1)
    e2 = Conv2D(128, (3, 3), strides=(2, 1), padding='same')(e2)
    e2 = BatchNormalization()(e2)
    
    e3 = LeakyReLU(0.2)(e2)
    e3 = Conv2D(256, (3, 3), strides=(2, 1), padding='same')(e3)
    e3 = BatchNormalization()(e3)
    
    e4 = LeakyReLU(0.2)(e3)
    e4 = Conv2D(512, (3, 3), strides=(2, 1), padding='same')(e4)
    e4 = BatchNormalization()(e4)
    
    e5 = LeakyReLU(0.2)(e4)
    e5 = Conv2D(512, (3, 3), strides=(2, 1), padding='same')(e5)
    e5 = BatchNormalization()(e5)
    
    e6 = LeakyReLU(0.2)(e5)
    e6 = Conv2D(512, (3, 3), strides=(2, 1), padding='same')(e6)
    e6 = BatchNormalization()(e6)    

    e6 = Reshape((-1, 512))(e6)
        
    e6 = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False)(e6) 
    e6 = BatchNormalization()(e6)

    e6 = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False)(e6) 
    e6 = BatchNormalization()(e6)

    e6 = Reshape((4, -1, 512))(e6)

    
    d5 = Activation(activation='relu')(e6)
    d5 = Conv2DTranspose(512, (2, 2), strides=(2, 1), use_bias=False, 
                         kernel_initializer='he_uniform', padding='same')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Dropout(0.5)(d5)
    d5 = concatenate([d5, e5], axis=-1)
    
    d4 = Activation(activation='relu')(d5)
    d4 = Conv2DTranspose(512, (2, 2), strides=(2, 1), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Dropout(0.5)(d4)
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Activation(activation='relu')(d4)
    d3 = Conv2DTranspose(256, (2, 2), strides=(2, 1), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.5)(d3)
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Activation(activation='relu')(d3)
    d2 = Conv2DTranspose(128, (2, 2), strides=(2, 1), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Activation(activation='relu')(d2)
    d1 = Conv2DTranspose(64, (2, 2), strides=(2, 1), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = concatenate([d1, e1], axis=-1)
    
    d0 = Activation(activation='sigmoid')(d1)
    d0 = Conv2DTranspose(n_classes, (2, 2), strides=(2, 1), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d0)
                
    if nChannels > 1:
        d0 = multiply([inputs2, d0])
        model = Model(input=[inputs, inputs2], output=d0)
    else:
        d0 = multiply([inputs, d0])
        model = Model(input=inputs, output=d0)
    
    return model


def Pre_UNet(n_classes, input_height=256, input_width=512, nChannels=3,
              soft=True, mul=True, trainable=True):
    inputs = Input((input_height, input_width, nChannels))
    if nChannels > 1:
        inputs2 = Input((input_height, input_width, 1))
        
    dir_name = "UNet_1class_8direction_8ch_mulTrue_cinTrue_ipdTrue_vonMisesFalse_multi_segdata3_256_no_sound_random_sep/"
    weight = "UNet_1class_8direction_8ch_mulTrue_cinTrue_ipdTrue_vonMisesFalse_weights.hdf5"

    pre_cnn = UNet(n_classes=8, input_height=256, input_width=input_width, nChannels=nChannels)
    pre_cnn.load_weights(os.getcwd()+"/model_results/2019_0728/"+dir_name+weight)

    e1 = pre_cnn.layers[1](inputs)
    pre_cnn.layers[1].trainable = trainable # fixed weight
    
    e2 = pre_cnn.layers[2](e1)
    pre_cnn.layers[2].trainable = trainable # fixed weight or fine-tuning    
    e2 = pre_cnn.layers[3](e2)
    pre_cnn.layers[3].trainable = trainable # fixed weight or fine-tuning    
    e2 = pre_cnn.layers[4](e2)
    pre_cnn.layers[4].trainable = trainable # fixed weight or fine-tuning    

    e3 = pre_cnn.layers[5](e2)
    pre_cnn.layers[5].trainable = trainable # fixed weight or fine-tuning    
    e3 = pre_cnn.layers[6](e3)
    pre_cnn.layers[6].trainable = trainable # fixed weight or fine-tuning    
    e3 = pre_cnn.layers[7](e3)
    pre_cnn.layers[7].trainable = trainable # fixed weight or fine-tuning    
    
    e4 = pre_cnn.layers[8](e3)
    pre_cnn.layers[8].trainable = trainable # fixed weight or fine-tuning    
    e4 = pre_cnn.layers[9](e4)
    pre_cnn.layers[9].trainable = trainable # fixed weight or fine-tuning    
    e4 = pre_cnn.layers[10](e4)
    pre_cnn.layers[10].trainable = trainable # fixed weight or fine-tuning    

    e5 = pre_cnn.layers[11](e4)
    pre_cnn.layers[11].trainable = trainable # fixed weight or fine-tuning    
    e5 = pre_cnn.layers[12](e5)
    pre_cnn.layers[12].trainable = trainable # fixed weight or fine-tuning    
    e5 = pre_cnn.layers[13](e5)
    pre_cnn.layers[13].trainable = trainable # fixed weight or fine-tuning    
    
    e6 = pre_cnn.layers[14](e5)
    pre_cnn.layers[14].trainable = trainable # fixed weight or fine-tuning    
    e6 = pre_cnn.layers[15](e6)
    pre_cnn.layers[15].trainable = trainable # fixed weight or fine-tuning    
    e6 = pre_cnn.layers[16](e6)
    pre_cnn.layers[16].trainable = trainable # fixed weight or fine-tuning    
    
    
    d5 = pre_cnn.layers[17](e6)
    pre_cnn.layers[17].trainable = trainable # fixed weight or fine-tuning    
    d5 = pre_cnn.layers[18](d5)
    pre_cnn.layers[18].trainable = trainable # fixed weight or fine-tuning    
    d5 = pre_cnn.layers[19](d5)
    pre_cnn.layers[19].trainable = trainable # fixed weight or fine-tuning    
    d5 = pre_cnn.layers[20](d5)
    pre_cnn.layers[20].trainable = trainable # fixed weight or fine-tuning    
    d5 = pre_cnn.layers[21](d5)
    pre_cnn.layers[21].trainable = trainable # fixed weight or fine-tuning    

    d4 = pre_cnn.layers[22](d5)
    pre_cnn.layers[22].trainable = trainable # fixed weight or fine-tuning    
    d4 = pre_cnn.layers[23](d4)
    pre_cnn.layers[23].trainable = trainable # fixed weight or fine-tuning    
    d4 = pre_cnn.layers[24](d4)
    pre_cnn.layers[24].trainable = trainable # fixed weight or fine-tuning    
    d4 = pre_cnn.layers[25](d4)
    pre_cnn.layers[25].trainable = trainable # fixed weight or fine-tuning    
    d4 = pre_cnn.layers[26](d4)
    pre_cnn.layers[26].trainable = trainable # fixed weight or fine-tuning    

    d3 = pre_cnn.layers[27](d4)
    pre_cnn.layers[27].trainable = trainable # fixed weight or fine-tuning    
    d3 = pre_cnn.layers[28](d3)
    pre_cnn.layers[28].trainable = trainable # fixed weight or fine-tuning    
    d3 = pre_cnn.layers[29](d3)
    pre_cnn.layers[29].trainable = trainable # fixed weight or fine-tuning    
    d3 = pre_cnn.layers[30](d3)
    pre_cnn.layers[30].trainable = trainable # fixed weight or fine-tuning    
    d3 = pre_cnn.layers[31](d3)
    pre_cnn.layers[31].trainable = trainable # fixed weight or fine-tuning    

    d2 = pre_cnn.layers[32](d3)
    pre_cnn.layers[32].trainable = trainable # fixed weight or fine-tuning    
    d2 = pre_cnn.layers[33](d2)
    pre_cnn.layers[33].trainable = trainable # fixed weight or fine-tuning    
    d2 = pre_cnn.layers[34](d2)
    pre_cnn.layers[34].trainable = trainable # fixed weight or fine-tuning    
    d2 = pre_cnn.layers[35](d2)
    pre_cnn.layers[35].trainable = trainable # fixed weight or fine-tuning    

    d1 = pre_cnn.layers[36](d2)
    pre_cnn.layers[36].trainable = trainable # fixed weight or fine-tuning    
    d1 = pre_cnn.layers[37](d1)
    pre_cnn.layers[37].trainable = trainable # fixed weight or fine-tuning    
    d1 = pre_cnn.layers[38](d1)
    pre_cnn.layers[38].trainable = trainable # fixed weight or fine-tuning    
    d1 = pre_cnn.layers[39](d1)
    pre_cnn.layers[39].trainable = trainable # fixed weight or fine-tuning    
    
    """
    d5 = Activation(activation='relu')(e6)
    d5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), use_bias=False, 
                         kernel_initializer='he_uniform', padding='same')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Dropout(0.5)(d5)
    d5 = concatenate([d5, e5], axis=-1)
    
    d4 = Activation(activation='relu')(d5)
    d4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Dropout(0.5)(d4)
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Activation(activation='relu')(d4)
    d3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.5)(d3)
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Activation(activation='relu')(d3)
    d2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Activation(activation='relu')(d2)
    d1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = concatenate([d1, e1], axis=-1)
    """
    
    d0 = Activation(activation='sigmoid')(d1)
    d0 = Conv2DTranspose(n_classes, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d0)
                    
    if nChannels > 1:
        d0 = multiply([inputs2, d0])
        model = Model(input=[inputs, inputs2], output=d0)
    else:
        d0 = multiply([inputs, d0])
        model = Model(input=inputs, output=d0)
    
    return model


def Pre_RNN_UNet(n_classes, input_height=256, input_width=512, nChannels=3,
              soft=True, mul=True, trainable=True):
    inputs = Input((input_height, input_width, nChannels))
    if nChannels > 1:
        inputs2 = Input((input_height, input_width, 1))
        
    dir_name = "RNN_UNet_1class_8direction_8ch_mulTrue_cinTrue_ipdTrue_vonMisesFalse_multi_segdata3_256_no_sound_random_sep/"
    weight = "RNN_UNet_1class_8direction_8ch_mulTrue_cinTrue_ipdTrue_vonMisesFalse_weights.hdf5"

    pre_cnn = RNN_UNet(n_classes=8, input_height=256, input_width=input_width, nChannels=nChannels)
    pre_cnn.load_weights(os.getcwd()+"/model_results/2019_0728/"+dir_name+weight)

    e1 = pre_cnn.layers[1](inputs)
    pre_cnn.layers[1].trainable = trainable # fixed weight
    
    e2 = pre_cnn.layers[2](e1)
    pre_cnn.layers[2].trainable = trainable # fixed weight or fine-tuning    
    e2 = pre_cnn.layers[3](e2)
    pre_cnn.layers[3].trainable = trainable # fixed weight or fine-tuning    
    e2 = pre_cnn.layers[4](e2)
    pre_cnn.layers[4].trainable = trainable # fixed weight or fine-tuning    

    e3 = pre_cnn.layers[5](e2)
    pre_cnn.layers[5].trainable = trainable # fixed weight or fine-tuning    
    e3 = pre_cnn.layers[6](e3)
    pre_cnn.layers[6].trainable = trainable # fixed weight or fine-tuning    
    e3 = pre_cnn.layers[7](e3)
    pre_cnn.layers[7].trainable = trainable # fixed weight or fine-tuning    
    
    e4 = pre_cnn.layers[8](e3)
    pre_cnn.layers[8].trainable = trainable # fixed weight or fine-tuning    
    e4 = pre_cnn.layers[9](e4)
    pre_cnn.layers[9].trainable = trainable # fixed weight or fine-tuning    
    e4 = pre_cnn.layers[10](e4)
    pre_cnn.layers[10].trainable = trainable # fixed weight or fine-tuning    

    e5 = pre_cnn.layers[11](e4)
    pre_cnn.layers[11].trainable = trainable # fixed weight or fine-tuning    
    e5 = pre_cnn.layers[12](e5)
    pre_cnn.layers[12].trainable = trainable # fixed weight or fine-tuning    
    e5 = pre_cnn.layers[13](e5)
    pre_cnn.layers[13].trainable = trainable # fixed weight or fine-tuning    
    
    e6 = pre_cnn.layers[14](e5)
    pre_cnn.layers[14].trainable = trainable # fixed weight or fine-tuning    
    e6 = pre_cnn.layers[15](e6)
    pre_cnn.layers[15].trainable = trainable # fixed weight or fine-tuning    
    e6 = pre_cnn.layers[16](e6)
    pre_cnn.layers[16].trainable = trainable # fixed weight or fine-tuning    
    
    # RNN
    e6 = pre_cnn.layers[17](e6)
    pre_cnn.layers[17].trainable = trainable # fixed weight or fine-tuning    
    
    e6 = pre_cnn.layers[18](e6)
    pre_cnn.layers[18].trainable = trainable # fixed weight or fine-tuning    
    e6 = pre_cnn.layers[19](e6)
    pre_cnn.layers[19].trainable = trainable # fixed weight or fine-tuning    
        
    e6 = pre_cnn.layers[20](e6)
    pre_cnn.layers[20].trainable = trainable # fixed weight or fine-tuning    
    e6 = pre_cnn.layers[21](e6)
    pre_cnn.layers[21].trainable = trainable # fixed weight or fine-tuning    

    e6 = pre_cnn.layers[22](e6)
    pre_cnn.layers[22].trainable = trainable # fixed weight or fine-tuning    
    e6 = pre_cnn.layers[23](e6)
    pre_cnn.layers[23].trainable = trainable # fixed weight or fine-tuning    

    e6 = pre_cnn.layers[24](e6)
    pre_cnn.layers[24].trainable = trainable # fixed weight or fine-tuning    


    d5 = pre_cnn.layers[25](e6)
    pre_cnn.layers[25].trainable = trainable # fixed weight or fine-tuning    
    d5 = pre_cnn.layers[26](d5)
    pre_cnn.layers[26].trainable = trainable # fixed weight or fine-tuning    
    d5 = pre_cnn.layers[27](d5)
    pre_cnn.layers[27].trainable = trainable # fixed weight or fine-tuning    
    d5 = pre_cnn.layers[28](d5)
    pre_cnn.layers[28].trainable = trainable # fixed weight or fine-tuning    
    d5 = pre_cnn.layers[29](d5)
    pre_cnn.layers[29].trainable = trainable # fixed weight or fine-tuning    

    d4 = pre_cnn.layers[30](d5)
    pre_cnn.layers[30].trainable = trainable # fixed weight or fine-tuning    
    d4 = pre_cnn.layers[31](d4)
    pre_cnn.layers[31].trainable = trainable # fixed weight or fine-tuning    
    d4 = pre_cnn.layers[32](d4)
    pre_cnn.layers[32].trainable = trainable # fixed weight or fine-tuning    
    d4 = pre_cnn.layers[33](d4)
    pre_cnn.layers[33].trainable = trainable # fixed weight or fine-tuning    
    d4 = pre_cnn.layers[34](d4)
    pre_cnn.layers[34].trainable = trainable # fixed weight or fine-tuning    

    d3 = pre_cnn.layers[35](d4)
    pre_cnn.layers[35].trainable = trainable # fixed weight or fine-tuning    
    d3 = pre_cnn.layers[36](d3)
    pre_cnn.layers[36].trainable = trainable # fixed weight or fine-tuning    
    d3 = pre_cnn.layers[37](d3)
    pre_cnn.layers[37].trainable = trainable # fixed weight or fine-tuning    
    d3 = pre_cnn.layers[38](d3)
    pre_cnn.layers[38].trainable = trainable # fixed weight or fine-tuning    
    d3 = pre_cnn.layers[39](d3)
    pre_cnn.layers[39].trainable = trainable # fixed weight or fine-tuning    

    d2 = pre_cnn.layers[40](d3)
    pre_cnn.layers[40].trainable = trainable # fixed weight or fine-tuning    
    d2 = pre_cnn.layers[41](d2)
    pre_cnn.layers[41].trainable = trainable # fixed weight or fine-tuning    
    d2 = pre_cnn.layers[42](d2)
    pre_cnn.layers[42].trainable = trainable # fixed weight or fine-tuning    
    d2 = pre_cnn.layers[43](d2)
    pre_cnn.layers[43].trainable = trainable # fixed weight or fine-tuning    

    d1 = pre_cnn.layers[44](d2)
    pre_cnn.layers[44].trainable = trainable # fixed weight or fine-tuning    
    d1 = pre_cnn.layers[45](d1)
    pre_cnn.layers[45].trainable = trainable # fixed weight or fine-tuning    
    d1 = pre_cnn.layers[46](d1)
    pre_cnn.layers[46].trainable = trainable # fixed weight or fine-tuning    
    d1 = pre_cnn.layers[47](d1)
    pre_cnn.layers[47].trainable = trainable # fixed weight or fine-tuning  
    
    """
    d5 = Activation(activation='relu')(e6)
    d5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), use_bias=False, 
                         kernel_initializer='he_uniform', padding='same')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Dropout(0.5)(d5)
    d5 = concatenate([d5, e5], axis=-1)
    
    d4 = Activation(activation='relu')(d5)
    d4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Dropout(0.5)(d4)
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Activation(activation='relu')(d4)
    d3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.5)(d3)
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Activation(activation='relu')(d3)
    d2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Activation(activation='relu')(d2)
    d1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = concatenate([d1, e1], axis=-1)
    """
    
    d0 = Activation(activation='sigmoid')(d1)
    d0 = Conv2DTranspose(n_classes, (2, 2), strides=(2, 2), use_bias=False,
                        kernel_initializer='he_uniform', padding='same')(d0)
                    
    if nChannels > 1:
        d0 = multiply([inputs2, d0])
        model = Model(input=[inputs, inputs2], output=d0)
    else:
        d0 = multiply([inputs, d0])
        model = Model(input=inputs, output=d0)
    
    return model

def GLU_Mask_UNet(n_classes, input_height=256, input_width=512, nChannels=3,
              soft=True, mul=True, trainable=False):
    inputs = Input((input_height, input_width, nChannels))
    if nChannels == 3:
        inputs2 = Input((input_height, input_width, nChannels//3))
    
    #pre_cnn = CNN.CNN4(n_classes=75, input_height=256, input_width=input_width, nChannels=1)
    #pre_cnn.load_weights(os.getcwd()+"/model_results/2018_1118/CNN_75_class/CNN_75_class_weights.hdf5")
#    pre_cnn = CNN.CRNN8(n_classes=9, input_height=256, input_width=input_width, nChannels=1)
#    pre_cnn.load_weights(os.getcwd()+"/model_results/2019_0226/CRNN8_9_class/CRNN8_9_class_weights.hdf5")
    pre_cnn = CNN.CRNN8(n_classes=75, input_height=256, input_width=input_width, nChannels=1)
    pre_cnn.load_weights(os.getcwd()+"/model_results/2019_0122/CRNN8_75_class/CRNN8_75_class_weights.hdf5")
    x = pre_cnn.layers[1](inputs)
    pre_cnn.layers[1].trainable = trainable # fixed weight
    
    for i in range(2, 42): #CRNN8: 42 CNN8:34 CNN4:18
        x = pre_cnn.layers[i](x)
        pre_cnn.layers[i].trainable = trainable # fixed weight or fine-tuning    
    
    x = Flatten()(x)
    x = RepeatVector(256)(x)
    x = Reshape((256, input_width, n_classes))(x)
    
    e1 = concatenate([x, inputs], axis=-1)

    # GLU 
    x = Conv2D(64, (3, 3), strides=2, padding='same')(e1)
    x = BatchNormalization()(x)
    g = Conv2D(64, (3, 3), strides=2, activation='sigmoid', padding='same')(e1)
    g = BatchNormalization()(g)
    e1 = multiply([x, g])

    x = Conv2D(128, (3, 3), strides=2, padding='same')(e1)
    x = BatchNormalization()(x)
    g = Conv2D(128, (3, 3), strides=2, activation='sigmoid', padding='same')(e1)
    g = BatchNormalization()(g)
    e2 = multiply([x, g])
    
    x = Conv2D(256, (3, 3), strides=2, padding='same')(e2)
    x = BatchNormalization()(x)
    g = Conv2D(256, (3, 3), strides=2, activation='sigmoid', padding='same')(e2)
    g = BatchNormalization()(g)
    e3 = multiply([x, g])
    
    x = Conv2D(512, (3, 3), strides=2, padding='same')(e3)
    x = BatchNormalization()(x)
    g = Conv2D(512, (3, 3), strides=2, activation='sigmoid', padding='same')(e3)
    g = BatchNormalization()(g)
    e4 = multiply([x, g])
    
    x = Conv2D(512, (3, 3), strides=2, padding='same')(e4)
    x = BatchNormalization()(x)
    g = Conv2D(512, (3, 3), strides=2, activation='sigmoid', padding='same')(e4)
    g = BatchNormalization()(g)
    e5 = multiply([x, g])
    
    x = Conv2D(512, (3, 3), strides=2, padding='same')(e5)
    x = BatchNormalization()(x)    
    g = Conv2D(512, (3, 3), strides=2, activation='sigmoid', padding='same')(e5)
    g = BatchNormalization()(g)
    e6 = multiply([x, g])
    
    
    d5 = Activation(activation='relu')(e6)
    d5 = Conv2DTranspose(512, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Dropout(0.5)(d5)
    d5 = concatenate([d5, e5], axis=-1)
    
    d4 = Activation(activation='relu')(d5)
    d4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Dropout(0.5)(d4)
    d4 = concatenate([d4, e4], axis=-1)

    d3 = Activation(activation='relu')(d4)
    d3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.5)(d3)
    d3 = concatenate([d3, e3], axis=-1)

    d2 = Activation(activation='relu')(d3)
    d2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d2)
    d2 = BatchNormalization()(d2)
    d2 = concatenate([d2, e2], axis=-1)

    d1 = Activation(activation='relu')(d2)
    d1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d1)
    d1 = BatchNormalization()(d1)
    d1 = concatenate([d1, e1], axis=-1)
    
    d0 = Activation(activation='sigmoid')(d1)
    d0 = Conv2DTranspose(n_classes, (3, 3), strides=(2, 2), use_bias=False, 
                         padding="same", kernel_initializer='he_uniform')(d0)

    if soft == True:
        d0 = core.Activation('softmax')(d0)
        
    if nChannels == 3:
        if mul == True:
            d0 = multiply([inputs2, d0])
            model = Model(input=[inputs, inputs2], output=d0)
    else:
        if mul == True:
            d0 = multiply([inputs, d0])
        model = Model(input=inputs, output=d0)
                        
    return model
