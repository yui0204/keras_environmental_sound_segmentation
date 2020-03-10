from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv1D, Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout, Dense
from keras.layers import Activation, RepeatVector, Flatten, Reshape, Permute
from keras.layers import merge, MaxPooling2D, UpSampling2D, core, GRU, LSTM, GlobalAveragePooling2D
from keras.layers.wrappers import Bidirectional

from keras.layers.merge import multiply, dot



def CNN(n_classes, input_height=256, input_width=512, nChannels=3, 
        filter_list=[64, 64, 128, 128, 256, 256, 512, 512], RNN=2, Bidir=False,
        ang_reso=1, ssl_model=None, ssl_mask=False):
    inputs = Input((input_height, input_width, nChannels))
    
    x = inputs
                
    for filters in filter_list:
        if len(filter_list) == 8:
            freq_pool = (2, 1)
        elif len(filter_list) == 4:
            freq_pool = (4, 1)
        
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)    
        x = MaxPooling2D(freq_pool, strides=freq_pool)(x)
                
    if RNN > 0:
        x = Reshape((input_width, 512))(x)
        
        for i in range(RNN):
            if Bidir == False:
                x = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
                        return_sequences=True, stateful=False)(x) 
            else:
                x = Bidirectional(GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', 
                                      return_sequences=True, stateful=False))(x) 
            
            #x = BatchNormalization()(x)
        if ang_reso == 1:
            x = Conv1D(n_classes, 1, activation='sigmoid')(x)
            x = Reshape((1, -1, n_classes))(x)
        else:
            x = Reshape((1, -1, 512))(x)

    else:
        if ang_reso == 1:
            x = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
        
    
    if ang_reso > 1: 
        x = Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 1), padding="same")(x) # 8dir
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        #x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(3, 1), padding="same")(x)
        #x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        #x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(3, 1), padding="same")(x) # 36dir
        #x = BatchNormalization()(x)
        #x = Activation('relu')(x)
        
        x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(3, 1), padding="same")(x)    
        x = BatchNormalization()(x)
        x = Activation('relu')(x)    
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(3, 1), padding="same")(x) # 72dir   
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)

        if ssl_mask == True:
            ssl = ssl_model.layers[1](inputs)
            ssl_model.layers[1].trainable = False # fixed weight
            
            for i in range(2, len(ssl_model.layers)):
                ssl = ssl_model.layers[i](ssl)
                ssl_model.layers[i].trainable = False # fixed weight or fine-tuning
            ssl = Permute((3, 2, 1))(ssl)

            x = multiply([x, ssl])
            #x = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    return model



def CNNtag(n_classes, input_height=256, input_width=512, nChannels=3, 
        filter_list=[64, 64, 128, 128, 256, 256, 512, 512]):
    inputs = Input((input_height, input_width, nChannels))
    
    x = inputs
    
    for filters in filter_list:
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)    
        #x = Conv2D(filters, (3, 3), padding='same')(x)
        #x = BatchNormalization()(x)
        #x = Activation('relu')(x)    
        x = MaxPooling2D((2, 2), strides=(2,2))(x)
        x = Dropout(0.3)(x)
                
    x = GlobalAveragePooling2D()(x)
    
#    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    return model