from keras.layers import merge, MaxPooling2D, UpSampling2D, core, GRU, LSTM, Dense

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import Dropout, RepeatVector, Flatten, Reshape
from keras.layers.merge import concatenate
from keras.layers.merge import multiply, add, average, subtract

from keras.layers.convolutional import ZeroPadding2D, Conv2DTranspose
from keras.layers import LeakyReLU, BatchNormalization, Activation

from keras.applications.vgg16 import VGG16
from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras.regularizers import l2


def GLU_DC(n_classes, input_height=256, input_width=512, nChannels=1):
    inputs = Input((input_height, input_width, nChannels))
    
    x = Conv2D(128, (3, 3), padding='same', dilation_rate=1)(inputs)
    x = BatchNormalization()(x)   
    g = Conv2D(128, (3, 3), activation='sigmoid', padding='same', dilation_rate=1)(inputs)
    g = BatchNormalization()(g)
    o = multiply([x, g])
    

    x = Conv2D(128, (3, 3), padding='same', dilation_rate=2)(o)
    x = BatchNormalization()(x)   
    g = Conv2D(128, (3, 3), activation='sigmoid', padding='same', dilation_rate=2)(o)
    g = BatchNormalization()(g)
    o = multiply([x, g])
    
    x = Conv2D(128, (3, 3), padding='same', dilation_rate=3)(o)
    x = BatchNormalization()(x)   
    g = Conv2D(128, (3, 3), activation='sigmoid', padding='same', dilation_rate=3)(o)
    g = BatchNormalization()(g)
    o = multiply([x, g])

    x = Conv2D(128, (3, 3), padding='same', dilation_rate=4)(o)
    x = BatchNormalization()(x)   
    g = Conv2D(128, (3, 3), activation='sigmoid', padding='same', dilation_rate=4)(o)
    g = BatchNormalization()(g)
    o = multiply([x, g])


    x = Conv2D(n_classes, (3, 3), padding='same', dilation_rate=5)(o)
    x = BatchNormalization()(x)   
    g = Conv2D(n_classes, (3, 3), activation='sigmoid', padding='same', dilation_rate=5)(o)
    g = BatchNormalization()(g)
    o = multiply([x, g])

    x = multiply([inputs, o])

    model = Model(inputs=inputs, outputs=x)
    
    return model    
    
    """
    x = Dense(20, activation=None, use_bias=True)(x)
    
    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)
    
    x = Bidirectional(GRU(256, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)
    
    #x = Conv1D(n_classes, 1, activation='sigmoid')(x)
    x = Reshape((256, -1, n_classes))(x) 
    
    x = multiply([inputs, x])

    model = Model(inputs=inputs, outputs=x)
    
    return model
    """

def LSTM_DC(n_classes, input_height=256, input_width=512, nChannels=3):
    inputs = Input((input_height, input_width, nChannels))
        
    x = Reshape((input_width, input_height))(inputs)
        
    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)
    
    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)


    x = TimeDistributed(Dense(75, activation='tanh',
                              W_regularizer=l2(1e-6),
                              b_regularizer=l2(1e-6)),
                              name='kmeans_o')(x)
        
    """
    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(300, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)
    
    x = Bidirectional(GRU(256, activation='tanh', recurrent_activation='hard_sigmoid', 
            return_sequences=True,
            dropout=0.25, recurrent_dropout=0.25, stateful=False))(x) 
    x = BatchNormalization()(x)
    
    #x = Conv1D(n_classes, 1, activation='sigmoid')(x)
    x = Reshape((256, -1, n_classes))(x) 
    
    x = multiply([inputs, x])
    """
    model = Model(inputs=inputs, outputs=x)
    
    return model
