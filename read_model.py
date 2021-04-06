import CNN, Unet, Deeplab
import tensorflow as tf
from keras.utils import multi_gpu_model

def read_model(Model, gpu_count, classes, image_size, channel, ang_reso, sed_model, ang_aux):
    if gpu_count == 1:
        device = '/gpu:0'
    else:
        device = '/cpu:0'
    
    with tf.device(device):            
        if Model == "Mask_UNet":
            model = Unet.UNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel, 
                              mask=True, sed_model=sed_model,
                              RNN=0, freq_pool=False)
            
        elif Model == "UNet":
            model = Unet.UNet(n_classes=classes*ang_reso, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              RNN=0, freq_pool=False)
     
        elif Model == "CR_UNet":
            model = Unet.UNet(n_classes=classes*ang_reso, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              mask=False, RNN=2, freq_pool=True)

        elif Model == "multi_purpose_UNet":
            model = Unet.UNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              RNN=0, freq_pool=False,
                              ssl_enc=False, ssls_out=True, ang_aux=ang_aux)

        elif Model == "Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                      input_shape=(256, image_size, channel), 
                                      classes=classes*ang_reso, OS=16, 
                                      ssl_enc=False, ssls_out=False, ang_aux=ang_aux)
               
        elif Model == "multi_purpose_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                      input_shape=(256, image_size, channel), 
                                      classes=classes, OS=16, 
                                      ssl_enc=False, ssls_out=True, ang_aux=ang_aux)
            
        elif Model == "WUNet":
            model = Unet.WNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              RNN=0, freq_pool=False, ang_reso=8)
        
        elif Model == "UNet_CNN":
            model = Unet.UNet_CNN(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              RNN=0, freq_pool=False, ang_reso=8) 
        elif Model == "Deeplab_CNN":
            model = Unet.Deeplab_CNN(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              RNN=0, freq_pool=False, ang_reso=8) 
            

        elif Model == "CNN8":
            model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=0, Bidir=False)
        elif Model == "CRNN8":
            model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=2, Bidir=False)
        elif Model == "BiCRNN8":
            model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=2, Bidir=True)
            
        elif Model == "SELD_CNN8":
            model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=0, Bidir=False, ang_reso=ang_reso)
        elif Model == "SELD_BiCRNN8":
            model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=2, Bidir=True, ang_reso=ang_reso)
        
        elif Model == "SSL_CNN8":
            model = CNN.CNN(n_classes=ang_reso, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=0, Bidir=False, ang_reso=1)
        elif Model == "SSL_BiCRNN8":
            model = CNN.CNN(n_classes=ang_reso, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=2, Bidir=True, ang_reso=1)

            
        elif Model == "Cascade":
            model = CNN.CNNtag(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512])
             
    if gpu_count > 1:
        multi_model = multi_gpu_model(model, gpus=gpu_count)
    else:
        multi_model = model
        
    return model, multi_model
