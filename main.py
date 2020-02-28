#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 23:29:14 2018

@author: yui-sudo
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pydot, graphviz
import pickle
from keras.optimizers import SGD, Adam, Adagrad
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model

import CNN, Unet, PSPNet, Deeplab

from sound import WavfileOperate, Stft
import cmath

import shutil
from mir_eval.separation import bss_eval_sources


import tensorflow as tf
from keras.utils import multi_gpu_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
    config.gpu_options.visible_device_list = "0"
elif os.getcwd() == '/home/sudou/python/sound_segtest':
    config.gpu_options.visible_device_list = "0"
else:
    config.gpu_options.visible_device_list = "0,1,2"
sess = tf.Session(config=config)
K.set_session(sess)

import time
import soundfile as sf
from scipy import signal

from sklearn.metrics import f1_score

import re
import gc

import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate


def create_message(body):
    msg = MIMEText(body)
    msg['Subject'] = "Training status"
    msg['From'] = "ysudo0204@gmail.com"
    msg['To'] = "yui.sudo@gmail.com"
    msg['Date'] = formatdate()
    return msg


def send_mail(body_msg):
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.ehlo()
    smtpobj.starttls()
    smtpobj.ehlo()
    smtpobj.login("ysudo0204@gmail.com", "mrchildren0510")
    smtpobj.sendmail("ysudo0204@gmail.com", "yui.sudo@gmail.com", body_msg.as_string())
    smtpobj.close()



def normalize(inputs, labels):
    max = inputs.max()
    inputs = inputs / max
    labels = labels / max
    
    inputs = np.clip(inputs, 0.0, 1.0)
    labels = np.clip(labels, 0.0, 1.0)
    
    return inputs, labels, max


def log(inputs, labels):
    inputs += 10**-7
    labels += 10**-7
    inputs = 20 * np.log10(inputs)
    labels = 20 * np.log10(labels)
    
    return inputs, labels


def load(segdata_dir, n_classes=8, load_number=9999999, complex_input=False):   
    print("data loading\n")
    if mic_num == 1:
        if complex_input == True or vonMises == True:
            input_dim = 3
        else:
            input_dim = 1
    else:
        if ipd == True:
            input_dim = 15   
        elif complex_input == True or vonMises == True:
            input_dim = 24
        else:
            input_dim = 8        
        
    if train_mode == "category":
        cat = 2
    else:
        cat = 0
        
    inputs = np.zeros((load_number, input_dim, 256, image_size), dtype=np.float16)
    if task == "event":
        inputs_phase = 1
    else:    
        inputs_phase = np.zeros((load_number, 512, image_size), dtype=np.float16)
        
    if n_classes > 1 and ang_reso == 1:
        if task == "event":
            labels = np.zeros((load_number, n_classes, image_size), dtype=np.float16)
        elif task == "segmentation":
            labels = np.zeros((load_number, n_classes, 256, image_size), dtype=np.float16)
    elif ang_reso > 1 and n_classes == 1:
        labels = np.zeros((load_number, ang_reso, 256, image_size), dtype=np.float16)
    else:
        if task == "event":
            labels = np.zeros((load_number, n_classes, ang_reso, image_size), dtype=np.float16)
        elif task == "cube":
            labels = np.zeros((load_number, n_classes, ang_reso, 256, image_size), dtype=np.float16)            
        
    
    for i in range(load_number):
        data_dir = segdata_dir + str(i) + "/"
        filelist = os.listdir(data_dir)  
        
        with open(segdata_dir + str(i) + "/sound_direction.txt", "r") as f:
            direction = f.read().split("\n")[:-1]
            
        dn = 0
        for n in range(len(filelist)):
            if filelist[n][-4:] == ".wav":
                waveform, fs = sf.read(data_dir + filelist[n]) 
                if filelist[n][0:3] == "0__":
                    if mic_num == 1:
                        freqs, t, stft = signal.stft(x=waveform, fs=fs, nperseg=512, 
                                                               return_onesided=False)
                        stft = stft[:, 1:len(stft.T) - 1]
                        if not task == "event":
                            inputs_phase[i] = np.angle(stft)
                        if vonMises == True:
                            inputs[i][1] = np.cos(np.angle(stft[:256]))
                            inputs[i][2] = np.sin(np.angle(stft[:256]))
                        elif complex_input == True:
                            inputs[i][1] = stft[:256].real
                            inputs[i][2] = stft[:256].imag                        
                        inputs[i][0] = abs(stft[:256])
                
                elif filelist[n][:7] == "0_multi":
                    if mic_num == 8:
                        freqs, t, stft = signal.stft(x=waveform.T, fs=fs, nperseg=512, 
                                                               return_onesided=False)
                        stft = stft[:, :, 1:len(stft.T) - 1]
                        if not task == "event":
                            inputs_phase[i] = np.angle(stft[0])
                        for nchan in range(mic_num):
                            if ipd == True:
                                if nchan == 0:
                                    inputs[i][nchan] = abs(stft[nchan][:256])
                                else:
                                    inputs[i][nchan*2-1] = np.cos(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                                    inputs[i][nchan*2] = np.sin(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                                    
                            elif vonMises == True:
                                inputs[i][nchan] = abs(stft[nchan][:256])
                                inputs[i][nchan*3+1] = np.cos(np.angle(stft[nchan][:256]))
                                inputs[i][nchan*3+2] = np.sin(np.angle(stft[nchan][:256]))                          
                                    
                            elif complex_input == True:
                                inputs[i][nchan * 3] = abs(stft[nchan][:256])
                                inputs[i][nchan*3 + 1] = stft[nchan][:256].real
                                inputs[i][nchan*3 + 2] = stft[nchan][:256].imag
                            else:
                                inputs[i][nchan] = abs(stft[nchan][:256])
                                
                else:
                    if filelist[n][:-4] == "BGM":
                        continue
                    freqs, t, stft = signal.stft(x=waveform, fs=fs, nperseg = 512, 
                                                 return_onesided=False)
                    stft = stft[:, 1:len(stft.T) - 1]

                    if n_classes > 1 and ang_reso == 1:
                        if task == "event": # SED
                            labels[i][label.T[filelist[n][:-4]][cat]] += abs(stft[:256]).max(0)
                        elif task == "segmentation": # Segmentation
                            labels[i][label.T[filelist[n][:-4]][cat]] += abs(stft[:256])                        
                    elif ang_reso > 1:
                        angle = int(re.sub("\\D", "", direction[dn].split("_")[1])) // (360 // ang_reso)
                        if n_classes == 1: # SSLS
                            labels[i][angle] += abs(stft[:256])          
                        elif task == "event": # SELD
                            labels[i][label.T[filelist[n][:-4]][cat]][angle] += abs(stft[:256]).max(0)
                        elif task == "cube": #CUBE
                            labels[i][label.T[filelist[n][:-4]][cat]][angle] += abs(stft[:256])
                        dn += 1

    
    if complex_input == True and ipd == False and vonMises == False:
        sign = (inputs > 0) * 2 - 1
        sign = sign.astype(np.float16)
        inputs = abs(inputs) ############################# bug fix
            
        
    if ipd == True or vonMises == True:
        inputs = inputs.transpose(1,0,2,3)
        inputs[0], labels = log(inputs[0], labels)   
        inputs[0] = np.nan_to_num(inputs[0])
        if vonMises == True and mic_num == 8:
            for ch in range(1, mic_num):
                inputs[ch * 3], labels = log(inputs[ch * 3], labels)
                inputs[ch * 3] = np.nan_to_num(inputs[ch * 3])
                inputs[ch * 3] += 120
                inputs[ch * 3], a, a = normalize(inputs[ch * 3], labels)
        labels = np.nan_to_num(labels) 
        inputs[0] += 120
        labels += 120
        inputs[0], labels, max = normalize(inputs[0], labels)
        inputs = inputs.transpose(1,0,2,3)

    else:
        inputs, labels = log(inputs, labels)   
        inputs = np.nan_to_num(inputs)
        labels = np.nan_to_num(labels) 
        inputs += 120
        labels += 120
        inputs, labels, max = normalize(inputs, labels)


    if complex_input == True and ipd == False and vonMises == False:
        inputs = inputs * sign
    
    if task == "event":
        labels = ((labels > 0.1) * 1)

    inputs = inputs.transpose(0, 2, 3, 1)
    if n_classes > 1 and ang_reso == 1:
        if task == "event":
            labels = labels.transpose(0, 2, 1)  
        elif task == "segmentation":
            labels = labels.transpose(0, 2, 3, 1)  
    elif ang_reso > 1 and n_classes == 1:
        labels = labels.transpose(0, 2, 3, 1)  
    else:
        if task == "event":
            labels = labels.transpose(0, 2, 3, 1)  
        elif task == "cube":
            labels = labels.transpose(0, 3, 4, 1, 2)
            labels = labels.reshape((load_number, 256, image_size, n_classes * ang_reso))
        
        
    
    return inputs, labels, max, inputs_phase


def read_model(Model):
    if gpu_count == 1:
        device = '/gpu:0'
    else:
        device = '/cpu:0'
    
    with tf.device(device):            
        if Model == "aux_Mask_UNet":
            model = Unet.UNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel, 
                              trainable=trainable, 
                              sed_model=sed_model, num_layer=num_layer, aux=aux,
                              mask=True, RNN=0, freq_pool=False)
        elif Model == "aux_enc_UNet":
            model = Unet.UNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=aux,
                              mask=False, RNN=0, freq_pool=False, enc=True)   
        elif Model == "aux_Mask_RNN_UNet":
            model = Unet.UNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel, 
                              trainable=trainable, 
                              sed_model=sed_model, num_layer=num_layer, aux=aux,
                              mask=True, RNN=2, freq_pool=False) 
        elif Model == "Mask_UNet":
            model = Unet.UNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel, 
                              trainable=trainable, 
                              sed_model=sed_model, num_layer=num_layer, aux=aux,
                              mask=True, RNN=0, freq_pool=False) 
        elif Model == "UNet":
            model = Unet.UNet(n_classes=classes*ang_reso, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=0, freq_pool=False)
        elif Model == "doa_UNet":
            model = Unet.UNet(n_classes=classes*ang_reso, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=0, freq_pool=False, doa=True, sad=False)
        elif Model == "sad_UNet":
            model = Unet.UNet(n_classes=classes*ang_reso, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=0, freq_pool=False, doa=True, sad=True)
        elif Model == "RNN_UNet":
            model = Unet.UNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=2, freq_pool=False)                  
        elif Model == "CR_UNet":
            model = Unet.UNet(n_classes=classes*ang_reso, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=2, freq_pool=True)   
               
        elif Model == "Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                      input_shape=(256, image_size, channel), 
                                      classes=classes * ang_reso, OS=16, 
                                      RNN=0,
                                      mask=False, trainable=False, sed_model=None, 
                                      num_layer=None, aux=False)
        elif Model == "SSL_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                      input_shape=(256, image_size, channel), 
                                      classes=classes * ang_reso, OS=16, 
                                      RNN=0,
                                      mask=False, trainable=False, sed_model=None, 
                                      num_layer=None, aux=False, ssl=True)       
            
        elif Model == "RNN_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                      input_shape=(256, image_size, channel), 
                                      classes=classes * ang_reso, OS=16, 
                                      RNN=2,
                                      mask=False, trainable=False, sed_model=None, 
                                      num_layer=None, aux=False)
            
        elif Model == "Mask_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                      input_shape=(256, image_size, channel), 
                                      classes=classes * ang_reso, OS=16, 
                                      RNN=0,
                                      mask=True, trainable=trainable, sed_model=sed_model, 
                                      num_layer=num_layer, aux=False)    
        elif Model == "aux_Mask_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                      input_shape=(256, image_size, channel), 
                                      classes=classes * ang_reso, OS=16, 
                                      RNN=0,
                                      mask=True, trainable=trainable, sed_model=sed_model, 
                                      num_layer=num_layer, aux=aux)
        elif Model == "aux_enc_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                      input_shape=(256, image_size, channel), 
                                      classes=classes * ang_reso, OS=16, 
                                      RNN=0,
                                      mask=False, trainable=False, sed_model=None, 
                                      num_layer=None, aux=aux, enc=True)
            
        elif Model == "WUNet":
            model = Unet.WNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=0, freq_pool=False, ang_reso=8)   
        
        elif Model == "WDeeplab":
            model = Unet.UNet_Deeplab(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=0, freq_pool=False, ang_reso=8)
        elif Model == "UNet_CNN":
            model = Unet.UNet_CNN(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=0, freq_pool=False, ang_reso=72) 
        elif Model == "Deeplab_CNN":
            model = Unet.Deeplab_CNN(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=0, freq_pool=False, ang_reso=72) 

        elif Model == "UNet_CNN_Deeplab":
            model = Unet.UNet_CNN(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=channel,
                              trainable=False, 
                              sed_model=None, num_layer=None, aux=False,
                              mask=False, RNN=0, freq_pool=False, ang_reso=8, seg=True) 
            
        elif Model == "CNN4":
            model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 128, 256, 512], 
                            RNN=0, Bidir=False)
        elif Model == "CNN8":
            model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=0, Bidir=False)
        elif Model == "CRNN":
            model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 128, 256, 512], 
                            RNN=2, Bidir=False)
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
            
        elif Model == "Cascade":
            model = CNN.CNNtag(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512])
             
    if gpu_count > 1:
        multi_model = multi_gpu_model(model, gpus=gpu_count)
    else:
        multi_model = model
        
    return model, multi_model



def train(X_train, Y_train, Model):
    model, multi_model = read_model(Model)
    
    if aux == False:
        multi_model.compile(loss=loss, optimizer=Adam(lr=lr),metrics=["accuracy"])     

    elif aux == True or Model == "SSL_Deeplab":
        multi_model.compile(loss=["binary_crossentropy", "mean_squared_error"],
                            loss_weights=[0.01, 1.0],
                            optimizer=Adam(lr=lr),metrics=["accuracy"])
        

    plot_model(model, to_file = results_dir + model_name + '.png')
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1,mode="auto")
#    checkpoint = ModelCheckpoint(filepath=results_dir+"/checkpoint/"+model_name+"_{epoch}.hdf5", save_best_only=False, period=100)
#    tensorboard = TensorBoard(log_dir=results_dir, histogram_freq=0, write_graph=True)

    if task == "segmentation":
        if complex_input == True  or mic_num > 1 or ipd == True:
            X_train = [X_train, 
                       X_train.transpose(3,0,1,2)[0][np.newaxis,:,:,:].transpose(1,2,3,0)]
        
        if Model == "aux_Mask_UNet" or Model == "aux_Mask_RNN_UNet" or Model == "aux_Mask_Deeplab" or Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
            Y_train = [((Y_train.transpose(3,0,1,2).max(2)[:,:,np.newaxis,:] > 0.0) * 1).transpose(1,2,3,0), 
                       Y_train]
        elif Model == "SSL_Deeplab":
            Y_train = [((Y_train.max(1).max(1) > 0.1) * 1), Y_train]
    
    history = multi_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
                            epochs=NUM_EPOCH, verbose=1, validation_split=0.1,
                            callbacks=[early_stopping])

    with open(results_dir + "history.pickle", mode="wb") as f:
        pickle.dump(history.history, f)

    model_json = model.to_json()
    with open(results_dir + "model.json", mode="w") as f:
        f.write(model_json)
    
    model.save_weights(results_dir + model_name + '_weights.hdf5')
    
    return history



def plot_history(history, model_name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(0, 100)
#    plt.ylim(0.0, 0.03)
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(results_dir + "loss_"+str(np.array((history.history["val_loss"])).min())+".png")
    plt.close()


def predict(X_test, Model):
    model, multi_model = read_model(Model)
    model.load_weights(results_dir + model_name + '_weights.hdf5')

    print("\npredicting...")
    if task == "segmentation":
        if complex_input == True or mic_num > 1:
            X_test = [X_test, 
                      X_test.transpose(3,0,1,2)[0][np.newaxis,:,:,:].transpose(1,2,3,0)]
    Y_pred = model.predict(X_test, BATCH_SIZE * gpu_count)
    print("prediction finished\n")
    
    return Y_pred


def pred_dir_make(no):
    if not os.path.exists(results_dir + "prediction/" + str(no)):
        os.mkdir(results_dir + "prediction/" + str(no))
    pred_dir = results_dir + "prediction/" + str(no) + "/"
    
    return pred_dir


def get_filename(no=0):
    data_dir = valdata_dir + str(no)
    filelist = os.listdir(data_dir)
    for n in range(len(filelist)):
        if filelist[n][0] == "0":
            filename = filelist[n]
    
    return filename


def origin_stft(X, no=0):
    pred_dir = pred_dir_make(no)

    X = X.transpose(0, 3, 1, 2)    
    filename = get_filename(no)
    plt.title(str(no) + "__" + filename)
    plt.pcolormesh((X[no][0]))
    plt.xlabel("time")
    plt.ylabel('frequency')
    plt.clim(0, 1)
    plt.colorbar()
    plt.savefig(pred_dir + filename[:-4] + ".png")
    plt.close()


def event_plot(Y_true, Y_pred, no=0):
    pred_dir = pred_dir_make(no)

    if ang_reso == 1:
        plt.pcolormesh((Y_true[no][0].T))
        plt.title("truth")
        plt.xlabel("time")
        plt.ylabel('class index')
        plt.clim(0, 1)
        plt.colorbar()
        plt.savefig(pred_dir + "true.png")
        plt.close()
        
        plt.pcolormesh((Y_pred[no][0].T))
        plt.title("prediction")
        plt.xlabel("time")
        plt.ylabel('class index')
        plt.clim(0, 1)
        plt.colorbar()
        plt.savefig(pred_dir + "pred.png")    
        plt.close()
    
    else:
        Y_pred = Y_pred.transpose(0, 3, 1, 2)
        Y_true = Y_true.transpose(0, 3, 1, 2)
            
        Y_true_total = np.zeros((ang_reso, image_size))
        Y_pred_total = np.zeros((ang_reso, image_size))
        for i in range(classes):
            if Y_true[no][i].max() > 0: #含まれているクラスのみグラフ表示
                
                plt.pcolormesh((Y_true[no][i]))
                plt.title(label.index[i] + "_truth")
                plt.xlabel("time")
                plt.ylabel('frequency')
                plt.clim(0, 1)
                plt.colorbar()
                plt.savefig(pred_dir + label.index[i] + "_true.png")
                plt.close()
    
                plt.pcolormesh((Y_pred[no][i]))
                plt.title(label.index[i] + "_prediction")
                plt.xlabel("time")
                plt.ylabel('frequency')
                plt.clim(0, 1)
                plt.colorbar()
                plt.savefig(pred_dir + label.index[i] + "_pred.png")
                plt.close()
                
            Y_true_total += (Y_true[no][i] > 0.45) * (i + 4)
            Y_pred_total += (Y_pred[no][i] > 0.45) * (i + 4)
        
        filename = get_filename(no)
        
        plt.pcolormesh((Y_true_total), cmap="gist_ncar")
        plt.title(str(no) + "__" + filename + "_truth")
        plt.xlabel("time")
        plt.ylabel('angle')
        plt.clim(0, Y_true_total.max())
        plt.savefig(pred_dir + filename + "_truht.png")
        plt.close()
    
        plt.pcolormesh((Y_pred_total), cmap="gist_ncar")
        plt.title(str(no) + "__" + filename + "_prediction")
        plt.xlabel("time")
        plt.ylabel('angle')
        plt.clim(0, Y_true_total.max())
        plt.savefig(pred_dir + filename + "_pred.png")
        plt.close()


def plot_stft(Y_true, Y_pred, no=0):
    plot_num = classes * ang_reso
    if ang_reso > 1:
        ylabel = "angle"
    else:
        ylabel = "frequency"
    pred_dir = pred_dir_make(no)

    Y_pred = Y_pred.transpose(0, 3, 1, 2)
    Y_true = Y_true.transpose(0, 3, 1, 2)
        
    Y_true_total = np.zeros((256, image_size))
    Y_pred_total = np.zeros((256, image_size))
    for i in range(plot_num):
        if Y_true[no][i].max() > 0: #含まれているクラスのみグラフ表示
            plt.pcolormesh((Y_true[no][i]))
            if ang_reso == 1:
                plt.title(label.index[i] + "_truth")
            else:
                plt.title(label.index[i // ang_reso] + "_" + str((360 // ang_reso) * (i % ang_reso)) + "deg_truth")
            plt.xlabel("time")
            plt.ylabel(ylabel)
            plt.clim(0, 1)
            plt.colorbar()
            if ang_reso == 1:
                plt.savefig(pred_dir + label.index[i] + "_true.png")
            else:
                plt.savefig(pred_dir + label.index[i // ang_reso] + "_" + str((360 // ang_reso) * (i % ang_reso)) + "deg_true.png")
            plt.close()

            plt.pcolormesh((Y_pred[no][i]))
            if ang_reso == 1:
                plt.title(label.index[i] + "_prediction")
            else:
                plt.title(label.index[i // ang_reso] + "_" + str((360 // ang_reso) * (i % ang_reso)) + "deg_prediction")                
            plt.xlabel("time")
            plt.ylabel(ylabel)
            plt.clim(0, 1)
            plt.colorbar()
            if ang_reso == 1:
                plt.savefig(pred_dir + label.index[i] + "_pred.png")
            else:
                plt.savefig(pred_dir + label.index[i // ang_reso] + "_" + str((360 // ang_reso) * (i % ang_reso)) + "deg_pred.png")         
            plt.close()
            
        Y_true_total += (Y_true[no][i] > 0.45) * (i + 4)
        Y_pred_total += (Y_pred[no][i] > 0.45) * (i + 4)
    
    filename = get_filename(no)
    
    plt.pcolormesh((Y_true_total), cmap="gist_ncar")
    plt.title(str(no) + "__" + filename + "_truth")
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.clim(0, Y_true_total.max())
    plt.savefig(pred_dir + filename + "_truth.png")
    plt.close()

    plt.pcolormesh((Y_pred_total), cmap="gist_ncar")
    plt.title(str(no) + "__" + filename + "_prediction")
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.clim(0, Y_true_total.max())
    plt.savefig(pred_dir + filename + "_pred.png")
    plt.close()


def restore(Y_true, Y_pred, max, phase, no=0, class_n=1, save=False):
    plot_num = classes * ang_reso

    if save == True:
        pred_dir = pred_dir_make(no)
    
    Y_pred = Y_pred.transpose(0, 3, 1, 2)
    Y_true = Y_true.transpose(0, 3, 1, 2)    
    
    data_dir = valdata_dir + str(no)
    sdr_array = np.zeros((plot_num, 1))
    sir_array = np.zeros((plot_num, 1))
    sar_array = np.zeros((plot_num, 1))
    
    for class_n in range(plot_num):
        if Y_true[no][class_n].max() > 0:
            Y_linear = 10 ** ((Y_pred[no][class_n] * max - 120) / 20)
            Y_linear = np.vstack((Y_linear, Y_linear[::-1]))

            Y_complex = np.zeros((512, image_size), dtype=np.complex128)
            for i in range (512):
                for j in range (image_size):
                    Y_complex[i][j] = cmath.rect(Y_linear[i][j], phase[no][i][j])

            if ang_reso == 1:
                Y_Stft = Stft(Y_complex, 16000, label.index[class_n]+"_prediction")
            else:
                Y_Stft = Stft(Y_complex, 16000, label.index[i % ang_reso * 0] + "_" + str((360 // ang_reso) * (class_n % ang_reso)) + "deg_prediction")
                
            Y_pred_wave = Y_Stft.scipy_istft()
            
            if save == True:
                Y_pred_wave.write_wav_sf(dir=pred_dir, filename=None, bit=16)

            if task == "segmentation" and ang_reso == 1:
                Y_pred_wave = Y_pred_wave.norm_sound
                Y_true_wave = WavfileOperate(data_dir + "/" + label.index[class_n] + ".wav").wavedata.norm_sound            
                Y_true_wave = Y_true_wave[:len(Y_pred_wave)]    
                sdr, sir, sar, per = bss_eval_sources(Y_true_wave[np.newaxis,:], Y_pred_wave[np.newaxis,:], compute_permutation=True)
                #print("No.", no, class_n, label.index[class_n], round(sdr[0], 2))
                
                sdr_array[class_n] = sdr
                sir_array[class_n] = sir
                sar_array[class_n] = sar
            
    return sdr_array, sir_array, sar_array
    

def RMS(Y_true, Y_pred):
    Y_pred = Y_pred.transpose(0, 3, 1, 2) # (data number, class, freq, time)
    Y_true = Y_true.transpose(0, 3, 1, 2)
        
    Y_pred_db = (Y_pred * max) # 0~120dB
    Y_true_db = (Y_true * max)
    
    total_rmse = np.sqrt(((Y_true_db - Y_pred_db) ** 2).mean())
    print("Total RMSE =", total_rmse)

    rms_array = np.zeros(classes + 1)
    num_array = np.zeros(classes + 1) # the number of data of each class

    area_array = np.zeros(classes + 1) # area size of each class
    
    spl_array =  np.zeros(classes + 1) #average SPL of each class 
    percent_array =  np.zeros(classes + 1) 

    for no in range(load_number):
        for class_n in range(classes):
            if Y_true[no][class_n].max() > 0: # calculate RMS of active event class               
                num_array[classes] += 1 # total number of all classes
                num_array[class_n] += 1 # the number of data of each class
                on_detect = Y_true[no][class_n].max(0) > 0.0 # active section of this spectrogram image
                
                per_rms = ((Y_true_db[no][class_n] - Y_pred_db[no][class_n]) ** 2).mean(0) # mean squared error about freq axis of this spectrogram
                rms_array[class_n] += per_rms.sum() / on_detect.sum() # mean squared error of one data
                
                per_spl = Y_true_db[no][class_n].mean(0) # mean spl about freq axis
                spl_array[class_n] += per_spl.sum() / on_detect.sum() # mean spl of one data
                
                area_array[class_n] += ((Y_true[no][class_n] > 0.0) * 1).sum() # number of active bins = area size
                
    rms_array[classes] = rms_array.sum()
    rms_array = np.sqrt(rms_array / num_array) # Squared error is divided by the number of data = MSE then root = RMSE

    spl_array[classes] = spl_array.sum()
    spl_array = spl_array / num_array # Sum of each spl is divided by the number of data = average spl

    area_array[classes] = area_array.sum()
    area_array = area_array // num_array # average area size of each class
    
    percent_array = rms_array / spl_array * 100 


    #print(area_array)
    print("rms\n", rms_array, "\n")     
    print("num\n", num_array, "\n")
    print("percent\n", percent_array, "\n")

    np.savetxt(results_dir+"prediction/datanum_"+str(load_number)+".csv", num_array)
    np.savetxt(results_dir+"prediction/rmse_"+str(load_number)+".csv", rms_array, fmt ='%.3f')
    np.savetxt(results_dir+"prediction/area_"+str(load_number)+".csv", area_array)
    np.savetxt(results_dir+"prediction/spl_"+str(load_number)+".csv", spl_array, fmt ='%.3f')
    np.savetxt(results_dir+"prediction/percent_"+str(load_number)+".csv", percent_array, fmt ='%.3f')
    

    plt.plot(rms_array, marker='o', linestyle="None")
    plt.title("rms")
    plt.xlabel("")
    plt.ylabel('rms')
    plt.savefig(results_dir + "rms_result.png")
    plt.ylim(0, 50)
    plt.close()

    plt.plot(area_array, rms_array, marker='o', linestyle="None")
    plt.title("area-rms")
    plt.xlabel("area")
    plt.ylabel('rms')
    plt.savefig(results_dir + "area-rms_result.png")
    plt.ylim(0, 50)
    plt.close()
    
    plt.plot(spl_array, rms_array, marker='o', linestyle="None")
    plt.title("spl-rms")
    plt.xlabel("spl")
    plt.ylabel('rms')
    plt.savefig(results_dir + "spl-rms_result.png")
    plt.ylim(0, 50)
    plt.close()



def save_npy(X, Y, max, phase, name):    
    np.save(dataset+"X_"+name+".npy", X)
    np.save(dataset+"max_"+name+".npy", max)
    np.save(dataset+"phase_"+name+".npy", phase)
    np.save(dataset+"Y_"+name+".npy", Y)
            
    print("npy files were saved\n")
        
        
def load_npy(name):
    X = np.load(dataset+"X_"+name+".npy")
    max = np.load(dataset+"max_"+name+".npy")
    phase = np.load(dataset+"phase_"+name+".npy")
    Y = np.load(dataset+"Y_"+name+".npy")

    print("npy files were loaded\n")
    
    return X, Y, max, phase


def load_sed_model(Model):
    if Model == "CNN8":
        sed_model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=0, Bidir=False)
        sed_model.load_weights(os.getcwd()+"/model_results/advanced_robotics/CNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/CNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")
    elif Model == "CRNN8":
        sed_model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=2, Bidir=False)
        sed_model.load_weights(os.getcwd()+"/model_results/advanced_robotics/CRNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/CRNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")
    elif Model == "BiCRNN8":
        sed_model = CNN.CNN(n_classes=classes, input_height=256, 
                            input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], 
                            RNN=2, Bidir=True)
        sed_model.load_weights(os.getcwd()+"/model_results/advanced_robotics/BiCRNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/BiCRNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")
    
    num_layer = len(sed_model.layers)

    return sed_model, num_layer


def load_cascade(segdata_dir, load_number=9999999):
    print("data loading\n")
                
    inputs = np.zeros((1, 256, image_size), dtype=np.float16)
    sep_num = np.zeros((load_number), dtype=np.int16)
    
    for i in range(load_number):
        data_dir = segdata_dir + str(i) + "/"
        filelist = os.listdir(data_dir)  
                    
        for n in range(len(filelist)):
            if filelist[n][:4] == "sep_":
                waveform, fs = sf.read(data_dir + filelist[n]) 
                freqs, t, stft = signal.stft(x=waveform, fs=fs, nperseg=512, 
                                                       return_onesided=False)
                stft = stft[:, 1:len(stft.T) - 1]
                
                inputs = np.concatenate((inputs, abs(stft[:256])[np.newaxis, :, :]), axis=0)
                sep_num[i] += 1
                
    inputs += 10**-7
    inputs = 20 * np.log10(inputs)
    inputs = np.nan_to_num(inputs)
    inputs += 120
    max = inputs.max()
    inputs = inputs / max
    inputs = np.clip(inputs, 0.0, 1.0)
    
    return inputs[1:][:, :, :, np.newaxis], sep_num


def Segtoclsdata(Y_in):
    Y_in = Y_in.transpose(0, 3, 1, 2) # (data number, class, freq, time)
    X_cls = np.zeros((load_number * 3, 256, image_size))
    Y_cls = np.zeros((load_number * 3, classes))
    
    data_num = 0
    for no in range(load_number):
        for class_n in range(classes):
            if Y_in[no][class_n].max() > 0:
                X_cls[data_num] = Y_in[no][class_n]
                Y_cls[data_num][class_n] = 1
                data_num += 1
    print(data_num)

    return X_cls[:data_num][:, :, :, np.newaxis], Y_cls[:data_num]




if __name__ == '__main__':
    train_mode = "class"
    classes = 75
    image_size = 256
    task = "segmentation"
    ang_reso = 1
    
    if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
        gpu_count = 1
    elif os.getcwd() == '/home/sudou/python/sound_segtest':
        gpu_count = 1
    else:
        gpu_count = 3
    BATCH_SIZE = 16 * gpu_count
    NUM_EPOCH = 100
    
    lr = 0.001
    
    loss = "mean_squared_error"
    if task == "event":
        loss = "binary_crossentropy"

    mode = "train"
    date = mode       
    plot = True
    graph_num = 10


    if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
        datasets_dir = "/home/yui-sudo/document/dataset/sound_segmentation/datasets/"
    elif os.getcwd() == '/home/sudou/python/sound_segtest':
        datasets_dir = '/media/sudou/d0e7ca7c-34a8-4983-945f-a0783e5a55c5/dataset/dataset/datasets/'
    else:
        datasets_dir = "/misc/export3/sudou/sound_data/datasets/"
    
    for datadir in ["multi_segdata"+str(classes) + "_"+str(image_size)+"_no_sound_random_sep_72/", 
                    #"multi_segdata"+str(classes) + "_"+str(image_size)+"_-20dB_random_sep_72/", 
                    #"multi_segdata"+str(classes) + "_"+str(image_size)+"_no_sound/", 
                    #"multi_segdata"+str(classes) + "_"+str(image_size)+"_-30dB/", 
                    #"multi_segdata"+str(classes) + "_"+str(image_size)+"_-20dB_random/", 
                    #"multi_segdata"+str(classes) + "_"+str(image_size)+"_-10dB/", 
                    #"multi_segdata"+str(classes) + "_"+str(image_size)+"_0dB/"
                    ]:
        dataset = datasets_dir + datadir    
        segdata_dir = dataset + "train/"
        valdata_dir = dataset + "val/"
        
        labelfile = dataset + "label.csv"
        label = pd.read_csv(filepath_or_buffer=labelfile, sep=",", index_col=0)            
        
        for Model in [#"CNN8", "CRNN8", "BiCRNN8", 
                      #"SELD_CNN8", "SELD_BiCRNN8", 
                      #"WUNet", 
                      "SSL_Deeplab", 
                      #"CR_UNet", 
                      #"aux_Mask_UNet", "aux_Mask_Deeplab", 
                      #"aux_enc_UNet", "aux_enc_Deeplab", 
                      #"Cascade"
                      ]:
            
            if Model == "Cascade":
                loss = "categorical_crossentropy"

            for vonMises in [False]:
                for ipd in [True]:
                    for mic_num in [8]: # 1 or 8                        
                        for complex_input in [True]:
                            channel = 0
                            if mic_num == 1:
                                if complex_input == True and ipd == False:
                                    channel = 3
                                elif complex_input == False and ipd == False and vonMises == False:
                                    channel = 1
                            else:                                
                                if complex_input == True:
                                    if ipd == True and vonMises == False:
                                        channel = 15
                                    elif vonMises == True and ipd == False:
                                        channel = 24
                                    elif vonMises == False and ipd == False:
                                        channel = 8
                                    elif vonMises == False and ipd == False:
                                        channel = 24
                                    else:
                                        continue

                                elif complex_input == False and ipd == False and vonMises == False:
                                    channel = 8
                            
                            if channel == 0:
                                continue
                            
                            for Sed_Model in ["BiCRNN8"]:
                                if Model == "Mask_UNet" or Model == "Mask_Deeplab":
                                    sed_model, num_layer = load_sed_model(Sed_Model)
                                    mask=True
                                    aux = False
                                    trainable = False # SED mask
                                elif Model == "aux_Mask_UNet" or Model == "aux_Mask_RNN_UNet" or Model == "aux_Mask_Deeplab":
                                    sed_model, num_layer = load_sed_model(Sed_Model)
                                    mask = True
                                    aux = True
                                    trainable = True # SED mask
                                elif Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
                                    mask = False
                                    aux = True
                                else:
                                    mask = False
                                    aux = False
                                                
                                load_number = 10000
            
                                
                                model_name = Model+"_"+str(classes)+"class_"+str(ang_reso)+"direction_" + str(mic_num)+"ch_cin"+str(complex_input) + "_ipd"+str(ipd) + "_vonMises"+str(vonMises)
                                if mask == True:
                                    model_name = model_name + "_"+Sed_Model + "_aux" + str(aux)
                                dir_name = model_name + "_"+datadir
                                date = datetime.datetime.today().strftime("%Y_%m%d")
                                results_dir = "./model_results/" + date + "/" + dir_name
                                
                                if mode == "train":
                                    print("\nTraining start...")
                                    if not os.path.exists(results_dir + "prediction"):
                                        os.makedirs(results_dir + "prediction/")
                                        os.makedirs(results_dir + "checkpoint/")
    
                                    npy_name = "train_" + task + "_" +str(classes)+"class_"+str(ang_reso)+"direction_" + str(mic_num)+"ch_cin"+str(complex_input) + "_ipd"+str(ipd)  + "_vonMises"+str(vonMises) + "_"+str(load_number)
                                    if not os.path.exists(dataset+"X_"+npy_name+".npy"):
                                        X_train, Y_train, max, phase = load(segdata_dir, 
                                                                              n_classes=classes, 
                                                                              load_number=load_number,
                                                                              complex_input=complex_input)
                                        save_npy(X_train, Y_train, max, phase, npy_name)
    
                                    else:
                                        X_train, Y_train, max, phase = load_npy(npy_name)
                                    
                                    if Model == "Cascade":
                                        X_train, Y_train = Segtoclsdata(Y_train)
                                    

                                    # save train condition
                                    train_condition = date + "\t" + results_dir                     + "\n" + \
                                                      "\t"+"Comapare IPD input and normal complex input and 1ch"                          + "\n" + \
                                                      "\t\t segdata_dir, " + segdata_dir            + "\n" + \
                                                      "\t\t valdata_dir, " + valdata_dir            + "\n" + \
                                                      "\t\t X"+str(X_train.shape)+" Y"+str(Y_train.shape)+"\n" \
                                                      "\t\t data_byte,      " + str(X_train.dtype)  + "\n" + \
                                                      "\t\t BATCH_SIZE,     " + str(BATCH_SIZE)     + "\n" + \
                                                      "\t\t NUM_EPOCH,      " + str(NUM_EPOCH)      + "\n" + \
                                                      "\t\t Loss function,  " + loss                + "\n" + \
                                                      "\t\t Learning_rate,  " + str(lr)             + "\n" + \
                                                      "\t\t Mic num,        " + str(mic_num)        + "\n" + \
                                                      "\t\t Complex_input,  " + str(complex_input)  + "\n" + \
                                                      "\t\t IPD input,      " + str(ipd) + "\n" + \
                                                      "\t\t von Mises input," + str(vonMises) + "\n" + \
                                                      "\t\t task     ,      " + task + "\n" + \
                                                      "\t\t Angle reso,     " + str(360 // ang_reso) + "\n" + \
                                                      "\t\t Model,          " + Model               + "\n" + \
                                                      "\t\t classes,        " + str(classes)        + "\n\n\n"
                        
            
                                    print(train_condition)
                                    
                                    with open(results_dir + 'train_condition.txt','w') as f:
                                        f.write(train_condition)
                                    
                                    #msg = create_message("Training start\n" + train_condition)
                                    #send_mail(msg)

                                    
                                    history = train(X_train, Y_train, Model)
                                    plot_history(history, model_name)
                                
                                    with open('research_log.txt','a') as f:
                                        f.write(train_condition)    
                        
                                    #del X_train, Y_train, max, phase
                                    #gc.collect()
                        
                                # prediction            
                                elif not mode == "train":
                                    print("Prediction\n")
                                    date = mode
                                    results_dir = "./model_results/" + date + "/" + dir_name
                                    with open(results_dir + 'train_condition.txt','r') as f:
                                        train_condition = f.read() 
                                        print(train_condition)

                                if load_number >= 1000:
                                    load_number = 1000
    
                                    
                                npy_name = "test_" + task+ "_" +str(classes)+"class_"+str(ang_reso)+"direction_" + str(mic_num)+"ch_cin"+str(complex_input) + "_ipd"+str(ipd)  + "_vonMises"+str(vonMises) + "_"+str(load_number)
                                if not os.path.exists(dataset+"X_"+npy_name+".npy"):
                                    X_test, Y_test, max, phase = load(valdata_dir, 
                                                                      n_classes=classes, 
                                                                      load_number=load_number, 
                                                                      complex_input=complex_input)
                                    save_npy(X_test, Y_test, max, phase, npy_name)
    
                                else:
                                    X_test, Y_test, max, phase  = load_npy(npy_name)
                                
                                if Model == "Cascade":
                                    X_origin = X_test
                                    X_test, sep_num = load_cascade(dataset + "val_hark/", load_number=load_number)
                                
                                start = time.time()
                                Y_pred = predict(X_test, Model)
                                elapsed_time = time.time() - start
                                print("prediction time = ", elapsed_time)
                                
                                if Model == "aux_Mask_UNet" or Model == "aux_Mask_RNN_UNet" or Model == "aux_Mask_Deeplab" or Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
                                    Y_sedp = Y_pred[0]
                                    Y_pred = Y_pred[1]
                                    Y_sedt = ((Y_test.transpose(3,0,1,2).max(2)[:,:,np.newaxis,:] > 0.1) * 1).transpose(1,2,3,0)
                                elif Model == "Cascade":
                                    Y_argmax = np.argmax(Y_pred, axis=1)
                                    Y_pred = np.zeros((load_number, classes, 256, image_size))
                                    datanum = 0
                                    for n in range(load_number):
                                        for sep in range(sep_num[n]):
                                            Y_pred[n][Y_argmax[datanum]] = X_test[datanum].transpose(2,0,1)[0]
                                            datanum += 1
                                    Y_pred = Y_pred.transpose(0,2,3,1)
                                    X_test = X_origin
                                elif Model == "doa_UNet" or Model == "sad_UNet" or Model == "SSL_Deeplab":
                                    Y_pred = Y_pred[1]
                                    
                                if plot == True:
                                    sdr_array = np.zeros((classes, 1))
                                    sir_array = np.zeros((classes, 1))
                                    sar_array = np.zeros((classes, 1))
                                    sdr_num = np.zeros((classes, 1))
                                        
                                    for i in range (0, load_number):
                                        save = False
                                        if i < graph_num:
                                            origin_stft(X_test, no=i)
                                            save = True
                                        if task == "event":
                                            if i < graph_num:
                                                event_plot(Y_test, Y_pred, no=i)
                                        else:
                                            if i < graph_num:
                                                if Model == "aux_Mask_UNet" or Model == "aux_Mask_RNN_UNet" or Model == "aux_Mask_Deeplab" or Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
                                                    event_plot(Y_sedt, Y_sedp, no=i)
                                                plot_stft(Y_test, Y_pred, no=i)
                                            sdr, sir, sar = restore(Y_test, Y_pred, max, phase, no=i, save=save)
                                            if task == "segmentation" and ang_reso == 1:
                                                sdr_array += sdr
                                                sir_array += sir
                                                sar_array += sar
                                                sdr_num += (sdr != 0.000) * 1
                                                                        
                                    if task == "segmentation" and ang_reso == 1:
                                        sdr_array = sdr_array / sdr_num
                                        sir_array = sir_array / sdr_num
                                        sar_array = sar_array / sdr_num
                                        
                                        sdr_array = np.append(sdr_array, sdr_array.mean())
                                        sir_array = np.append(sir_array, sir_array.mean())
                                        sar_array = np.append(sar_array, sar_array.mean())
                                        
                                        np.savetxt(results_dir+"prediction/sdr_"+str(load_number)+".csv", sdr_array, fmt ='%.3f')
                                        print("SDR\n", sdr_array, "\n")   
                                        
                                if task == "event":
                                    Y_pred = (Y_pred > 0.5) * 1
                                    f1 = f1_score(Y_test.ravel(), Y_pred.ravel())
                                    Y_pred = np.argmax(Y_pred, axis=3)
                                    print("F_score", f1)
                                    #Y_pred = Y_pred[:,:,:,np.newaxis]
                                    with open(results_dir + "f1_" + str(f1) + ".txt","w") as f:
                                        f.write(str(f1))   
                                        
                                elif task == "segmentation":
                                    RMS(Y_test, Y_pred) 
                                    if Model == "aux_Mask_UNet" or Model == "aux_Mask_RNN_UNet" or Model == "aux_Mask_Deeplab" or Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
                                        Y_sedp = (Y_sedp > 0.5) * 1
                                        f1 = f1_score(Y_sedt.ravel(), Y_sedp.ravel())
                                        Y_sedp = np.argmax(Y_sedp, axis=3)
                                        print("aux_F_score", f1)
                                        with open(results_dir + "aux_f1_" + str(f1) + ".txt","w") as f:
                                            f.write(str(f1)) 
                                    
                                    f1 = f1_score(((Y_test.max(1) > 0.1) * 1).ravel(),
                                                  ((Y_pred.max(1) > 0.1) * 1).ravel())
                                    print("segmentation F-score =", f1)
                                    with open(results_dir + "segmentation_f1_" + str(f1) + ".txt","w") as f:
                                        f.write(str(f1))  
                                    
                                if not os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
                                    shutil.copy("main.py", results_dir)
                                    if not task == "event":
                                        shutil.copy("Unet.py", results_dir)
                                        shutil.copy("PSPNet.py", results_dir)
                                        shutil.copy("Deeplab.py", results_dir)
                                    elif task == "event":
                                        shutil.copy("CNN.py", results_dir)
                                    #shutil.move("nohup.out", results_dir)
                
                                    # copy to export2
                                    shutil.copytree(results_dir, "/misc/export3/sudou/model_results/" + date + "/" + dir_name)

                                    #msg = create_message("Evaluation finished\n" + train_condition)
                                    #send_mail(msg)
                                                    

    os.remove("Unet.pyc")
    os.remove("PSPNet.pyc")
    os.remove("Deeplab.pyc")
    #os.remove("DC.pyc")
    os.remove("CNN.pyc")
    os.remove("sound.pyc")
