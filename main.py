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

import CNN, Unet, DC, PSPNet, Deeplab, SELD_CNN, SELD_Deeplab

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
else:
    config.gpu_options.visible_device_list = "0,1,2"
sess = tf.Session(config=config)
K.set_session(sess)

import time
import soundfile as sf
from scipy import signal

from sklearn.metrics import f1_score

import re


def normalize(inputs, labels, r_labels=None, i_labels=None):
    max = inputs.max()
    inputs = inputs / max
    labels = labels / max
    r_labels = r_labels / max
    i_labels = i_labels / max
    
    inputs = np.clip(inputs, 0.0, 1.0)
    labels = np.clip(labels, 0.0, 1.0)
    r_labels = np.clip(r_labels, 0.0, 1.0)
    i_labels = np.clip(i_labels, 0.0, 1.0)
    
    return inputs, labels, max, r_labels, i_labels


def log(inputs, labels):
    inputs += 10**-7
    labels += 10**-7
    inputs = 20 * np.log10(inputs)
    labels = 20 * np.log10(labels)
    
    return inputs, labels


def load(segdata_dir, n_classes=8, load_number=9999999, complex_input=False):   
    print("data loading\n")
    if mic_num == 1:
        if complex_input == True or VGG > 0 or vonMises == True:
            input_dim = 3
        else:
            input_dim = 1
    else:
        if ipd == True:
            input_dim = 15    
        elif complex_input == True or VGG > 0 or vonMises == True:
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
        
    if ang_reso ==1:
        labels = np.zeros((load_number, n_classes, 256, image_size), dtype=np.float16)
    else:
        labels = np.zeros((load_number, n_classes, ang_reso, 256, image_size), dtype=np.float16)

    if complex_output == True:
        r_labels = np.zeros((load_number, n_classes, 256, image_size), dtype=np.float16)
        i_labels = np.zeros((load_number, n_classes, 256, image_size), dtype=np.float16)
    else:
        r_labels, i_labels = 1, 1
    
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
                                if nchan % 8 == 0:
                                    inputs[i][nchan] = abs(stft[nchan][:256])
                                else:
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
                    if complex_output == True:
                        r_labels[i][label.T[filelist[n][:-4]][cat]] = stft[:256].real
                        i_labels[i][label.T[filelist[n][:-4]][cat]] = stft[:256].imag
                    if ang_reso == 1:
                        labels[i][label.T[filelist[n][:-4]][cat]] += abs(stft[:256])
                    else:
                        angle = int(re.sub("\\D", "", direction[dn].split("_")[1])) // (360 // ang_reso)
                        labels[i][label.T[filelist[n][:-4]][cat]][angle] += abs(stft[:256])
                        dn += 1
    
    if complex_input == True and ipd == False and vonMises == False:
        sign = (inputs > 0) * 2 - 1
        sign = sign.astype(np.float16)
        if complex_output == True:
            r_sign = (r_labels > 0) * 2 - 1
            i_sign = (i_labels > 0) * 2 - 1
            r_sign = r_sign.astype(np.float16)
            i_sign = i_sign.astype(np.float16)
            r_labels, i_labels = log(abs(r_labels), abs(i_labels))
            r_labels += 120
            i_labels += 120
        inputs = abs(inputs) ############################# bug fix
            
            
    if task == "event":    
        if ang_reso == 1:
            labels = labels.max(2)[:,:,np.newaxis,:]
        else:
            labels = labels.max(3)#[:,:,:,np.newaxis,:]
        
    if ipd == True or vonMises == True:
        inputs = inputs.transpose(1,0,2,3)
        inputs[0], labels = log(inputs[0], labels)   
        inputs[0] = np.nan_to_num(inputs[0])
        labels = np.nan_to_num(labels) 
        inputs[0] += 120
        labels += 120
        inputs[0], labels, max, r_labels, i_labels = normalize(inputs[0], labels, r_labels, i_labels)
        inputs = inputs.transpose(1,0,2,3)

    else:
        inputs, labels = log(inputs, labels)   
        inputs = np.nan_to_num(inputs)
        labels = np.nan_to_num(labels) 
        inputs += 120
        labels += 120
        inputs, labels, max, r_labels, i_labels = normalize(inputs, labels, r_labels, i_labels)


    if complex_input == True and ipd == False and vonMises == False:
        inputs = inputs * sign
        if complex_output == True:
            r_labels = r_labels * r_sign
            i_labels = i_labels * i_sign
            r_labels = r_labels.transpose(0, 2, 3, 1)
            i_labels = i_labels.transpose(0, 2, 3, 1)
    
    if task == "event":
        labels = ((labels > 0.1) * 1)
        """
        labels = labels.transpose(1,0,2,3)
        labels[classes - 1] = labels.max(0) * -1 + 1
        labels = labels.transpose(1,0,2,3)
        """
    if task == "double_event":
        t = np.repeat(labels.max(2)[:,:,np.newaxis,:], 256, axis=2)
        f = np.repeat(labels.max(3)[:,:,:,np.newaxis], 256, axis=3)
        t = ((t > 0.1) * 1)
        #f = ((f > 0.5) * 1)
        labels = t * f
        
    inputs = inputs.transpose(0, 2, 3, 1)
    if ang_reso == 1 or task == "event":
        labels = labels.transpose(0, 2, 3, 1)  
    else:
        labels = labels.transpose(0, 3, 4, 1, 2)
        labels = labels.reshape((load_number, 256, image_size, n_classes * ang_reso))
    
    if VGG > 0:
        inputs = inputs.transpose(3,0,1,2)
        if VGG == 1:
            inputs[1:3] = 0       # R only
        elif VGG == 3:
            inputs[1] = inputs[0]
            inputs[2] = inputs[0] # Grayscale to RGB
        inputs = inputs.transpose(1,2,3,0)
    
    return inputs, labels, max, inputs_phase, r_labels, i_labels


def read_model(Model):
    with tf.device('/cpu:0'):
        if Model == "unet":    
            model = Unet.UNet(n_classes=classes, 
                              input_height=256, 
                              input_width=image_size, nChannels=channel, 
                              soft=soft, mul=True).get_model()
        elif Model == "complex_unet2":
            model = Unet.Complex_UNet2(n_classes=classes, input_height=256, 
                                       input_width=image_size, nChannels=3).get_model()
        elif Model == "vgg_unet":
            model = Unet.VGG_UNet(n_classes=classes, input_height=256, 
                                       input_width=image_size, nChannels=3)
        elif Model == "glu_mask_unet":
            model = Unet.GLU_Mask_UNet(n_classes=classes, input_height=256, 
                                   input_width=image_size, nChannels=channel, 
                                   soft=soft, mul=True) 
        elif Model == "complex_mask_unet":
            model = Unet.Mask_UNet(n_classes=classes, input_height=256, 
                                   input_width=image_size, nChannels=channel, 
                                   soft=soft, mul=True) 
                    
        elif Model == "aux_Mask_UNet":
            model = Unet.Mask_UNet(n_classes=classes, input_height=256, 
                                   input_width=image_size, nChannels=1, 
                                   soft=soft, mul=True, trainable=True, 
                                   sed_model=sed_model, num_layer=num_layer, aux=aux) 
        elif Model == "Mask_UNet":
            model = Unet.Mask_UNet(n_classes=classes, input_height=256, 
                                   input_width=image_size, nChannels=1, 
                                   soft=soft, mul=True, trainable=True, 
                                   sed_model=sed_model, num_layer=num_layer) 
        elif Model == "UNet":
            model = Unet.UNet(n_classes=classes * ang_reso, input_height=256, 
                                   input_width=image_size, nChannels=channel)            
        elif Model == "RNN_UNet":
            model = Unet.RNN_UNet(n_classes=classes * ang_reso, input_height=256, 
                                   input_width=image_size, nChannels=channel)                
        elif Model == "CR_UNet":
            model = Unet.CR_UNet(n_classes=classes * ang_reso, input_height=256, 
                                   input_width=image_size, nChannels=channel) 
        elif Model == "Pre_UNet":
            model = Unet.Pre_UNet(n_classes=classes * ang_reso, input_height=256, 
                                   input_width=image_size, nChannels=channel)            
        elif Model == "Pre_RNN_UNet":
            model = Unet.Pre_RNN_UNet(n_classes=classes * ang_reso, input_height=256, 
                                   input_width=image_size, nChannels=channel)                

        elif Model == "PSPNet":
            model = PSPNet.build_pspnet(nb_classes=classes, resnet_layers=101, 
                                    input_shape=(256,image_size), activation='softmax')
        elif Model == "Mask_PSPNet":
            model = PSPNet.build_pspnet(nb_classes=classes, resnet_layers=101, 
                                    input_shape=(256,image_size), activation='softmax', 
                                    mask=True)
        elif Model == "Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                              input_shape=(256, image_size, channel), 
                              classes=classes * ang_reso, OS=16, mul=mul, 
                              soft=soft, sed_model=sed_model, num_layer=num_layer)
        elif Model == "RNN_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                              input_shape=(256, image_size, channel), 
                              classes=classes * ang_reso, OS=16, mul=mul, 
                              soft=soft, RNN=True)
            
        elif Model == "Mask_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256,image_size,1), classes=classes, 
                                  OS=16, mask=True, mul=mul, soft=soft, trainable=True, 
                                  sed_model=sed_model, num_layer=num_layer)
        elif Model == "aux_Mask_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256,image_size,1), classes=classes, 
                                  OS=16, mask=True, mul=mul, soft=soft, trainable=True, 
                                  sed_model=sed_model, num_layer=num_layer, aux=aux)
            
        elif Model == "CNN4":
            model = CNN.CNN4(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=channel)
        elif Model == "CNN8":
            model = CNN.CNN8(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=channel)
        elif Model == "CRNN":
            model = CNN.CRNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=channel)
        elif Model == "CRNN8":
            model = CNN.CRNN8(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=channel)
        elif Model == "BiCRNN8":
            model = CNN.BiCRNN8(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=channel)
        elif Model == "Double_CRNN8":
            model = CNN.Double_CRNN8(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=channel)
        elif Model == "RNN":
            model = CNN.RNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=channel)
        elif Model == "BiRNN":
            model = CNN.BiRNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=channel)
        
        elif Model == "SELD_CRNN":
            if complex_input == False:
                model = SELD_CNN.CRNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=mic_num)

            elif ipd == True:
                model = SELD_CNN.CRNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=(mic_num-1)*2+1)
            else:
                model = SELD_CNN.CRNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=mic_num * 3)
        
        elif Model == "SELD_Deeplab":
            if complex_input == False:
                model = SELD_Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256, image_size, mic_num), 
                                  classes=classes, OS=16, mul=mul, soft=soft)
            elif ipd == True:
                model = SELD_Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256, image_size, (mic_num-1)*2+1), 
                                  classes=classes, OS=16, mul=mul, soft=soft)
            else:
                model = SELD_Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256, image_size, mic_num * 3), 
                                  classes=classes, OS=16, mul=mul, soft=soft)
            
        elif Model == "GLU":
            model = DC.GLU_DC(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        elif Model == "LSTM":
            model = DC.LSTM_DC(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        
    if gpu_count > 1:
        multi_model = multi_gpu_model(model, gpus=gpu_count)
    else:
        multi_model = model
        
    return model, multi_model



def train(X_train, Y_train, Model, Y_train2, Y_train3):
    model, multi_model = read_model(Model)
    
    if gpu_count == 1:
        model.compile(loss=loss, optimizer=Adam(lr=lr),metrics=["accuracy"])
    else:
        multi_model.compile(loss=loss, optimizer=Adam(lr=lr),metrics=["accuracy"])                

    plot_model(model, to_file = results_dir + model_name + '.png')

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1,mode="auto")
    checkpoint = ModelCheckpoint(filepath=results_dir+"/checkpoint/"+model_name+"_{epoch}.hdf5", save_best_only=False, period=100)
    tensorboard = TensorBoard(log_dir=results_dir, histogram_freq=0, write_graph=True)

    model.summary()

    if complex_output == False:
        if task == "segmentation":
            if complex_input == True  or mic_num > 1 or ipd == True:
                X_train = [X_train, 
                           X_train.transpose(3,0,1,2)[0][np.newaxis,:,:,:].transpose(1,2,3,0)]
            
            if Model == "aux_Mask_UNet" or Model == "aux_Mask_Deeplab":
                Y_train = [((Y_train.transpose(3,0,1,2).max(2)[:,:,np.newaxis,:] > 0.1) * 1).transpose(1,2,3,0), 
                           Y_train]
        
        if gpu_count == 1:            
            history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
                                epochs=NUM_EPOCH, verbose=1, validation_split=0.1,
                                callbacks=[checkpoint, tensorboard, early_stopping])        
        else:       
            history = multi_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
                                epochs=NUM_EPOCH, verbose=1, validation_split=0.1,
                                callbacks=[early_stopping, tensorboard])                 
    else:
        history = model.fit(X_train, [Y_train, Y_train2, Y_train3], 
                            batch_size=BATCH_SIZE, epochs=NUM_EPOCH, 
                            verbose=1, validation_split=0.1,
                            callbacks=[checkpoint, early_stopping, tensorboard])

    with open(results_dir + "history.pickle", mode="wb") as f:
        pickle.dump(history.history, f)

    model_json = model.to_json()
    with open(results_dir + "model.json", mode="w") as f:
        f.write(model_json)
    
    model.save_weights(results_dir + model_name + '_weights.hdf5')
    
    return history



def plot_history(history, model_name):
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(results_dir + model_name + "_accuracy.png")
    plt.close()
    """
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
        if complex_output == True:
            Y_pred = Y_pred[0].transpose(0, 3, 1, 2)
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

    if complex_output == True:
        Y_pred = Y_pred[0].transpose(0, 3, 1, 2)
    else:
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


def restore(Y_true, Y_pred, max, phase, no=0, class_n=1):
    plot_num = classes * ang_reso

    pred_dir = pred_dir_make(no)
    
    if complex_output == True:
        Y_pred_r = Y_pred[1].transpose(0, 3, 1, 2)
        Y_pred_i = Y_pred[2].transpose(0, 3, 1, 2)
        Y_pred = Y_pred[0].transpose(0, 3, 1, 2)
    else:
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

            if complex_output == True:
                Y_linear_r = 10 ** ((Y_pred_r[no][class_n] * max - 120) / 20)
                Y_linear_r = np.vstack((Y_linear_r, Y_linear_r[::-1]))
                Y_linear_i = 10 ** ((Y_pred_i[no][class_n] * max - 120) / 20)
                Y_linear_i = np.vstack((Y_linear_i, Y_linear_i[::-1]))

            Y_complex = np.zeros((512, image_size), dtype=np.complex128)
            for i in range (512):
                for j in range (image_size):
                    if complex_output == False:
                        Y_complex[i][j] = cmath.rect(Y_linear[i][j], phase[no][i][j])
                    else:
                        Y_complex[i][j] = cmath.rect(Y_linear[i][j], phase[no][i][j])
                        #Y_complex[i][j] = cmath.rect(Y_linear[i][j], np.arctan2(Y_linear_r[i][j], Y_linear_i[i][j]))
                        #Y_complex[i][j] = cmath.rect(np.sqrt(Y_linear_r[i][j]**2 + Y_linear_i[i][j]**2), np.arctan2(Y_linear_r[i][j], Y_linear_i[i][j]))
            if ang_reso == 1:
                Y_Stft = Stft(Y_complex, 16000, label.index[class_n]+"_prediction")
            else:
                Y_Stft = Stft(Y_complex, 16000, label.index[i % ang_reso * 0] + "_" + str((360 // ang_reso) * (class_n % ang_reso)) + "deg_prediction")
                
            Y_pred_wave = Y_Stft.scipy_istft()
            print(class_n)
            Y_pred_wave.write_wav_sf(dir=pred_dir, filename=None, bit=16)

            if task == "segmentation" and ang_reso == 1:
                Y_pred_wave = Y_pred_wave.norm_sound
                Y_true_wave = WavfileOperate(data_dir + "/" + label.index[class_n] + ".wav").wavedata.norm_sound            
                Y_true_wave = Y_true_wave[:len(Y_pred_wave)]    
                sdr, sir, sar, per = bss_eval_sources(Y_true_wave[np.newaxis,:], Y_pred_wave[np.newaxis,:], compute_permutation=True)
                print(sdr)
                
                sdr_array[class_n] = sdr
                sir_array[class_n] = sir
                sar_array[class_n] = sar    
            
    return sdr_array, sir_array, sar_array
    

def RMS(Y_true, Y_pred):
    if complex_output == True:
        Y_pred = Y_pred[0].transpose(0, 3, 1, 2)
    else:
        Y_pred = Y_pred.transpose(0, 3, 1, 2)
    Y_true = Y_true.transpose(0, 3, 1, 2)    
        
    Y_pred_db = (Y_pred * max - 120)
    Y_true_db = (Y_true * max - 120)
    total_rms_array = np.zeros(classes + 1)
    rms_array = np.zeros(classes + 1)
    num_array = np.zeros(classes + 1)
    area_array = np.zeros(classes + 1)
    spl_array =  np.zeros(classes + 1)
    overlap_array = np.zeros(classes + 1)
    lapnum_array = np.zeros(classes + 1)
    for no in range(load_number):
        lap_detect = ((Y_true[no].max(1) > 0.5).sum(0) > 1)
        for class_n in range(classes):
            if Y_true[no][class_n].max() > 0: #含まれているクラスのみRMS
                
                num_array[classes] += 1
                num_array[class_n] += 1
                on_detect = Y_true[no][class_n].max(0) > 0.000
                lap_on_detect = on_detect * lap_detect
                
                per_rms = ((Y_true_db[no][class_n] - Y_pred_db[no][class_n]) ** 2).mean(0)
                rms_array[class_n] += per_rms.sum()  / on_detect.sum()
                
                if lap_on_detect.sum() > 0:
                    lapnum_array[classes] += 1
                    lapnum_array[class_n] += 1
                    overlap_array[class_n] += (per_rms * lap_on_detect).sum() / lap_on_detect.sum()
                    
                area_array[class_n] += ((Y_true[no][class_n] > 0.3) * 1).sum()
                spl_array[class_n] += ((Y_true_db[no][class_n]+120).mean(0)*lap_on_detect).sum() / lap_on_detect.sum() - 120
                total_rms_array[class_n] += per_rms.mean()
    rms_array[classes] = rms_array.sum()
    rms_array = np.sqrt(rms_array / num_array)
    overlap_array[classes] = overlap_array.sum()
    overlap_array = np.sqrt(overlap_array / lapnum_array)
    area_array[classes] = area_array.sum()
    area_array = area_array // num_array
    spl_array[classes] = spl_array.sum()
    spl_array = spl_array // num_array
    total_rms_array[classes] = total_rms_array.sum()
    total_rms_array = np.sqrt(total_rms_array / num_array)
    #print(area_array)
    #print("total\n", total_rms_array, "\n") 
    print("per-class\n", rms_array, "\n")     
    print("overlap\n", overlap_array, "\n") 
    print("num\n", num_array, "\n")
    print("lapnum\n", lapnum_array, "\n")

    np.savetxt(results_dir+"prediction/class_num_"+str(load_number)+".csv", num_array)
    np.savetxt(results_dir+"prediction/class_rms_"+str(load_number)+".csv", rms_array, fmt ='%.3f')
    np.savetxt(results_dir+"prediction/lap_rms_"+str(load_number)+".csv", overlap_array, fmt ='%.3f')
    np.savetxt(results_dir+"prediction/class_area_"+str(load_number)+".csv", area_array)
    np.savetxt(results_dir+"prediction/total_num_"+str(load_number)+".csv", num_array)
    
    
    plt.plot(rms_array, marker='o', linestyle="None")
    plt.plot(overlap_array, marker='o', linestyle="None")
    #plt.plot(total_rms_array, marker='o', linestyle="None")
    plt.title("rms_overlap")
    plt.xlabel("overlap")
    plt.ylabel('rms')
    plt.savefig(results_dir + "rms_laprms_result.png")
    plt.ylim(0, 50)
    plt.close()

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

    plt.plot(spl_array, overlap_array, marker='o', linestyle="None")
    plt.title("spl-overlap")
    plt.xlabel("")
    plt.ylabel('')
    plt.savefig(results_dir + "spl-rms_result.png")
    plt.ylim(0, 50)
    plt.close()

    """
    for i in range(75):
        print(i)
        plt.pcolormesh(Y_true_db[i][0])
        plt.clim(-120,0)
        plt.colorbar()
        plt.pcolormesh(Y_pred_db[i][0])
        plt.clim(-120,0)
        plt.colorbar()
    if task == "event":    
        labels = labels.max(2)[:,:,np.newaxis,:]
        labels = ((labels > 0.1) * 1)
    """
    return rms_array


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
        sed_model = CNN.CNN8(n_classes=75, input_height=256, input_width=image_size, nChannels=1)
        sed_model.load_weights(os.getcwd()+"/model_results/2019_0917/CNN8_75class_1direction_1ch_mulTrue_cinFalse_ipdFalse_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/CNN8_75class_1direction_1ch_mulTrue_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")
    elif Model == "CRNN8":
        sed_model = CNN.CRNN8(n_classes=75, input_height=256, input_width=image_size, nChannels=1)
        sed_model.load_weights(os.getcwd()+"/model_results/2019_0917/CRNN8_75class_1direction_1ch_mulTrue_cinFalse_ipdFalse_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/CRNN8_75class_1direction_1ch_mulTrue_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")
    elif Model == "BiCRNN8":
        sed_model = CNN.BiCRNN8(n_classes=75, input_height=256, input_width=image_size, nChannels=1)
        sed_model.load_weights(os.getcwd()+"/model_results/2019_0918/BiCRNN8_75class_1direction_1ch_mulTrue_cinFalse_ipdFalse_vonMisesFalse_multi_segdata75_256_no_sound_random_sep/BiCRNN8_75class_1direction_1ch_mulTrue_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")
    
    num_layer = len(sed_model.layers)

    return sed_model, num_layer


if __name__ == '__main__':
    train_mode = "class"
    classes = 75
    image_size = 256
    task = "segmentation"
    ang_reso = 1
    
    if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
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

    mul = True   #multiply
    soft = False # softmax

    Y_train_r, Y_train_i, Y_test_r, Y_test_i = 1,1,1,1
    if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
        datasets_dir = "/home/yui-sudo/document/dataset/sound_segmentation/datasets/"
    else:
        datasets_dir = "/misc/export2/sudou/sound_data/datasets/"
    
    for datadir in ["multi_segdata"+str(classes) + "_"+str(image_size)+"_no_sound_random_sep/", 
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
        
        for Model in ["aux_Mask_UNet", "aux_Mask_Deeplab"]:
            for Sed_Model in ["CNN4", "CNN8", "BiCRNN8"]:
                if Model == "Mask_UNet" or Model == "Mask_Deeplab":
                    sed_model, num_layer = load_sed_model(Sed_Model)
                    aux = False
                elif "aux_Mask_UNet" or Model == "aux_Mask_Deeplab":
                    sed_model, num_layer = load_sed_model(Sed_Model)
                    aux = True
                    
                for vonMises in [False]:
                    for ipd in [False]:
                        for mic_num in [1]: # 1 or 8                        
                            for complex_input in [False]:
                                complex_output = False
                                VGG = 0                     #0: False, 1: Red 3: White
                                
    
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
                                            channel = 24
                                        else:
                                            continue
                                    elif complex_input == False and ipd == False and vonMises == False:
                                        channel = 8
                                
                                if channel == 0:
                                    continue
                            
                                load_number = 10000
            
                                
                                model_name = Model+"_"+str(classes)+"class_"+str(ang_reso)+"direction_" + str(mic_num)+"ch_mul"+str(mul) + "_cin"+str(complex_input) + "_ipd"+str(ipd) + "_vonMises"+str(vonMises) + "_"+Sed_Model + "_aux" + aux
                                dir_name = model_name + "_"+datadir
                                date = datetime.datetime.today().strftime("%Y_%m%d")
                                results_dir = "./model_results/" + date + "/" + dir_name
                                
                                if mode == "train":
                                    print("\nTraining start...")
                                    if not os.path.exists(results_dir + "prediction"):
                                        os.makedirs(results_dir + "prediction/")
                                        os.makedirs(results_dir + "checkpoint/")
    
                                    npy_name = "train_" + task + "_" +str(classes)+"class_"+str(ang_reso)+"direction_" + str(mic_num)+"ch_cin"+str(complex_input) + "_ipd"+str(ipd)  + "_vonMises"+str(vonMises)+"_"+str(load_number)
                                    if not os.path.exists(dataset+"X_"+npy_name+".npy"):
                                        X_train, Y_train, max, phase, Y_train_r, Y_train_i = load(segdata_dir, 
                                                                                                  n_classes=classes, 
                                                                                                  load_number=load_number,
                                                                                                  complex_input=complex_input)
                                        save_npy(X_train, Y_train, max, phase, npy_name)
    
                                    else:
                                        X_train, Y_train, max, phase = load_npy(npy_name)
                                    
                        
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
                                                      "\t\t Multiply,       " + str(mul)            + "\n" + \
                                                      "\t\t Softmax,        " + str(soft)           + "\n" + \
                                                      "\t\t Complex_input,  " + str(complex_input)  + "\n" + \
                                                      "\t\t Complex_output, " + str(complex_output) + "\n" + \
                                                      "\t\t IPD input,      " + str(ipd) + "\n" + \
                                                      "\t\t von Mises input," + str(vonMises) + "\n" + \
                                                      "\t\t task     ,      " + task + "\n" + \
                                                      "\t\t Angle reso,     " + str(360 // ang_reso) + "\n" + \
                                                      "\t\t Model,          " + Model               + "\n" + \
                                                      "\t\t classes,        " + str(classes)        + "\n\n\n"
                        
            
                                    print(train_condition)
                                    
                                    with open(results_dir + 'train_condition.txt','w') as f:
                                        f.write(train_condition)
                                    
                                    history = train(X_train, Y_train, Model, Y_train_r, Y_train_i)
                                    plot_history(history, model_name)
                                
                                    with open('research_log.txt','a') as f:
                                        f.write(train_condition)    
                        
                                # prediction            
                                elif not mode == "train":
                                    print("Prediction\n")
                                    date = mode
                                    results_dir = "./model_results/" + date + "/" + dir_name
                                    with open(results_dir + 'train_condition.txt','r') as f:
                                        train_condition = f.read() 
                                        print(train_condition)
                                    
                                if load_number >= 100:
                                    load_number = 50
    
                                    
                                npy_name = "test_" + task+ "_" +str(classes)+"class_"+str(ang_reso)+"direction_" + str(mic_num)+"ch_cin"+str(complex_input) + "_ipd"+str(ipd)  + "_vonMises"+str(vonMises)+"_"+str(load_number)
                                if not os.path.exists(dataset+"X_"+npy_name+".npy"):
                                    X_test, Y_test, max, phase, Y_test_r, Y_test_i = load(valdata_dir, 
                                                                                      n_classes=classes, 
                                                                                      load_number=load_number, 
                                                                                      complex_input=complex_input)
                                    save_npy(X_test, Y_test, max, phase, npy_name)
    
                                else:
                                    X_test, Y_test, max, phase = load_npy(npy_name)
                                
                                Y_pred = predict(X_test, Model)
                                if Model == "aux_Mask_UNet" or Model == "aux_Mask_Deeplab":
                                    Y_sedp = Y_pred[0]
                                    Y_pred = Y_pred[1]
                                    Y_sedt = Y_test[0]
                                    Y_test = Y_test[1]
                                         
                                if plot == True:
                                    sdr_array, sir_array, sar_array = np.array(()) ,np.array(()), np.array(())
                                    for i in range (0, load_number):
                                        origin_stft(X_test, no=i)
                                        
                                        if task == "event":
                                            event_plot(Y_test, Y_pred, no=i)
                                        elif Model == "aux_Mask_UNet" or Model == "aux_Mask_Deeplab":
                                            event_plot(Y_sedt, Y_sedp, no=i)
                                            plot_stft(Y_test, Y_pred, no=i)
                                            sdr, sir, sar = restore(Y_test, Y_pred, max, phase, no=i)
                                            sdr_array = np.append(sdr_array, sdr)
                                            sir_array = np.append(sir_array, sir)
                                            sar_array = np.append(sar_array, sar)
                                        else:
                                            plot_stft(Y_test, Y_pred, no=i)
                                            sdr, sir, sar = restore(Y_test, Y_pred, max, phase, no=i)
                                            sdr_array = np.append(sdr_array, sdr)
                                            sir_array = np.append(sir_array, sir)
                                            sar_array = np.append(sar_array, sar)
                            
                                    if task == "segmentation" and ang_reso == 1:
                                        sdr_array = sdr_array.reshape(load_number, classes)
                                        sir_array = sir_array.reshape(load_number, classes)
                                        sar_array = sar_array.reshape(load_number, classes)
                            
                                if task == "event":
                                    Y_pred = (Y_pred > 0.5) * 1
                                    f1 = f1_score(Y_test.ravel(), Y_pred.ravel())
                                    Y_pred = np.argmax(Y_pred, axis=3)
                                    print("F1_score", f1)
                                    #Y_pred = Y_pred[:,:,:,np.newaxis]
                                    with open(results_dir + "f1_" + str(f1) + ".txt","w") as f:
                                        f.write(str(f1))   
                                        
                                elif task == "segmentation":
                                    rms = RMS(Y_test, Y_pred) 
                                    print("Total RMSE", rms)
                                    if Model == "aux_Mask_UNet" or Model == "aux_Mask_Deeplab":
                                        Y_sedp = (Y_sedp > 0.5) * 1
                                        f1 = f1_score(Y_sedt.ravel(), Y_sedp.ravel())
                                        Y_sedp = np.argmax(Y_sedp, axis=3)
                                        print("F1_score", f1)
                                        with open(results_dir + "f1_" + str(f1) + ".txt","w") as f:
                                            f.write(str(f1))   
                                    
                                    
                                if not os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
                                    shutil.copy("main.py", results_dir)
                                    if not task == "event":
                                        shutil.copy("Unet.py", results_dir)
                                        shutil.copy("PSPNet.py", results_dir)
                                        shutil.copy("Deeplab.py", results_dir)
                                    elif task == "event":
                                        shutil.copy("CNN.py", results_dir)
                                        shutil.copy("SELD_CNN.py", results_dir)
                                    #shutil.move("nohup.out", results_dir)
                
                                    # copy to export2
                                    shutil.copytree(results_dir, "/misc/export2/sudou/model_results/" + date + "/" + dir_name)
                                                    

    os.remove("Unet.pyc")
    os.remove("PSPNet.pyc")
    os.remove("Deeplab.pyc")
    os.remove("DC.pyc")
    os.remove("CNN.pyc")
    os.remove("sound.pyc")
