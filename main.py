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
import matplotlib.pyplot as plt

import pydot, graphviz
import pickle
from keras.optimizers import SGD, Adam, Adagrad
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model

#import FCN32, FCN8, Segnet, Deeplab, PSPNet, Unet
import CNN, Unet, DC, PSPNet, Deeplab

from sound import WavfileOperate, Stft
import cmath

import shutil
from mir_eval.separation import bss_eval_sources


import tensorflow as tf
from keras.utils import multi_gpu_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
K.set_session(sess)

import time
import soundfile as sf
from scipy import signal

from sklearn.metrics import f1_score


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
    if multi == False:
        if complex_input == True or VGG > 0:
            input_dim = 3
        else:
            input_dim = 1
    else:
        if complex_input == True or VGG > 0:
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
    labels = np.zeros((load_number, n_classes, 256, image_size), dtype=np.float16)
    if complex_output == True:
        r_labels = np.zeros((load_number, n_classes, 256, image_size), dtype=np.float16)
        i_labels = np.zeros((load_number, n_classes, 256, image_size), dtype=np.float16)
    else:
        r_labels, i_labels = 1, 1
    
    for i in range(load_number):
        data_dir = segdata_dir + str(i) + "/"
        filelist = os.listdir(data_dir)        
        for n in range(len(filelist)):
            if filelist[n][-4:] == ".wav":
                waveform, fs = sf.read(data_dir + filelist[n]) 
                if filelist[n][0:3] == "0__":
                    if multi == False:
                        freqs, t, stft = signal.stft(x=waveform, fs=fs, nperseg=512, 
                                                               return_onesided=False)
                        stft = stft[:, 1:len(stft.T) - 1]
                        if not task == "event":
                            inputs_phase[i] = np.angle(stft)
                        if complex_input == True:
                            inputs[i][1] = stft[:256].real
                            inputs[i][2] = stft[:256].imag
                        inputs[i][0] = abs(stft[:256])
                
                elif filelist[n][:7] == "0_multi":
                    if multi == True:
                        freqs, t, stft = signal.stft(x=waveform.T, fs=fs, nperseg=512, 
                                                               return_onesided=False)
                        stft = stft[:, :, 1:len(stft.T) - 1]
                        if not task == "event":
                            inputs_phase[i] = np.angle(stft[0])
                        for nchan in range(8):
                            if complex_input == True:
                                inputs[i][nchan*3] = abs(stft[nchan][:256])
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
                    labels[i][label.T[filelist[n][:-4]][cat]] += abs(stft[:256])
            
    
    if complex_input == True:
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
            
    if task == "event":    
        labels = labels.max(2)[:,:,np.newaxis,:]
        
    inputs, labels = log(inputs, labels)   
    inputs = np.nan_to_num(inputs)
    labels = np.nan_to_num(labels) 
    inputs += 120
    labels += 120

    inputs, labels, max, r_labels, i_labels = normalize(inputs, labels, r_labels, i_labels)


    if complex_input == True:
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
    labels = labels.transpose(0, 2, 3, 1)  
    
    
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
                              input_width=image_size, nChannels=1, 
                              soft=soft, mul=True).get_model()
        elif Model == "complex_unet":
            model = Unet.UNet(n_classes=classes, input_height=256, 
                              input_width=image_size, nChannels=3, 
                              soft=soft, mul=True).get_model()
        elif Model == "complex_unet2":
            model = Unet.Complex_UNet2(n_classes=classes, input_height=256, 
                                       input_width=image_size, nChannels=3).get_model()
        elif Model == "vgg_unet":
            model = Unet.VGG_UNet(n_classes=classes, input_height=256, 
                                       input_width=image_size, nChannels=3)
        elif Model == "mask_unet":
            model = Unet.Mask_UNet(n_classes=classes, input_height=256, 
                                   input_width=image_size, nChannels=1, 
                                   soft=soft, mul=True) 
        elif Model == "glu_mask_unet":
            model = Unet.GLU_Mask_UNet(n_classes=classes, input_height=256, 
                                   input_width=image_size, nChannels=1, 
                                   soft=soft, mul=True) 
        elif Model == "complex_mask_unet":
            model = Unet.Mask_UNet(n_classes=classes, input_height=256, 
                                   input_width=image_size, nChannels=3, 
                                   soft=soft, mul=True) 
                    
        elif Model == "PSPNet":
            model = PSPNet.build_pspnet(nb_classes=classes, resnet_layers=101, 
                                    input_shape=(256,image_size), activation='softmax')
        elif Model == "Mask_PSPNet":
            model = PSPNet.build_pspnet(nb_classes=classes, resnet_layers=101, 
                                    input_shape=(256,image_size), activation='softmax', 
                                    mask=True)
        elif Model == "Deeplab":
            if multi == True:
                if complex_input == True:
                    model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256,image_size,24), classes=classes, OS=16)
                else:
                    model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256,image_size,8), classes=classes, OS=16)
            else:
                model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256,image_size,1), classes=classes, OS=16)
        elif Model == "Mask_Deeplab":
            model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256,image_size,1), classes=classes, 
                                  OS=16, mask=True)
    
            
        elif Model == "CNN4":
            model = CNN.CNN4(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        elif Model == "CNN8":
            model = CNN.CNN8(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        elif Model == "CRNN":
            model = CNN.CRNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        elif Model == "CRNN8":
            model = CNN.CRNN8(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        elif Model == "BiCRNN8":
            model = CNN.BiCRNN8(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        elif Model == "Double_CRNN8":
            model = CNN.Double_CRNN8(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        elif Model == "RNN":
            model = CNN.RNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        elif Model == "BiRNN":
            model = CNN.BiRNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        elif Model == "UNet":
            model = CNN.Unet(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
            
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


def affinitykmeans(Y, V):
    def norm(tensor):
        square_tensor = K.square(tensor)
        frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
        return frobenius_norm2

    def dot(x, y):
        return K.batch_dot(x, y, axes=(2, 1))

    def T(x):
        return K.permute_dimensions(x, [0, 2, 1])

    V = K.l2_normalize(K.reshape(V, [BATCH_SIZE, -1,
                                     classes]), axis=-1)
    Y = K.reshape(Y, [BATCH_SIZE, -1, classes])

    silence_mask = K.sum(Y, axis=2, keepdims=True)
    V = silence_mask * V

    return norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))



def train(root_dir, X_train, Y_train, Model, Y_train2, Y_train3):
    model, multi_model = read_model(Model)
    
    if gpu_count == 1:
        model.compile(loss=loss, optimizer=Adam(lr=lr),metrics=["accuracy"])
    else:
        multi_model.compile(loss=loss, optimizer=Adam(lr=lr),metrics=["accuracy"])                

    plot_model(model, to_file = results_dir + model_name + '.png')

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1,mode="auto")
    checkpoint = ModelCheckpoint(filepath=results_dir+"/checkpoint/"+model_name+"_{epoch}.hdf5",
                                 save_best_only=False, period=100)

    tensorboard = TensorBoard(log_dir=results_dir, histogram_freq=0, write_graph=True)
    model.summary()

    if complex_output == False:
        if complex_input == True  or multi == True:
            X_train = [X_train, X_train.transpose(3,0,1,2)[0][np.newaxis,:,:,:].transpose(1,2,3,0)]
        if gpu_count == 1:            
            history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
                                epochs=NUM_EPOCH, verbose=1, validation_split=0.1,
                                callbacks=[checkpoint, tensorboard, early_stopping])        
        else:       
            history = multi_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
                                epochs=NUM_EPOCH, verbose=1, validation_split=0.1,
                                callbacks=[checkpoint, early_stopping, tensorboard])
                        
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
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(results_dir + model_name + "_accuracy.png")
    plt.show()
    plt.close()
    """
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.ylim(0.0,0.0025)
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(results_dir + model_name + "_loss.png")
    plt.show()
    
    plt.close()

def predict(X_test, Model):
    model, multi_model = read_model(Model)
    model.load_weights(results_dir + model_name + '_weights.hdf5')

    print("\npredicting...")
    if complex_input == True:
        X_test = [X_test, X_test.transpose(3,0,1,2)[0][np.newaxis,:,:,:].transpose(1,2,3,0)]
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
    plt.clim(0, 1)
    plt.colorbar()
    plt.savefig(pred_dir + filename[:-4] + ".png")
    plt.show()
    plt.close()


def event_plot(Y_true, Y_pred, no=0):
    pred_dir = pred_dir_make(no)

    plt.pcolormesh((Y_true[no][0].T))
    plt.title("truth")
    plt.clim(0,1)
    plt.colorbar()
    plt.savefig(pred_dir + "true.png")
    plt.show()
    plt.close()
    
    plt.pcolormesh((Y_pred[no][0].T))
    plt.title("prediction")
    plt.clim(0,1)
    plt.colorbar()
    plt.savefig(pred_dir + "pred.png")
    plt.show()
    


def plot_stft(Y_true, Y_pred, no=0):
    pred_dir = pred_dir_make(no)

    if complex_output == True:
        Y_pred = Y_pred[0].transpose(0, 3, 1, 2)
    else:
        Y_pred = Y_pred.transpose(0, 3, 1, 2)
    Y_true = Y_true.transpose(0, 3, 1, 2)
        
    Y_true_total = np.zeros((256, image_size))
    Y_pred_total = np.zeros((256, image_size))
    for i in range(classes):
        if Y_true[no][i].max() > 0: #含まれているクラスのみグラフ表示
            
            plt.pcolormesh((Y_true[no][i]))
            plt.title(label.index[i] + "_truth")
            #plt.title("category_" + str(i) + "_truth")
            plt.clim(0,1)
            plt.colorbar()
            plt.savefig(pred_dir + label.index[i] + "_true.png")
            #plt.savefig(pred_dir + "category_" + str(i) + "_truth.png")
            plt.show()
            plt.close()
            
            plt.pcolormesh((Y_pred[no][i]))
            plt.title(label.index[i] + "_prediction")
            #plt.title("category_" + str(i) + "_prediction")
            plt.clim(0,1)
            plt.colorbar()
            plt.savefig(pred_dir + label.index[i] + "_pred.png")
            #plt.savefig(pred_dir + "category_" + str(i) + "_pred.png")
            plt.show()
            plt.close()
            
        Y_true_total += (Y_true[no][i] > 0.45) * (i + 4)
        Y_pred_total += (Y_pred[no][i] > 0.45) * (i + 4)
    
    filename = get_filename(no)
    
    plt.pcolormesh((Y_true_total), cmap="gist_ncar")
    plt.title(str(no) + "__" + filename + "_truth")
    plt.clim(0, Y_true_total.max())
    plt.savefig(pred_dir + filename + "_truht.png")
    plt.show()
    plt.close()
    
    plt.pcolormesh((Y_pred_total), cmap="gist_ncar")
    plt.title(str(no) + "__" + filename + "_prediction")
    plt.clim(0, Y_true_total.max())
    plt.savefig(pred_dir + filename + "_pred.png")
    plt.show() 
    plt.close()


def restore(Y_true, Y_pred, max, phase, no=0, class_n=1):
    pred_dir = pred_dir_make(no)
    
    if complex_output == True:
        Y_pred_r = Y_pred[1].transpose(0, 3, 1, 2)
        Y_pred_i = Y_pred[2].transpose(0, 3, 1, 2)
        Y_pred = Y_pred[0].transpose(0, 3, 1, 2)
    else:
        Y_pred = Y_pred.transpose(0, 3, 1, 2)
    Y_true = Y_true.transpose(0, 3, 1, 2)    
    
    data_dir = valdata_dir + str(no)
    sdr_array = np.zeros((classes,1))
    sir_array = np.zeros((classes,1))
    sar_array = np.zeros((classes,1))
    
    for class_n in range(classes):
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
        
            Y_Stft = Stft(Y_complex, 16000, label.index[class_n]+"_prediction")
            #Y_Stft = Stft(Y_complex, 16000, "category_"+str(class_n)+"_prediction")
            Y_pred_wave = Y_Stft.scipy_istft()
            Y_pred_wave.plot()
            print(class_n)
            Y_pred_wave.write_wav_sf(dir=pred_dir, filename=None, bit=16)

            if not task == "event":
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
    
    if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
        """
        plt.plot(rms_array, marker='o', linestyle="None")
        plt.plot(overlap_array, marker='o', linestyle="None")
        #plt.plot(total_rms_array, marker='o', linestyle="None")
        plt.title("rms_overlap")
        plt.savefig(results_dir + "rms_laprms_result.png")
        plt.ylim(0,50)
        plt.show()
        plt.close()
        
        plt.plot(rms_array, marker='o', linestyle="None")
        plt.title("rms")
        plt.savefig(results_dir + "rms_result.png")
        plt.ylim(0,50)
        plt.show()
        plt.close()
        """
        plt.plot(area_array, rms_array, marker='o', linestyle="None")
        plt.title("area-rms")
        plt.savefig(results_dir + "area-rms_result.png")
        plt.ylim(0,50)
        plt.show()
        plt.close()
        
        plt.plot(spl_array, overlap_array, marker='o', linestyle="None")
        plt.title("spl-overlap")
        plt.savefig(results_dir + "spl-rms_result.png")
        plt.ylim(0,50)
        plt.show()
        plt.close()

    """
    for i in range(75):
        print(i)
        plt.pcolormesh(Y_true_db[i][0])
        plt.clim(-120,0)
        plt.colorbar()
        plt.show()
        plt.pcolormesh(Y_pred_db[i][0])
        plt.clim(-120,0)
        plt.colorbar()
        plt.show()
    if task == "event":    
        labels = labels.max(2)[:,:,np.newaxis,:]
        labels = ((labels > 0.1) * 1)
    """
    return rms_array


def save_npy(X, Y, max, phase):
    name = "train"
    if not mode == "train":
        name = "test"

    if complex_input == False and VGG == 0:
        np.save(root_dir+"X_"+name+"_dim1_"+str(load_number)+".npy", X)
    else:
        np.save(root_dir+"X_"+name+"_dim3_"+str(load_number)+".npy", X)
    np.save(root_dir+name+"_max_"+str(load_number)+".npy", max)
    
    if not task == "event":
        np.save(root_dir+"Y_"+name+"_"+str(classes)+train_mode+"_"+str(load_number)+".npy", Y)
        np.save(root_dir+name+"_phase_"+str(load_number)+".npy", phase)
    else:
        np.save(root_dir+"event_Y_"+name+"_"+str(classes)+train_mode+"_"+str(load_number)+".npy", Y)
        
    print("npy files were saved\n")
        
        
def load_npy():
    start = time.time()
    name = "train"
    if not mode == "train":
        name = "test"
        
    if complex_input == False and VGG == 0:
        X = np.load(root_dir+"X_"+name+"_dim1_"+str(load_number)+".npy")
    else:
        X = np.load(root_dir+"X_"+name+"_dim3_"+str(load_number)+".npy")
    max = np.load(root_dir+name+"_max_"+str(load_number)+".npy")
    
    if not task == "event":
        Y = np.load(root_dir+"Y_"+name+"_"+str(classes)+train_mode+"_"+str(load_number)+".npy")
        phase = np.load(root_dir+name+"_phase_"+str(load_number)+".npy")
    else:
        Y = np.load(root_dir+"event_Y_"+name+"_"+str(classes)+train_mode+"_"+str(load_number)+".npy")
        phase = 1
    b = time.time() - start
    print(b, "sec")
    print("npy files were loaded\n")
    
    return X, Y, max, phase



if __name__ == '__main__':
    for datadir in ["multi_segdata75_256_no_sound/", 
                    #"segdata76_256_-30dB/", 
                    #"segdata76_256_-20dB/", 
                    #"segdata76_256_-10dB/", 
                    #"segdata76_256_0dB/"
                    ]:
        train_mode = "class"
        classes = 75
        image_size = 256
        #datadir = "segdata75_"+str(image_size)+"_new/"
    #    datadir = "SMALL_STUFF9_"+str(image_size)+"/"
    
        if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
            root_dir = "/home/yui-sudo/document/dataset/sound_segmentation/datasets/"+datadir
            segdata_dir = root_dir + "train/"
            valdata_dir = root_dir + "val/"
            gpu_count = 1
            BATCH_SIZE = 8
            NUM_EPOCH = 10
            load_number = 1
            mode = "train"
            plot = True
        else:
            root_dir = "/misc/export2/sudou/sound_data/datasets/"+datadir
            segdata_dir = root_dir + "train/"
            valdata_dir = root_dir + "val/"
            gpu_count = 3
            BATCH_SIZE = 16 * gpu_count
            NUM_EPOCH = 100
            load_number = 5000
            mode = "train"
            plot = False
        
        labelfile = root_dir + "label.csv"
        label = pd.read_csv(filepath_or_buffer=labelfile, sep=",", index_col=0)    
        
        multi = True
        soft = False
        complex_input = True
        complex_output = False
        VGG = 0                     #0: False, 1: Red 3: White
        task = "seg"
        Model = "Deeplab"
        loss = "mean_squared_error"
        if task == "event":
            loss = "binary_crossentropy"
        lr = 0.01
        model_name = Model + "_" + str(classes) + "_class"
    
        if mode == "train":
            print("\nDataset directory is", segdata_dir, "\n", "Training start...")
            date = datetime.datetime.today().strftime("%Y_%m%d")
            results_dir = "./model_results/" + date + "/" + model_name + "_"+datadir+"/"        
            if not os.path.exists(os.path.join("./model_results", date, 
                                               model_name + "_"+datadir, "prediction")):
                os.makedirs(results_dir + "prediction/")
                os.makedirs(results_dir + "checkpoint/")
                print(results_dir, "was newly made")
            print("Training mode using", date, model_name, "\n")
            print("Loss function is " + loss + "\n")
            
            if not task == "event":
                npfile = "Y_"+mode+"_"+str(classes)+train_mode+"_"+str(load_number)+".npy"
            elif task == "event":
                npfile = "event_Y_"+mode+"_"+str(classes)+train_mode+"_"+str(load_number)+".npy"
            
            
            #if os.path.exists(root_dir+npfile):
            #    X_train, Y_train, max, phase = load_npy()
            #    Y_train_r, Y_train_i = 1,1
            #else:       
            X_train, Y_train, max, phase, Y_train_r, Y_train_i = load(segdata_dir=segdata_dir, 
                                                                      n_classes=classes, 
                                                                      load_number=load_number,
                                                                      complex_input=complex_input)
                #save_npy(X_train, Y_train, max, phase) 
    
            print("X", X_train.shape,"Y", Y_train.shape)
    
            history = train(segdata_dir, X_train, Y_train, Model, Y_train_r, Y_train_i)
            train_condition = "segdata_dir, " + segdata_dir + "\n" + \
                              "valdata_dir, " + valdata_dir + "\n" + \
                              "data_num," + str(load_number) + "\n" + \
                              "image_size, " + str(image_size) + "\n" + \
                              "data_byte, " + str(X_train.dtype) + "\n" + \
                              "BATCH_SIZE," + str(BATCH_SIZE) + "\n" + \
                              "Loss function," + loss + "\n" + \
                              "Learning_rate," + str(lr) + "\n" + \
                              "NUM_EPOCH," + str(NUM_EPOCH) + "\n" + \
                              "Model," + Model + "\n" + \
                              "classes," + str(classes) + "\n"
            
            with open(results_dir + 'train_condition.txt','w') as f:
                f.write(train_condition)
                
            del X_train, Y_train, max, phase, Y_train_r, Y_train_i
            
            
            
        elif not mode == "train":
            date = mode
            results_dir = "./model_results/" + date + "/" + model_name + "_"+datadir+"/"
            print("Prediction mode using", date, model_name, "\n")
        
        
        if load_number >= 1000:
            load_number = 1000
       # if os.path.exists(root_dir+npfile):
       #     X_test, Y_test, max, phase = load_npy()
       #     Y_test_r, Y_test_i = 1,1
        #else:
        X_test, Y_test, max, phase, Y_test_r, Y_test_i = load(valdata_dir, 
                                                              n_classes=classes, 
                                                              load_number=load_number, 
                                                              complex_input=complex_input)
        #    save_npy(X_test, Y_test, max, phase)
    
        Y_pred = predict(X_test, Model)
        #if task == "event":
        #    Y_pred = (Y_pred > 0.5) * 1
        #    f1 = f1_score(Y_test.ravel(), Y_pred.ravel())
        #Y_pred = np.argmax(Y_pred, axis=3)
        #print(f1)
        
        
        
        
        rms = RMS(Y_test, Y_pred) 
        #print(rms)
             
        if plot == True:
            sdr_array = np.array(())
            sir_array = np.array(())
            sar_array = np.array(())
            for i in range (0, load_number):
                origin_stft(X_test, no=i)
                
                if task == "event":
                    event_plot(Y_test, Y_pred, no=i)
                else:
                    plot_stft(Y_test, Y_pred, no=i)
                    sdr, sir, sar = restore(Y_test, Y_pred, max, phase, no=i)
                    sdr_array = np.append(sdr_array, sdr)
                    sir_array = np.append(sir_array, sir)
                    sar_array = np.append(sar_array, sar)
    
            if not task == "event":
                sdr_array = sdr_array.reshape(load_number, classes)
                sir_array = sir_array.reshape(load_number, classes)
                sar_array = sar_array.reshape(load_number, classes)
    
            history = None
            with open(results_dir + 'history.pickle',mode='rb') as f:
                history = pickle.load(f, encoding='latin1')
            plot_history(history, model_name)
            print("val_loss", np.array((history["val_loss"])).min())
            
        if not os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
            shutil.copy("main.py", results_dir)
            if not task == "event":
                shutil.copy("Unet.py", results_dir)
                shutil.copy("PSPNet.py", results_dir)
                shutil.copy("Deeplab.py", results_dir)
            elif task == "event":
                shutil.copy("CNN.py", results_dir)
            shutil.move("nohup.out", results_dir)
        