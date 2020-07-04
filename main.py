import os
import re
import gc
import glob
import shutil
import time
import datetime
import cmath
import numpy as np
from scipy import signal
import pandas as pd
import soundfile as sf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from mir_eval.separation import bss_eval_sources

import pydot, graphviz
import pickle
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import plot_model

import read_model, CNN, Unet, Deeplab
import utils

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest' or os.getcwd() == '/home/sudou/python/sound_segtest':
    config.gpu_options.visible_device_list = "0"
else:
    config.gpu_options.visible_device_list = "2"
sess = tf.Session(config=config)
K.set_session(sess)


def normalize(inputs, labels):
    max = inputs.max()
    inputs = inputs / max
    labels = labels / max
    
    inputs = np.clip(inputs, 0.0, 1.0)
    labels = np.clip(labels, 0.0, 1.0)
    
    return inputs, labels, max


def log(inputs, labels):
    inputs = 20 * np.log10(inputs)
    labels = 20 * np.log10(labels)

    inputs = np.nan_to_num(inputs) + 120
    labels = np.nan_to_num(labels) + 120
    
    return inputs, labels


def load(segdata_dir, n_classes=8, load_number=99999, input_dim=1):   
    print("data loading\n")   
        
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
        if task == "event":
            labels = np.zeros((load_number, ang_reso, image_size), dtype=np.float16)
        elif task == "segmentation":
            labels = np.zeros((load_number, ang_reso, 256, image_size), dtype=np.float16)    
    else:
        if task == "event":
            labels = np.zeros((load_number, n_classes, ang_reso, image_size), dtype=np.float16)      
    
    for i in range(load_number):
        data_dir = segdata_dir + str(i) + "/"
        filelist = os.listdir(data_dir)  
        
        with open(segdata_dir + str(i) + "/sound_direction.txt", "r") as f:
            direction = f.read().split("\n")[:-1]
            
        direction_index = 0
        for n in range(len(filelist)):
            if filelist[n][-4:] == ".wav":
                waveform, fs = sf.read(data_dir + filelist[n]) 
                if filelist[n][0:3] == "0__":
                    if mic_num == 1:
                        _, _, stft = signal.stft(x=waveform, fs=fs, nperseg=512, return_onesided=False)
                        stft = stft[:, 1:len(stft.T) - 1]
                        if not task == "event":
                            inputs_phase[i] = np.angle(stft)
                        if complex_input:
                            inputs[i][1] = stft[:256].real
                            inputs[i][2] = stft[:256].imag                        
                        inputs[i][0] = abs(stft[:256])
                
                elif filelist[n][:7] == "0_multi":
                    if mic_num == 8:
                        _, _, stft = signal.stft(x=waveform.T, fs=fs, nperseg=512, return_onesided=False)
                        stft = stft[:, :, 1:len(stft.T) - 1]
                        if not task == "event":
                            inputs_phase[i] = np.angle(stft[0])
                        for nchan in range(mic_num):
                            if ipd:
                                if nchan == 0:
                                    inputs[i][nchan] = abs(stft[nchan][:256])
                                else:
                                    inputs[i][nchan*2-1] = np.cos(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                                    inputs[i][nchan*2] = np.sin(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))                         
                            elif complex_input:
                                inputs[i][nchan * 3] = abs(stft[nchan][:256])
                                inputs[i][nchan*3 + 1] = stft[nchan][:256].real
                                inputs[i][nchan*3 + 2] = stft[nchan][:256].imag
                                
                else:
                    if filelist[n][:-4] == "BGM":
                        continue
                    _, _, stft = signal.stft(x=waveform, fs=fs, nperseg = 512, return_onesided=False)
                    stft = stft[:, 1:len(stft.T) - 1]

                    if ang_reso == 1:
                        if n_classes > 1:
                            if task == "event":                             # SED
                                labels[i][label.T[filelist[n][:-4]][0]] += abs(stft[:256]).max(0)
                            elif task == "segmentation":                    # segmentation
                                labels[i][label.T[filelist[n][:-4]][0]] += abs(stft[:256])                        
                    elif ang_reso > 1:
                        angle = int(re.sub("\\D", "", direction[direction_index].split("_")[1])) // (360 // ang_reso)
                        if n_classes == 1:
                            if task == "event":          # SSL
                                labels[i][angle] += abs(stft[:256]).max(0)     
                            elif task == "segmentation": # SSLS
                                labels[i][angle] += abs(stft[:256])          
                        else: # n_classes > 1
                            if task == "event":                           # SELD
                                labels[i][label.T[filelist[n][:-4]][0]][angle] += abs(stft[:256]).max(0)
                        direction_index += 1

    
    if complex_input == True and ipd == False:
        sign = (inputs > 0) * 2 - 1
        sign = sign.astype(np.float16)
        inputs = abs(inputs)        
        
    if ipd:
        inputs = inputs.transpose(1,0,2,3)
        inputs[0], labels = log(inputs[0], labels)   
        inputs[0], labels, max = normalize(inputs[0], labels)
        inputs = inputs.transpose(1,0,2,3)
    else:
        inputs, labels = log(inputs, labels)   
        inputs, labels, max = normalize(inputs, labels)


    if complex_input == True and ipd == False:
        inputs = inputs * sign
    
    if task == "event":
        labels = ((labels > 0.1) * 1)

    inputs = inputs.transpose(0, 2, 3, 1)
    if n_classes > 1 and ang_reso == 1:
        if task == "event":                                 # SED
            labels = labels.transpose(0, 2, 1)[:,np.newaxis,:,:]
        elif task == "segmentation":                        # segmentation
            labels = labels.transpose(0, 2, 3, 1)  
    elif ang_reso > 1 and n_classes == 1:
        if task == "event":                                 # SSL
            labels = labels.transpose(0, 2, 1)[:,np.newaxis,:,:]
        elif task == "segmentation":                        # SSLS
            labels = labels.transpose(0, 2, 3, 1)  
    else:
        if task == "event":                                 # SELD
            labels = labels.transpose(0, 2, 3, 1)  
        
    return inputs, labels, max, inputs_phase


def load_desed(segdata_dir):
    folderlist = os.listdir(segdata_dir)
    folderlist.sort()
    datanum = len(folderlist)
    print(datanum, "data loading\n")  
    inputs = np.zeros((datanum, 1, 256, image_size), dtype=np.float16)
    if task == "event":
        inputs_phase = 1
    else:    
        inputs_phase = np.zeros((datanum, 512, image_size), dtype=np.float16)
        
    if task == "event":
        labels = np.zeros((datanum, 10, image_size), dtype=np.float16)
    elif task == "segmentation":
        labels = np.zeros((datanum, 10, 256, image_size), dtype=np.float16)
    
    for i, folder in enumerate(folderlist):
        if not folder[-4:] == ".wav" and not folder[-5:] == ".jams":
            data_dir = segdata_dir + folder + "/"
            filelist = os.listdir(data_dir) 
            for n in range(len(filelist)):
                if filelist[n][-4:] == ".wav":
                    waveform, fs = sf.read(data_dir + filelist[n]) 
                    _, _, stft = signal.stft(x=waveform, fs=fs, nperseg=512, return_onesided=False)
                    #stft = stft[:, 1:len(stft.T) - 1]
                    stft = stft[:, 1:image_size + 1]   #### 624 to 512
                    
                    if not eval_dir == "ls_0dB" and filelist[n][:10] == "background":
                        continue

                    elif  filelist[n][:10] == "foreground" or (eval_dir == "ls_0dB" and filelist[n][:10] == "background"):
                        if filelist[n][11] == "_":
                            cls_name = filelist[n][12:]
                        elif filelist[n][12] == "_":
                            cls_name = filelist[n][13:]
                        cls_name = cls_name[:-4]

                        if cls_name == "Frying_nOn_nOff" or cls_name == "Frying_nOff":
                            cls_name ="Frying"
                        elif cls_name == "Vacuum_cleaner_nOff" or cls_name == "Vacuum_cleaner_nOn" or cls_name == "Vacuum_cleaner_nOn_nOff":
                            cls_name ="Vacuum_cleaner"
                        elif cls_name == "Blender_nOff":
                            cls_name ="Blender"
                        elif cls_name == "Running_water_nOn_nOff" or cls_name == "Running_water_nOn":
                            cls_name ="Running_water"

                        if task == "event":                             # SED
                            labels[i][label.T[cls_name][0]] += abs(stft[:256]).max(0)
                        elif task == "segmentation":                    # segmentation
                            labels[i][label.T[cls_name][0]] += abs(stft[:256]) 

                    elif len(filelist[n]) < 9:
                        if not task == "event":
                            inputs_phase[i] = np.angle(stft)                   
                        inputs[i][0] = abs(stft[:256])         

    inputs, labels = log(inputs, labels)   
    inputs, labels, max = normalize(inputs, labels)

    inputs = inputs.transpose(0, 2, 3, 1)
    if task == "event":                                 # SED
        labels = ((labels > 0.1) * 1)
        labels = labels.transpose(0, 2, 1)[:,np.newaxis,:,:]
    elif task == "segmentation":                        # segmentation
        labels = labels.transpose(0, 2, 3, 1) 
        
    return inputs, labels, max, inputs_phase



def train(X_train, Y_train, Model):
    model, multi_model = read_model.read_model(Model, gpu_count=gpu_count, classes=classes, image_size=image_size, 
                                               channel=channel, ang_reso=ang_reso, aux=aux, sed_model=sed_model, trainable=trainable)
    
    if aux == False:
        multi_model.compile(loss=loss, optimizer=Adam(lr=lr),metrics=["accuracy"])
    else:
        multi_model.compile(loss=["binary_crossentropy", "mean_squared_error"],
                            loss_weights=[0.01, 1.0], optimizer=Adam(lr=lr),metrics=["accuracy"])
        
    plot_model(model, to_file = results_dir + model_name + '.png')
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1,mode="auto")
#    tensorboard = TensorBoard(log_dir=results_dir, histogram_freq=0, write_graph=True)

    if task == "segmentation":
        if complex_input == True  or mic_num > 1 or ipd == True:
            X_train = [X_train, 
                       X_train.transpose(3,0,1,2)[0][np.newaxis,:,:,:].transpose(1,2,3,0)]
        
        if Model == "aux_Mask_UNet" or Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
            Y_train = [((Y_train.transpose(3,0,1,2).max(2)[:,:,np.newaxis,:] > 0.1) * 1).transpose(1,2,3,0), 
                       Y_train]

    history = multi_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, validation_split=0.1, callbacks=[early_stopping])

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
    model, _ = read_model.read_model(Model, gpu_count=gpu_count, classes=classes, image_size=image_size, 
                                               channel=channel, ang_reso=ang_reso, aux=aux, sed_model=sed_model, trainable=trainable)
    model.load_weights(results_dir + model_name + '_weights.hdf5')
    print(utils.get_flops())

    if task == "segmentation":
        if complex_input == True or mic_num > 1:
            X_test = [X_test, 
                      X_test.transpose(3,0,1,2)[0][np.newaxis,:,:,:].transpose(1,2,3,0)]
    Y_pred = model.predict(X_test, BATCH_SIZE * gpu_count)
    
    return Y_pred



def restore(Y_true, Y_pred, max, phase, no=0, class_n=1, save=False, calc_sdr=False):
    plot_num = classes * ang_reso
    if save:
        pred_dir = utils.pred_dir_make(no, results_dir)
    
    Y_pred = Y_pred.transpose(0, 3, 1, 2)
    Y_true = Y_true.transpose(0, 3, 1, 2)    
    
    data_dir = valdata_dir + str(no)
    sdr_array = np.zeros((plot_num, 1))
    sir_array = np.zeros((plot_num, 1))
    sar_array = np.zeros((plot_num, 1))
    
    wavefile = glob.glob(data_dir + '/0__*.wav')
    X_wave, _ = sf.read(wavefile[0])

    for index_num in range(plot_num):
        if Y_true[no][index_num].max() > 0:
            Y_linear = 10 ** ((Y_pred[no][index_num] * max - 120) / 20)
            Y_linear = np.vstack((Y_linear, Y_linear[::-1]))

            Y_complex = np.zeros((512, image_size), dtype=np.complex128)
            for i in range (512):
                for j in range (image_size):
                    Y_complex[i][j] = cmath.rect(Y_linear[i][j], phase[no][i][j])

            if ang_reso == 1:
                filename = label.index[index_num]+"_prediction.wav"
            else:
                filename = label.index[index_num // ang_reso] + "_" + str((360 // ang_reso) * (index_num % ang_reso)) + "deg_prediction.wav"
                
            _, Y_pred_wave = signal.istft(Zxx=Y_complex, fs=16000, nperseg=512, input_onesided=False)
            Y_pred_wave = Y_pred_wave.real
            if save:
                sf.write(pred_dir + "/" + filename, Y_pred_wave.real, 16000, subtype="PCM_16")

            if calc_sdr:
                # calculate SDR
                if classes == 1:
                    with open(os.path.join(data_dir, "sound_direction.txt"), "r") as f:
                        directions = f.read().split("\n")[:-1]
                    for direction in directions:
                        if index_num == int(re.sub("\\D", "", direction.split("_")[1])) // (360 // ang_reso):
                            class_name = direction.split("_")[0]
                            Y_true_wave, _ = sf.read(data_dir + "/" + class_name + ".wav")
                else:                
                    Y_true_wave, _ = sf.read(data_dir + "/" + label.index[index_num // ang_reso] + ".wav")
                
                Y_true_wave = Y_true_wave[:len(Y_pred_wave)]
                X_wave = X_wave[:len(Y_pred_wave)]

                sdr_base, sir_base, sar_base, _ = bss_eval_sources(Y_true_wave[np.newaxis,:], X_wave[np.newaxis,:], compute_permutation=False)
                sdr, sir, sar, _ = bss_eval_sources(Y_true_wave[np.newaxis,:], Y_pred_wave[np.newaxis,:], compute_permutation=False)
                #print("No.", no, "Class", index_num // ang_reso, label.index[index_num // ang_reso], "SDR", round(sdr[0], 2), "SDR_Base", round(sdr_base[0], 2), "SDR improvement: ", round(sdr[0] - sdr_base[0], 2))
                
                sdr_array[index_num] = sdr
                sir_array[index_num] = sir
                sar_array[index_num] = sar
            
    return sdr_array, sir_array, sar_array
    


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
        sed_model = CNN.CNN(n_classes=classes, input_height=256, input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], RNN=0, Bidir=False)
        sed_model.load_weights(os.getcwd()+"/model_results/2020_0530/CNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_"+datadir+"/CNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")
    elif Model == "BiCRNN8":
        sed_model = CNN.CNN(n_classes=classes, input_height=256, input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], RNN=2, Bidir=True)
        sed_model.load_weights(os.getcwd()+"/model_results/2020_0530/BiCRNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_"+datadir+"/BiCRNN8_75class_1direction_1ch_cinFalse_ipdFalse_vonMisesFalse_weights.hdf5")

    return sed_model


def load_ssl_model(Model):
    if Model == "CNN8":
        ssl_model = CNN.CNN(n_classes=ang_reso, input_height=256, input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], RNN=0, Bidir=False)
        ssl_model.load_weights(os.getcwd()+"/model_results/nextjournal/SSL_CNN8_1class_72direction_8ch_cinTrue_ipdTrue_vonMisesFalse_multi_segdata75_256_no_sound_random_sep_72/SSL_CNN8_1class_72direction_8ch_cinTrue_ipdTrue_vonMisesFalse_weights.hdf5")
    elif Model == "BiCRNN8":
        ssl_model = CNN.CNN(n_classes=ang_reso, input_height=256, input_width=image_size, nChannels=channel,
                            filter_list=[64, 64, 128, 128, 256, 256, 512, 512], RNN=2, Bidir=True)
        ssl_model.load_weights(os.getcwd()+"/model_results/nextjournal/SSL_BiCRNN8_1class_72direction_8ch_cinTrue_ipdTrue_vonMisesFalse_multi_segdata75_256_no_sound_random_sep_72/SSL_BiCRNN8_1class_72direction_8ch_cinTrue_ipdTrue_vonMisesFalse_weights.hdf5")
    
    return ssl_model


def load_cascade(segdata_dir, load_number=9999999):
    print("cascade data loading\n")
                
    inputs = np.zeros((1, 256, image_size), dtype=np.float16)
    sep_num = np.zeros((load_number), dtype=np.int16)
    
    for i in range(load_number):
        data_dir = segdata_dir + str(i) + "/"
        filelist = os.listdir(data_dir)  
                    
        for n in range(len(filelist)):
            if filelist[n][:4] == "sep_":
                waveform, fs = sf.read(data_dir + filelist[n]) 
                _, _, stft = signal.stft(x=waveform, fs=fs, nperseg=512, return_onesided=False)
                stft = stft[:, 1:len(stft.T) - 0]

                inputs = np.concatenate((inputs, abs(stft[:256])[np.newaxis, :, :]), axis=0)
                sep_num[i] += 1
                
    inputs = 20 * np.log10(inputs)
    inputs = np.nan_to_num(inputs) + 120
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
    classes = 75#10
    image_size = 256#512#624
    task = "segmentation"
    ang_reso = 1
    
    if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest' or os.getcwd() == '/home/sudou/python/sound_segtest':
        gpu_count = 1
    else:
        gpu_count = 1
    BATCH_SIZE = 16 * gpu_count
    NUM_EPOCH = 10
    lr = 0.001
    
    loss = "mean_squared_error"
    if task == "event":
        loss = "binary_crossentropy"

    mode = "train"
    date = mode       
    graph_num = 10

    if os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
        datasets_dir = "/home/yui-sudo/document/dataset/sound_segmentation/datasets/"
    elif os.getcwd() == '/home/sudou/python/sound_segtest':
        datasets_dir = '/media/sudou/d0e7ca7c-34a8-4983-945f-a0783e5a55c5/dataset/dataset/datasets/'
    else:
        datasets_dir = "/misc/export3/sudou/sound_data/datasets/"
    
    for datadir in ["multi_segdata"+str(classes) + "_"+str(image_size)+"_-20dB_random_sep_72/", 
                    #"dcase2019/dataset/audio/"
                    ]:
        dataset = datasets_dir + datadir    
        segdata_dir = dataset + "train/"
        valdata_dir = dataset + "val/"
        #segdata_dir = dataset + "train/synthetic/"
        
        labelfile = dataset + "label.csv"
        label = pd.read_csv(filepath_or_buffer=labelfile, sep=",", index_col=0)            
        
        for Model in [#"CNN8", "BiCRNN8", 
                      #"SELD_CNN8", #"SELD_BiCRNN8", 
                      "UNet", 
                      #"UNet", #"CR_UNet", #"Deeplab", 
                      #"aux_enc_UNet", "aux_enc_Deeplab", 
                      #"Cascade"
                      ]:
            
            if Model == "Cascade":
                loss = "categorical_crossentropy"

            vonMises = False
            for ipd in [False]:
                for mic_num in [1]: # 1 or 8                        
                    for complex_input in [False]:
                        channel = 0
                        if mic_num == 1:
                            channel = 1
                        else:                                
                            if complex_input:
                                if ipd:
                                    channel = 15
                                elif ipd == False:
                                    channel = 24
                                else:
                                    continue
                        
                        for Sed_Model in ["BiCRNN8"]:
                            if Model == "Mask_UNet":
                                sed_model = load_sed_model(Sed_Model)
                                mask=True
                                aux = False
                                trainable = False # SED mask
                            elif Model == "aux_Mask_UNet":
                                sed_model = load_sed_model(Sed_Model)
                                mask = True
                                aux = True
                                trainable = True # SED mask
                            elif Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
                                mask = False
                                aux = True
                            else:
                                sed_model = None
                                mask = False
                                aux = False
                                trainable = False
                                            
                            load_number = 10000
                            
                            model_name = Model+"_"+str(classes)+"class_"+str(ang_reso)+"direction_" + str(mic_num)+"ch_cin"+str(complex_input) + "_ipd"+str(ipd) + "_vonMises"+str(vonMises)
                            if mask:
                                model_name = model_name + "_"+Sed_Model + "_aux" + str(aux)
                            dir_name = model_name + "_"+datadir
                            date = datetime.datetime.today().strftime("%Y_%m%d")
                            results_dir = "./model_results/" + date + "/" + dir_name
                            
                            if mode == "train":
                                print("\nTraining start...")
                                if not os.path.exists(results_dir + "prediction"):
                                    os.makedirs(results_dir + "prediction/")

                                npy_name = "train_" + task + "_" +str(classes)+"class_"+str(ang_reso)+"direction_" + str(mic_num)+"ch_cin"+str(complex_input) + "_ipd"+str(ipd)  + "_vonMises"+str(vonMises) + "_"+str(load_number)
                                if not os.path.exists(dataset+"X_"+npy_name+".npy"):
                                    #X_train, Y_train, max, phase = load_desed(segdata_dir)
                                    X_train, Y_train, max, phase = load(segdata_dir, n_classes=classes, load_number=load_number, input_dim=channel)
                                    save_npy(X_train, Y_train, max, phase, npy_name)
                                else:
                                    X_train, Y_train, max, phase = load_npy(npy_name)
                                
                                if Model == "Cascade":
                                    X_train, Y_train = Segtoclsdata(Y_train)
                                
                                # save train condition
                                train_condition = date + "\t" + results_dir                     + "\n" + \
                                                    "\t"+" "                          + "\n" + \
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
                                                    "\t\t task     ,      " + task + "\n" + \
                                                    "\t\t Angle reso,     " + str(360 // ang_reso) + "\n" + \
                                                    "\t\t Model,          " + Model               + "\n" + \
                                                    "\t\t classes,        " + str(classes)        + "\n\n\n"
                                print(train_condition)
                                
                                with open(results_dir + 'train_condition.txt','w') as f:
                                    f.write(train_condition)
                                
                                history = train(X_train, Y_train, Model)
                                plot_history(history, model_name)

                    
                            # prediction mode         
                            elif not mode == "train":
                                print("Prediction mode\n")
                                date = mode
                                results_dir = "./model_results/" + date + "/" + dir_name
                                with open(results_dir + 'train_condition.txt','r') as f:
                                    train_condition = f.read() 
                                    print(train_condition)

                            #for eval_dir in ["500ms", "fbsnr_0dB", "ls_0dB"]: 
#                                for eval_dir in ["500ms", "5500ms", "9500ms", "fbsnr_0dB", "fbsnr_15dB", "fbsnr_24dB", "fbsnr_30dB", "ls_0dB", "ls_15dB", "ls_30dB"]: 
                            #    valdata_dir = dataset + "eval/" + eval_dir + "/"

                            load_number = 1000
                                
                            npy_name = "test_" + "_" + task+ "_" +str(classes)+"class_"+str(ang_reso)+"direction_" + str(mic_num)+"ch_cin"+str(complex_input) + "_ipd"+str(ipd)  + "_vonMises"+str(vonMises) + "_"+str(load_number)
                            if not os.path.exists(dataset+"X_"+npy_name+".npy"):
                                #X_test, Y_test, max, phase = load_desed(valdata_dir)
                                X_test, Y_test, max, phase = load(valdata_dir, n_classes=classes, load_number=load_number, input_dim=channel)
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
                            
                            if Model == "aux_Mask_UNet" or Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
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
                                
                            # plot and SDR
                            sdr_array, sir_array, sar_array, sdr_num = np.zeros((classes, 1)), np.zeros((classes, 1)), np.zeros((classes, 1)), np.zeros((classes, 1))             
                            for i in range (0, load_number):
                            #folderlist = os.listdir(segdata_dir)
                            #for i, folder in enumerate(folderlist):
                                save = False
                                if i < graph_num:
                                    save = True
                                    utils.origin_stft(X_test, no=i, results_dir=results_dir)
                                    if task == "event":
                                        utils.event_plot(Y_test, Y_pred, no=i, results_dir=results_dir, image_size=image_size, ang_reso=ang_reso, classes=classes, label=label)
                                    else:
                                        utils.plot_stft(Y_test, Y_pred, no=i, results_dir=results_dir, image_size=image_size, ang_reso=ang_reso, classes=classes, label=label)
                                        if Model == "aux_Mask_UNet" or Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
                                            utils.event_plot(Y_sedt, Y_sedp, no=i, results_dir=results_dir, image_size=image_size, ang_reso=ang_reso, classes=classes, label=label)

                                calc_sdr = False
                                if save == True or calc_sdr == True:
                                    sdr, sir, sar = restore(Y_test, Y_pred, max, phase, no=i, save=save, calc_sdr=calc_sdr)
                                    sdr_array += sdr
                                    sir_array += sir
                                    sar_array += sar
                                    sdr_num += (sdr != 0.000) * 1                            

                            if calc_sdr:    
                                sdr_array = sdr_array / sdr_num
                                sir_array = sir_array / sdr_num
                                sar_array = sar_array / sdr_num
                                sdr_array = np.append(sdr_array, sdr_array.mean())
                                sir_array = np.append(sir_array, sir_array.mean())
                                sar_array = np.append(sar_array, sar_array.mean())
                                
                                np.savetxt(results_dir+"prediction/sdr_"+str(load_number)+".csv", sdr_array, fmt ='%.3f')
                                print("SDR\n", sdr_array, "\n")   

                            # Metrics
                            if task == "event":
                                Y_pred = (Y_pred > 0.5) * 1
                                f1 = f1_score(Y_test.ravel(), Y_pred.ravel())
                                Y_pred = np.argmax(Y_pred, axis=3)
                                print("F_score", f1)
                                with open(results_dir + "f1_" + str(f1) + ".txt","w") as f:
                                    f.write(str(f1))   
                                    
                            elif task == "segmentation":
                                utils.RMS(Y_test, Y_pred, results_dir=results_dir, classes=classes, max=max) 
                                if Model == "aux_Mask_UNet" or Model == "aux_enc_UNet" or Model == "aux_enc_Deeplab":
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
                                
                            shutil.copy("main.py", results_dir)
                            shutil.copy("Unet.py", results_dir)
                            shutil.copy("Deeplab.py", results_dir)
                            shutil.copy("CNN.py", results_dir)    
                            if os.path.exists(os.getcwd() + "/nohup.out"):
                                shutil.copy("nohup.out", results_dir)
