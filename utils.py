import os
import glob
import cmath
import numpy as np
from scipy import signal
import pandas as pd
import soundfile as sf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mir_eval.separation import bss_eval_sources

import keras.backend as K
import tensorflow as tf


def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.



def pred_dir_make(no, results_dir):
    if not os.path.exists(results_dir + "prediction/" + str(no)):
        os.mkdir(results_dir + "prediction/" + str(no))
    pred_dir = results_dir + "prediction/" + str(no) + "/"
    
    return pred_dir


def origin_stft(X, no, results_dir):
    pred_dir = pred_dir_make(no, results_dir)
    X = X.transpose(0, 3, 1, 2)    

    plt.title(str(no) + "__mixture")
    plt.pcolormesh((X[no][0]))
    plt.xlabel("time")
    plt.ylabel('frequency')
    plt.clim(0, 1)
    plt.colorbar()
    plt.savefig(pred_dir + "mixture.png")
    plt.close()


def event_plot(Y_true, Y_pred, no, results_dir, image_size, ang_reso, classes, label):
    pred_dir = pred_dir_make(no, results_dir)

    if ang_reso == 1 or classes == 1:
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
            if Y_true[no][i].max() > 0:
                
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
        
        plt.pcolormesh((Y_true_total), cmap="gist_ncar")
        plt.title(str(no) + "__color_truth")
        plt.xlabel("time")
        plt.ylabel('angle')
        plt.clim(0, Y_true_total.max())
        plt.savefig(pred_dir + "color_truth.png")
        plt.close()
    
        plt.pcolormesh((Y_pred_total), cmap="gist_ncar")
        plt.title(str(no) + "__color_prediction")
        plt.xlabel("time")
        plt.ylabel('angle')
        plt.clim(0, Y_true_total.max())
        plt.savefig(pred_dir + "color_pred.png")
        plt.close()


def plot_stft(Y_true, Y_pred, no, results_dir, image_size, ang_reso, classes, label):
    plot_num = classes * ang_reso
    if ang_reso > 1:
        ylabel = "angle"
    else:
        ylabel = "frequency"
    pred_dir = pred_dir_make(no, results_dir)

    Y_pred = Y_pred.transpose(0, 3, 1, 2)
    Y_true = Y_true.transpose(0, 3, 1, 2)
        
    Y_true_total = np.zeros((256, image_size))
    Y_pred_total = np.zeros((256, image_size))
    for i in range(plot_num):
        if Y_true[no][i].max() > 0:
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
            elif ang_reso > 1 and classes == 1:
                plt.savefig(pred_dir + str((360 // ang_reso) * (i % ang_reso)) + "deg_true.png")
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
    
    plt.pcolormesh((Y_true_total), cmap="gist_ncar")
    plt.title(str(no) + "__color_truth")
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.clim(0, Y_true_total.max())
    plt.savefig(pred_dir + "color_truth.png")
    plt.close()

    plt.pcolormesh((Y_pred_total), cmap="gist_ncar")
    plt.title(str(no) + "__color_prediction")
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.clim(0, Y_true_total.max())
    plt.savefig(pred_dir + "color_pred.png")
    plt.close()

    

def RMS(Y_true, Y_pred, results_dir, classes, max_magnitude):
    Y_pred = Y_pred.transpose(0, 3, 1, 2) # (data number, class, freq, time)
    Y_true = Y_true.transpose(0, 3, 1, 2)
        
    Y_pred_db = (Y_pred * max_magnitude) # 0~120dB
    Y_true_db = (Y_true * max_magnitude)
    
    total_rmse = np.sqrt(((Y_true_db - Y_pred_db) ** 2).mean())
    print("Total RMSE =", total_rmse)

    rms_array = np.zeros(classes + 1)
    num_array = np.zeros(classes + 1) # the number of data of each class

    area_array = np.zeros(classes + 1) # area size of each class
    duration_array = np.zeros(classes + 1) # duration size of each class
    freq_array = np.zeros(classes + 1) # frequency conponents size of each class
    
    spl_array =  np.zeros(classes + 1) #average SPL of each class 
    percent_array =  np.zeros(classes + 1) 

    for no in range(len(Y_true)):
        for class_n in range(classes):
            if Y_true[no][class_n].max() > 0: # calculate RMS of active event class               
                num_array[classes] += 1 # total number of all classes
                num_array[class_n] += 1 # the number of data of each class
                on_detect = Y_true[no][class_n].max(0) > 0.0 # active section of this spectrogram image
                
                per_rms = ((Y_true_db[no][class_n] - Y_pred_db[no][class_n]) ** 2).mean(0) # mean squared error about freq axis of this spectrogram
                rms_array[class_n] += per_rms.sum() / on_detect.sum() # mean squared error of one data
                
                per_spl = Y_true_db[no][class_n].mean(0) # mean spl about freq axis
                spl_array[class_n] += per_spl.sum() / on_detect.sum() # mean spl of one data
                
                area_array[class_n] += ((Y_true[no][class_n] > 0.1) * 1).sum() # number of active bins = area size
                duration_array[class_n] += ((Y_true[no][class_n].max(0) > 0.1) * 1).sum() # duration bins
                freq_array[class_n] += ((Y_true[no][class_n].max(1) > 0.1) * 1).sum() # duration bins
                
    rms_array[classes] = rms_array.sum()
    rms_array = np.sqrt(rms_array / num_array) # Squared error is divided by the number of data = MSE then root = RMSE

    spl_array[classes] = spl_array.sum()
    spl_array = spl_array / num_array # Sum of each spl is divided by the number of data = average spl

    area_array[classes] = area_array.sum()
    area_array = area_array // num_array # average area size of each class

    duration_array[classes] = duration_array.sum()
    duration_array = duration_array // num_array # average duration size of each class
    duration_array = duration_array * 16.0

    percent_array = rms_array / spl_array * 100 


    print("rms\n", rms_array, "\n")     
    all_array = np.vstack([num_array, area_array, duration_array, rms_array, percent_array, spl_array])
    np.savetxt(results_dir+"prediction/" + "total_score"+str(len(Y_true))+".csv", all_array.T, fmt ='%.3f')

