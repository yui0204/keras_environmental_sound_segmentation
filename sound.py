# -*- coding: utf-8 -*-
"""このファイルはwavファイルを読み込むためのものです"""

import shutil
import h5py
import wave
#import struct
from struct import unpack
from numpy import frombuffer

import numpy as np
import pandas as pd
import librosa
import librosa.display

from scipy import signal
#from scipy.fftpack import fft, ifft
from scipy import stats

from sklearn.decomposition import NMF

import matplotlib.pyplot as plt

import os, os.path

class WavfileOperate:
    """このクラスはwavファイルを読み込むためのものです"""
    def __init__(self, filepath, uselib='soundfile', logger = 1):
        """
        コンストラクタ
        
        :param filepath: wavファイルパス
        :param uselib: wavファイル読み込みに使用するライブラリ(デフォルト：soundfile)
        """
        
        self.__path = filepath
        self.__nframe = 0
        self.__nchan = 0
        self.__nbyte = 0
        self.__fs = 0

        if uselib == 'soundfile':
            self.__read_wav_soundfile(logger)

        elif uselib == 'wave':
            self.__read_wav()
        else:
            self.__read_wav()

    @property    
    def path(self):
        """読み込んだwavファイルのファイルパス"""
        return self.__path
    
    @property    
    def nframe(self):
        """フレーム数"""
        return self.__nframe
    
    @property    
    def nchan(self):
        """チャンネル数"""
        return self.__nchan
    
    @property    
    def nbyte(self):
        """1点あたりのバイト数"""
        return self.__nbyte
    
    @property    
    def fs(self):
        """サンプリングレート"""
        return self.__fs
    
    @property    
    def norm_sound(self):
        """-1~1に正規化された時系列データ"""
        return self.wavedata.norm_sound

    def __read_wav_soundfile(self, logger):
        """
        このメソッドは16bitか24bitかを判別し、それぞれに応じた読み込みメソッドを呼び出す
        ※soundfileモジュールを使用する
        """
        import soundfile as sf

        local_data, samplerate = sf.read(self.__path)   # ファイル読み込み
        local_data = local_data * logger # ロガーのレンジに応じて係数をかける
        self.__fs = samplerate         # サンプリングレート
        # チャンネル数
        if local_data.ndim == 1:
            self.__nchan = 1
        else:
            self.__nchan = local_data.shape[1]

        self.__nframe = local_data.shape[0]     # フレーム数

        if self.__nchan == 1:
            self.wavedata = Wavedata(self.__fs, local_data,
                                 self.__path, self.__nbyte, False)
            
        # マルチチャンネルwavの場合、Wavedataクラスのリストを出力
        elif self.__nchan > 1:
            wavedata_list = []  
            for i in range(self.__nchan):
                wavedata_list.append(Wavedata(self.__fs, local_data.T[i], 
                                          self.__path, self.__nbyte, False))
            self.wavedata = wavedata_list
            self.multiwave = Multiwave(wavedata_list)
            

    def __read_wav(self):
        """このメソッドは16bitか24bitかを判別し、それぞれに応じた読み込みメソッドを呼び出す"""
        with wave.open(self.__path, "rb") as fp:
            self.__nframe = fp.getnframes()       # フレーム数
            self.__nchan = fp.getnchannels()      # チャンネル数
            self.__nbyte = fp.getsampwidth()      # バイト数
            self.__fs = fp.getframerate()         # サンプリングレート
            buf = fp.readframes(self.__nframe * self.__nchan) # バッファ
            
        # 16bitか24bitか判定し、読み込みメソッドを実行
        if self.__nchan == 1:
            # 24bitファイルの場合                    
            if self.__nbyte == 3:            
                self.wavedata = Wavedata(self.__fs, self.__read_24bit(buf), 
                                         self.__path, self.__nbyte, True)
            # 16bitファイルの場合    
            elif self.__nbyte == 2:               
                self.wavedata = Wavedata(self.__fs, 
                                        frombuffer(buf, dtype="int16"), 
                                        self.__path, self.__nbyte, True)
            else:
                print("16bitでも24bitでもありません")
        
        # ステレオ、多チャンネルファイルの場合、リストにして書き出す
        elif self.__nchan > 1:
            wavedata_list = []          
            # 24bitファイルの場合                    
            if self.__nbyte == 3:               
                for i in range(self.__nchan):
                    wavedata_list.append(Wavedata(self.__fs, 
                                              self.__read_24bit(buf), 
                                              self.__path + "_ch" + str(i+1), 
                                              self.__nbyte, True))
                self.wavedata = wavedata_list
                self.multiwave = Multiwave(wavedata_list)
                
            # 16bitファイルの場合    
            elif self.__nbyte == 2:               
                for i in range(self.__nchan):
                    wavedata_list.append(Wavedata(self.__fs, 
                              frombuffer(buf, dtype="int16")[i::self.__nchan], 
                              self.__path + "_ch" + str(i+1), 
                              self.__nbyte, True))
   
                self.wavedata =wavedata_list
                self.multiwave = Multiwave(wavedata_list)
                
            else:
                print("16bitでも24bitでもありません")
            

    def __read_24bit(self, buf):
        """
        このメソッドは24bitwavファイルの情報を読み込み、音声データを格納するためのものです

        :param buf: 時系列データのバッファ
        :return : 24bitの時系列intデータ
        """

        read_sec = int(self.__nframe / self.__fs)# 読み込み秒数を整数に変換
        self.__nframe = read_sec * self.__fs      # 整数返還後のフレーム数に上書き
        read_sample = read_sec *self.__nchan *self.__fs
        
        unpacked_buf = [unpack("<i", bytearray([0]) + buf[self.__nbyte * i 
                        : self.__nbyte * (i + 1)])[0]
                        for i in range(read_sample)]
        ndarr_buf = np.array(unpacked_buf)
        local_data = np.where(ndarr_buf > 0, ndarr_buf, ndarr_buf)
        
        return local_data
        
        
    def print_wav_info(self):
        """このメソッドはread_wavメソッドで読み込んだ情報を表示するためのものです"""
        print("filepath =", self.__path)
        print("nframe =", self.__nframe)
        print("nchan =", self.__nchan)
        print("nbyte =", self.__nbyte, "(" + str(self.__nbyte * 8) + "bit)")
        print("fs =", self.__fs)

    
class Wavedata:
    """このクラスは読み込んだ時系列データを解析するためのものです"""
    def __init__(self, fs, sound_data, name, nbyte, norm = False, logger = 3.16):
        """
        コンストラクタ
        
        :param fs: サンプリングレート　Hz
        :param sound_data: 時系列データ
        :param name: この時系列データの名前
        :param norm: 正規化を行うかどうかのブール
        :param logger: データロガーのレンジ（デフォルトは3.16、norm=Trueのときのみ実行）
        """
        
        self.__fs = fs
        self.__sound = sound_data
        self.__name = name
        self.__nbyte = nbyte
        self.__norm_sound = sound_data
        self.__time = np.arange(np.alen(self.__sound)) / self.__fs
        
        if(norm == True):
            self.__normalize(logger) # ロガーのレンジを入力
            
    @property    
    def name(self):
        """時系列データの名前"""
        return self.__name
    @name.setter
    def name(self, name):
        self.__name = name

    """
    @property    
    def sound(self):
        return self.__sound
    """
    
    @property    
    def norm_sound(self):
        """-1~1に正規化された時系列データ"""
        return self.__norm_sound
    
    @property    
    def fs(self):
        """サンプリングレート"""
        return self.__fs
    
    @property    
    def nbyte(self):
        """1点あたりのバイト数"""
        return self.__nbyte
    
    @property    
    def time(self):
        """時間 [s]"""
        return self.__time
            
    
    def __normalize(self, logger):
        """このメソッドはロガーの変換係数を考慮し、Paに変換します"""
        
        # 16bitの場合
        if self.__nbyte == 2:
            self.__norm_sound = self.__sound / (2 ** 15 - 1) * logger
        
        # 24bitの場合
        elif self.__nbyte == 3:
            self.__norm_sound = self.__sound / (2 ** 31 - 1) * logger

            
    def plot(self, ymin = -1, ymax = 1):
        """このメソッドは時系列データをグラフ表示するためのものです"""
        
        plt.figure(figsize=(12,3))
        plt.plot(self.__time, self.__norm_sound)
        plt.title(self.__name)
        plt.ylim(ymin, ymax)
        plt.xlabel("Time [s]")
        plt.ylabel("Sound pressure [Pa]")
        plt.show()
        plt.close()
        
                
    def stft_plot(self, N = 1024, cmin = -100, cmax = -80):
        """このメソッドはSTFTカラーマップをグラフ表示のみ行います)
        
        :param N: STFTフレームサイズ（通常2048）
        """
        
        plt.figure(figsize=(15,3))
        pxx, freqs, bins, im = plt.specgram(self.__norm_sound,
                                            NFFT=N, Fs=self.__fs, 
                                            noverlap=N/2, window=np.hanning(N))
        plt.title(self.__name + "STFT")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency[Hz]")
        #plt.ylim(0,20000)
        #plt.clim(cmin, cmax)
        plt.colorbar()
        plt.show()
        plt.close()
        # STFT結果を時間平均し、FFTグラフを表示
        #plt.figure(figsize=(15,3))
        plt.plot(freqs, 10 * np.log10(pxx.mean(1)))
        plt.title(self.__name + "FFT")
        plt.xlabel("frequency [Hz]")
        plt.ylabel("spectrum[dB]")
        plt.xlim(0, self.__fs // 2)
        #plt.xlim(0,20000)
        #plt.ylim(cmin - 20 ,cmax + 20)
        plt.grid(True)
        plt.show()
        plt.close()
        
 
    def fft(self, N = 2048, bins = 1024):
        """このメソッドは平均のFFTを算出し、特徴量データフレームに出力します
        
        :param N: STFTのフレーム幅(通常、20~40ms程度が良いらしい。が調検では2048を使用)
        :param n: 平均のFFTを何個のbinに束ねるか(デフォルト1024:束ねない)
        :return 平均のFFTデータフレーム
        """
        
        fft_data = abs(self.scipy_stft(N = N).stft[0:N//2]).mean(1)
        fft_data = 20 * np.log10(fft_data)
        
        # バンド幅を束ねる(n個おきのFFTデータ取り出し、平均）
        n = (N // 2) // bins
        integrated_fft = fft_data[::n]
        for j in range(1, n):
            integrated_fft += fft_data[j::n]
        integrated_fft = integrated_fft / n
        
        # FFT_dataをデータフレームに書き込み
        fft_df = pd.DataFrame(integrated_fft, columns = [self.__name])                  
        
        # indexに特徴量を書き込む
        index_list = []
        for i in range(0, N//2, n):
            index_list.append(str(int(self.__fs/N*i)) + "Hz_FFT")
        fft_df.index = index_list
        
        return fft_df.T
    
    
    def cepstrum(self, N = 2048, plot=False):
        """このメソッドは平均のケプストラムを算出し、特徴量データフレームに出力します
        
        :param N: STFTのフレーム幅(通常、20~40ms程度が良いらしい。が調検では2048を使用)
        :return 平均のケプストラムデータフレーム
        """
        
        fft_data = abs(self.scipy_stft(N = N).stft).mean(1)
        fft_data = 20 * np.log10(fft_data)
        cepstrum = np.real(np.fft.ifft(fft_data))[:N//2]
        cepstrum_df = pd.DataFrame(cepstrum, columns = [self.__name])
        
        index_list = []
        for i in range(N//2):
            index_list.append(str(i) + "th_Cepstrum")                   
        cepstrum_df.index = index_list
        
        # STFTをケプストラム分析
        fft_data = abs(self.scipy_stft(N = N).stft)
        fft_data = 10 * np.log10(fft_data)
        st_cepstrum = np.real(np.fft.ifft(fft_data, axis=0))
        
        if plot == True:
            # グラフ表示
            plt.figure(figsize=(15,3))
            time = np.array(range(0, len(st_cepstrum.T))) * (N / self.__fs) / 2
            plt.pcolormesh(time, range(N), st_cepstrum)
            plt.title(self.__name + "Cepstrum")
            plt.clim(-1, 1)
            plt.colorbar()
            plt.show()
            
            #cepstrum = st_cepstrum.mean(1)[:N//2]   
            #cepstrum_df = pd.DataFrame(cepstrum, columns = [self.__name])
            
            plt.figure(figsize=(10,3))
            plt.plot(cepstrum)
            plt.title(self.__name + "Cepstrum")
            plt.xlabel("quefrency")
            plt.ylabel("cepstrum")
            plt.xlim(0, N // 2)
            #plt.ylim(-1, 4)
            plt.show()
        
        return cepstrum_df.T
                
    def melspectrogram(self, N = 2048, n_mels = 40, plot=False, 
                       cmin=-70, cmax=-50):
        """このメソッドはメルスペクトログラムを算出し、特徴量データフレームに出力します)
        
        :param N: STFTフレームサイズ（通常2048）
        :param n_mels: N個のSTFTデータを何次元のメルフィルタバンクに束ねるか
        :param plot: メルスペクトログラムのグラフ表示有無
        :return : メルスペクトログラム特徴量データフレーム）
        """
        
        index_list = [] # データフレームのインデックス名を格納するリスト
        
        melspectrogram = librosa.feature.melspectrogram(y=self.__norm_sound, 
                                                        sr=self.__fs,
                                                        n_fft = N, 
                                                        hop_length = N // 2, 
                                                        n_mels = n_mels)

        melspectrogram = 10 * np.log10(melspectrogram)
        ave_melspectrogram = melspectrogram.mean(1)                

        melspectrogram_df = pd.DataFrame(ave_melspectrogram, 
                                         columns = [self.__name])
        
        # インデックス名をリストに格納
        mels_list = librosa.core.mel_frequencies(n_mels=n_mels, fmax=self.__fs /2)   
        for i in range(n_mels):
            index_list.append(str(int(mels_list[i])) + "Hz_mel_FFT")
        melspectrogram_df.index = index_list

        if plot == True:
            plt.figure(figsize=(15,3))
            librosa.display.specshow(melspectrogram, sr=self.__fs, 
                                     hop_length = N // 2, 
                                     x_axis="time", y_axis="mel")
            plt.title(self.__name + "Mel_Spectrogram")
            plt.clim(cmin, cmax)
            plt.colorbar()
            plt.show()
        
            # STFT結果を時間平均し、FFTグラフを表示
            
            plt.figure(figsize=(10,3))
            plt.plot(mels_list, ave_melspectrogram)
            plt.title(self.__name + "Mel_Spectrogram")
            plt.xlabel("frequency [Hz]")
            plt.ylabel("spectrum[dB]")
            plt.xlim(0, self.__fs // 2)
            plt.ylim(cmin, cmax)
            plt.show()
            
        return melspectrogram_df.T
    
    
    def mfcc(self, N=2048, n_mels=40, n_mfcc=40, delta_k=0, 
                     plot=False, cmin=-50, cmax=50):
        """このメソッドはMFCCを算出し、特徴量データフレームに出力します
        
        :param N: STFTフレームサイズ（通常2048）
        :param n_mels: N個のSTFTデータを何次元のメルフィルタバンクに束ねるか
        :param n_mfcc: n_melsに束ねられたメルスペクトログラムを何次元のMFCCデータにするか
        :param delta_k: MFCCのデルタに用いる隣接するMFCC数（2~5）、0でデルタ未使用
        :param plot: MFCCのグラフ表示有無
        :return : MFCC特徴量データフレーム
        """
        
        # librosaを用いたMFCC算出
        melspectrogram = librosa.feature.melspectrogram(y=self.__norm_sound, 
                                                        sr=self.__fs,
                                                        n_fft = N, 
                                                        hop_length = N//2, 
                                                        n_mels = n_mels)    
        melspectrogram = 10 * np.log10(melspectrogram)
        mfccs = librosa.feature.mfcc(y=self.__norm_sound, sr=self.__fs, 
                                     S = melspectrogram, n_mfcc=n_mfcc)
        
        ave_mfcc = mfccs.mean(1)
        mfcc_df = pd.DataFrame(ave_mfcc, columns = [self.__name])
        
        index_list = []
        for i in range(n_mfcc):
            index_list.append(str(i+1) + "th_MFCC_")               
        mfcc_df.index = index_list
 
       
        # 1次デルタ算出
        if delta_k > 0:
            from scipy.ndimage.interpolation import shift
            
            numer = 0
            denom = 0
            for k in range(-delta_k, delta_k + 1):
                numer += k * shift(mfccs, [0,k] , cval=0)
                denom += k ** 2
            delta = numer / denom
            # デルタは平均すると0になると思われるため、絶対値を特徴量として算出
            delta_ave = abs(delta[:, k:len(delta.T) - k - 1]).mean(1)
            delta_df = pd.DataFrame(delta_ave, columns = [self.__name])
            
            if plot == True:
                # グラフ表示
                plt.figure(figsize=(15,3))
                time = np.array(range(0, len(delta.T))) * (N / self.__fs) / 2
                plt.pcolormesh(time, range(n_mfcc), delta)
                plt.title(self.__name + "MFCC_delta")
                plt.clim(0, 3)
                plt.colorbar()
                plt.show()
                
                plt.figure(figsize=(10,3))
                plt.plot(delta_ave)
                plt.title(self.__name + "MFCC_delta")
                plt.xlabel("MFCC")
                plt.ylabel("MFCC_delta")
                #plt.ylim(-10, 10)
                plt.show()
            
            index_list = []
            for i in range(n_mfcc):
                index_list.append(str(i+1) + "th_MFCC_delta")
            
            delta_df.index = index_list
            
            mfcc_df = mfcc_df.append(delta_df)
        
        
        if plot == True:
            plt.figure(figsize=(12, 3))
            librosa.display.specshow(mfccs, sr=self.__fs, hop_length = N // 2, 
                                     x_axis="time")
            plt.clim(cmin, cmax)
            plt.colorbar()
            plt.title(self.__name + "_MFCC")
            plt.show()
                
            # MFCC結果を時間平均し、グラフを表示
            plt.figure(figsize=(12, 4))
            plt.plot(ave_mfcc)
            plt.title(self.__name + "_MFCC")
            plt.xlabel("quefrency")
            #plt.ylabel("spectrum[dB]")
            plt.ylim(cmin, cmax)
            plt.show()       
        
        return mfcc_df.T   
    
            
    def down_sample(self, down_name = "_down_sampling", down_n = 2):
        """
        このメソッドは時系列データをデシメーションフィルリングした後、1/2倍のダウンサンプリングします
        :param down_name: ﾀﾞｳﾝｻﾝﾌﾟﾘﾝｸﾞ後の時系列データにつける名前
        :param down_n: ダウンサンプリング割合（整数） self.__fs/down_nにダウンサンプリング
        :return down_wavedata: ダウンサンプリングされたWavedata
        """
        
        down_name = self.__name + down_name
        down_fs = self.__fs // down_n # サンプリングタイムを変更

        # scipy.decimsteを用いてダウンサンプリング
        down_data = signal.decimate(self.__norm_sound, q = down_n, 
                                    n = down_n*20+1, ftype = "fir", 
                                    zero_phase = True)
        
        """
        # 昔使ってたダウンサンプリング　※データ先頭にスパイクが生じるのでdecimateメソッドに変更
        # 折り返しノイズを防ぐため、ローパス
        lpf = signal.firwin(255, (down_fs/2) / (self.__fs / 2))             
        # ﾀﾞｳﾝｻﾝﾌﾟﾘﾝｸﾞ
        down_data = signal.lfilter(lpf, 1, self.__norm_sound)[::2]
        """
        down_wavedata = Wavedata(down_fs, down_data, down_name, self.__nbyte)
        
        return down_wavedata
    
    def resample_poly(self, rename = "_resample", up_n = 5, down_n = 16):
        """
        このメソッドは時系列データをれサンプリングします
        :param rename: ﾘｻﾝﾌﾟﾘﾝｸﾞ後の時系列データにつける名前
        :param up_n: アップサンプリング割合（整数） self.__fs*up_nにアップサンプリング
        :param down_n: ダウンサンプリング割合（整数） self.__fs/down_nにダウンサンプリング
        :return down_wavedata: ダウンサンプリングされたWavedata
        """
        
        rename = self.__name + rename
        re_fs = self.__fs * up_n // down_n # サンプリングタイムを変更

        # scipy.decimsteを用いてダウンサンプリング
        resample = signal.resample_poly(self.__norm_sound, up_n, down_n, 
                                         window=('kaiser', 5.0))
        
        resample_wavedata = Wavedata(re_fs, resample, rename, self.__nbyte)
        
        return resample_wavedata
    
    
    def moving_ave(self, sec = 1, plot = False):
        """このメソッドは振幅の移動平均をプロットします
        
        :param sec: 移動平均の時間 
        :param plot: Trueならグラフ表示を行う
        :return ave: 振幅の移動平均
        """
        
        n = sec * self.__fs
        b=np.ones(n) / n
        ave = np.convolve(abs(self.__norm_sound), b, mode='same')
        
        if plot == True:
            plt.figure(figsize=(15,3))
            #plt.plot(self.__time, self.__norm_sound)
            plt.plot(self.__time, ave)
            plt.title(self.__name)
            plt.xlabel("Time [s]")
            plt.ylabel("Average amplitude [Pa]")
            plt.show()
            
        return ave

        
    def belgian(self, velosity = 20, bel_name = "+belgian"):
        """このメソッドはQCTベルジャン路走行部分だけを抽出します
        
        :param velosity: テストコースの走行速度（15, 20, 30 [km/h]） 
        :param bel_name: この時系列データの名前
        :return down_wavedata: ベルジャン路部分の時系列データ
        """
        
        # 抽出後の時系列データの名前
        bel_name = self.__name + bel_name
        
        """
        # 走行開始地点抽出
        for i in range(len(self.__norm_sound)):
            if abs(self.__norm_sound [i]) > 0.05:
                bel_sound  = self.__norm_sound[i : ]   
                break
        
        # ベルジャン路終了部分抽出(走行速度に応じて秒数決め打ち）
        if velosity == 15:
            bel_sound = bel_sound[int(27 * self.__fs):int(71 * self.__fs)] 
        elif velosity == 20:
            bel_sound = bel_sound[int(23 * self.__fs):int(55 * self.__fs)] 
        elif velosity == 30:
            bel_sound = bel_sound[int(17 * self.__fs):int(37 * self.__fs)]  
        """    
        
        # 移動平均を行い、振幅の大きくなる部分をベルジャン路として抽出
        ma = self.moving_ave()
        start = np.where(ma > 0.095)[0][0]
        
        if velosity == 15:
            bel_sound = self.__norm_sound[start:start + int(44 * self.__fs)] 
        elif velosity == 20:
            bel_sound = self.__norm_sound[start:start + int(30 * self.__fs)] 
        elif velosity == 30:
            bel_sound = self.__norm_sound[start:start + int(20 * self.__fs)]
        
        bel_wavedata = Wavedata(self.__fs, bel_sound, bel_name, self.__nbyte)
    
        return bel_wavedata
    
    
    def cut_wav(self, start_sec, end_sec):
        """このメソッドは指定した秒数部分だけを抽出し、Wavedataクラスを返します
        
        :param start_sec: 切り出し開始秒 
        :param end_sec: 切り出し終了秒
        :return : 切り出された時系列データのWavedataインスタンス
        """
        
        start_index = int(start_sec * self.__fs)
        end_index = int(end_sec * self.__fs)
        
        cut_sound = self.__norm_sound[start_index : end_index] 
        
        return Wavedata(self.__fs, cut_sound, self.__name, self.__nbyte)   
    
    
    def zero_padding(self, total = 131072 + 256):
        """
        """
        
        import random
        
        nframe = len(self.__norm_sound)
        start_idx = random.randrange(total - nframe)
        start_idx = start_idx

        pad_sound = np.ones(total) * 10 ** -16
        
        pad_sound[start_idx:start_idx + nframe] = self.__norm_sound
 
        return Wavedata(self.__fs, pad_sound, self.__name, self.__nbyte)    
 
    
    def scipy_stft(self, N = 512):
        """このメソッドはSTFTを算出します（複素行列）
        
        :param N: STFTのフレーム幅(通常、20~40ms程度が良いらしい。が調検では2048を使用)
        :return stft: STFT結果（複素数行列）
        """
        
        freqs, time, local_stft = signal.stft(x = self.__norm_sound, 
                                              fs = self.__fs, nperseg = N, 
                                              return_onesided=False)
        local_stft = local_stft[:, 1:len(local_stft.T) - 1]
        
        return Stft(local_stft, self.__fs, self.__name)
    
    
    def devide(self, t):
        """このメソッドは時系列データをt秒ごとのフレームに分割します(行：フレーム,列：時系列サンプル)
        
        :param t: 分割音声データの時間 [s]
        :return devide_data: 分割された音声データ
        """
          
        data_in_frame = t * self.__fs
        n_frame = int(len(self.__norm_sound) / data_in_frame)
        
        # もとの時系列データをt秒分ごとに行列に変形する(行:t秒フレーム、列：フレーム数)
        cut_sound = self.__norm_sound[0 : data_in_frame * n_frame]
        devide_sound = cut_sound.reshape(n_frame, data_in_frame)
        
        # devide_wavedataリストにt秒ごとに分割されたフレームを格納
        devide_wavedata = []
        for i in range(n_frame):
            devide_name = self.__name + "_frame_" + str(i)
            devide_wavedata.append(Wavedata(self.__fs, devide_sound[i], 
                                          devide_name, self.__nbyte))
            
        return devide_wavedata
        
    
    def bpf(self, fe1 = 500, fe2 = 10000, ntap = 255, bpf_name = "+BPF"):
        """このメソッドは時系列データにバンドパスフィルタをかけます
        
        :param sound: 時系列データ
        :param fe1: カットオフ周波数1 [Hz]
        :param fe2: カットオフ周波数2 [Hz]
        :param bpf_name: フィルタ後のWavedataインスタンスの名前
        :return bpf_wavedata: wavedata
        """
        
        # インスタンスの名前を変更
        bpf_name = self.__name + bpf_name
        
        # バンドパスフィルタをかける
        fe1 = fe1 / (self.__fs / 2.0)        # カットオフ周波数1
        fe2 = fe2 / (self.__fs / 2.0)        # カットオフ周波数2
        bpf = signal.firwin(ntap, [fe1, fe2], pass_zero = False)
        sound = signal.lfilter(bpf, 1, self.norm_sound)
        
        # 信号処理で発生したスパイクを削除
        #sound = self.__del_spike(sound)
        
        bpf_wavedata = Wavedata(self.__fs, sound, bpf_name, self.__nbyte)
    
        return bpf_wavedata
    
    
    def __del_spike(self, sound):
        """このメソッドは時系列データにバンドパスフィルタをかけます
        
        :param sound: 時系列データ
        :return sound: スパイク削除後の時系列データ
        """

        sound_threshold = abs(sound).mean() * 5               #閾値：平均値の5倍
        sound = np.where(abs(sound) > sound_threshold, 0, sound)   #閾値以上を0に置換

        return sound

    
    def __judge_wavfile(self, wavedata1, wavedata2):
        """このメソッドは2つのwavedataの形式が同じかどうか判定します
        
        :param wavedata1: wavedata
        :param wavedata2: wavedata
        :return : 判定結果0:OK 1:NG
        """
        
        # バイト数（ハイレゾかどうか）、サンプリングレートが同じであればOK
        if wavedata1.__nbyte == wavedata2.__nbyte \
        and wavedata1.__fs == wavedata2.__fs:
            judge = True
        else:
            judge = False
            
        return judge
    
    
    def synthesis(self, wavedata, synthesis_name = "synthesis"):
        """このメソッドは2つの時系列データを合成します
        
        :param wavedata: 合成する騒音のwavedata
        :return :合成後のwavedata
        """
        
        # wav形式が同じであれば、自身(self)の音声データ長を基準に音声合成
        judge = self.__judge_wavfile(self, wavedata)
        if judge == True:
            if len(self.norm_sound) <= len(wavedata.norm_sound):
                synthesis_sound = self.norm_sound \
                                + wavedata.norm_sound[0:len(self.norm_sound)]
            else:
                print("合成する音声データ長が、もとのデータ長より短いため、合成できません")
        else:
            print("wavファイル形式が違うため、合成できません")

        return Wavedata(self.__fs, synthesis_sound, synthesis_name, 
                        self.__nbyte)

    
    def extract(self, wavedata, extract_name = "_extract"):
        """このメソッドは正常音との差分フィルタです
        
        :param wavedata: 正常車両のwavedata(引く側)
        :return :正常車両フィルタリング後のwavedata
        """

        # 差分フィルタ後の名前
        extract_name = self.__name + extract_name

        # STFTフレーム数とステップ数の決定（とりあえず調検テーマの値で固定）
        N = 2048
        step = 2048 // 2

        # 試験車両音と正常車両音のSTFT算出
        local_stft = self.stft(N, step)
        local_stft_normal = wavedata.stft(N, step)
        # scipy_stftを使用
        #local_stft = self.scipy_stft(N).stft.T
        #local_stft_normal = wavedata.scipy_stft(N).stft.T

        # それぞれのwavedataの平均FFT算出
        stft_mean = np.zeros(N)
        stft_normal_mean = np.zeros(N)
        for k in range(N):
            stft_mean[k] = abs(local_stft[:, k]).mean()
            stft_normal_mean[k] = abs(local_stft_normal[:, k]).mean()

        # 差分フィルタを適用
        stft_xtract = np.array(local_stft)
        filter = (stft_mean - stft_normal_mean) / stft_mean
        stft_xtract = local_stft * filter    # フィルタを元データにかける
        # 正常車両の方が大きかった成分は重みを1/4にする
        stft_xtract = np.where(filter < -0.05, stft_xtract / 4, stft_xtract)
        # 低周波成分は1/1024にカットする
        stft_xtract = np.c_[stft_xtract[:, :5] / 1024, stft_xtract[:, 5:]]

        # 逆FFT変換により、時系列データを復元し、その際発生しうるスパイクを削除
        #xtract_sound = self.istft(stft_xtract, N, step)
        #xtract_sound = self.__del_spike(xtract_sound)
        # scipy_istftを使用
        time, xtract_sound = signal.istft(Zxx = stft_xtract.T, fs = self.__fs,
                                  nperseg = N, input_onesided = False)
        
        return Wavedata(self.__fs, xtract_sound.real, 
                        extract_name, self.__nbyte)
    

    def adap_filter(self, wavedata, adap_name = "adap_filter", tapn = 5):
        """このメソッドは時系列データに適応フィルタをかけます
        
        :param wavedata: ノイズのwavedata
        :return :適応フィルタ後のwavedata
        """
        
        # 適応フィルタ後の名前
        adap_name = self.__name + adap_name
        
        # wav形式が同じであれば、適応フィルタをかける
        judge = self.__judge_wavfile(self, wavedata)
        if judge == True:
            adap_filter = np.zeros(tapn)
            adap_filter[tapn // 2] = 1
            y = np.zeros(len(self.norm_sound))
            adap_sound = np.zeros(len(self.norm_sound))
            norm = 0
            for i in range(len(self.norm_sound) - tapn):
                for n in range(tapn):
                    y[i] = y[i] + adap_filter[n] * wavedata.norm_sound[i-n]
                    norm = norm + self.norm_sound[i-n] * self.norm_sound[i-n]
                adap_sound[i] = self.norm_sound[i] - y[i]
                for n in range(tapn):
                    adap_filter[n] = adap_filter[n] + 0.1 * adap_sound[i] * self.norm_sound[i-n] / norm
        else:
            print("wavファイル形式が違うため、適応フィルタにはかけられません")

        return Wavedata(self.__fs, adap_sound, adap_name, self.__nbyte)
    
    
    def vol(self, dB = 10):
        """このメソッドはフィルタリング等で、音量の下がってしまった時系列データの音量を変更します
        
        :param dB: upさせたい音圧 [dB]
        :return :音量変更後のwavedata
        """
        
        sound = self.norm_sound * 10 ** (dB / 20) 
       
        return Wavedata(self.__fs, sound, self.__name, self.__nbyte)
    
    
    def nmf(self, n):   
        """このメソッドはNMFにより、音源分離を行います
        
        :param n: 分離する音源数
        :return nmf_wavedata: 音源分離後の時系列データリスト
        """
        
        N = 2048
        model = NMF(n_components = n, init='random', random_state=0)
        W = model.fit_transform(abs(self.stft(N, N//2)).T);
        H = model.components_;
        
        abs_spec = np.zeros((len(W), len(H.T)))
        nmf_wavedata = []
        for k in range(n):
            for i in range(len(W)):
                for j in range(len(H.T)):
                    abs_spec[i, j] = W[i, k] * H[k, j]
            abs_nmf = abs_spec.T.copy()
            # NMF後の絶対値比をもとにSTFTを再計算
            nmf_stft = self.stft(N, N//2) * (abs_nmf / abs(self.stft(N, N//2)))
            # 逆STFT変換により、音源分離
            nmf_sound = self.istft(nmf_stft, N, N//2)
            nmf_sound = self.__del_spike(nmf_sound)
            # スパイク削除
            nmf_name = self.__name + "_nmf_" + str(i)
            nmf_wavedata.append(Wavedata(self.__fs, nmf_sound, 
                                          nmf_name, self.__nbyte))
                            
        return nmf_wavedata



    """このメソッドは時系列データを16bitwavファイルに書き出すためのメソッドです"""
    """
    def write_wav(self):
        with wave.Wave_write(self.__name + ".wav") as w:
            w.setparams((
                1,                        # channel
                2,                        # byte width
                self.__fs,                  # sampling rate
                len(self.__norm_sound),     # number of frames
                "NONE", "not compressed"  # no compression
            ))
            data = np.array([self.__norm_sound * 32768],dtype = "int16")[0]
    
            binary = struct.pack("h" * len(data), *data)
            w.writeframes(binary)
    """
            
    def write_wav_sf(self, dir=None, filename=None, bit=16):
        """このメソッドは時系列データを24bitwavファイルに書き出すためのメソッドです"""
        
        import soundfile as sf

        save_file_name = os.path.basename(self.__name) if filename is None \
            else filename
        save_dir = os.getcwd() if dir is None else dir
        save_path = "{0}{1}{2}{3}".format(save_dir, "/", save_file_name, ".wav")

        subtype = "PCM_24" if bit == 24 else "PCM_16"

        sf.write(save_path, self.__norm_sound, self.__fs, subtype=subtype)
        print("Saved with", bit, "bit")

    def write_feature(self, 
                      fft_N = 2048, fft_bins=1024,
                      n_mels = 40, 
                      n_mfcc = 40):    
        """このメソッドは特徴量をpandas DataFrameに書き出します
        
        :param fft_N: FFTに用いるデータ点数(デフォルト2048)
        :param n_mels: Melspectrogramにて、メルフィルタバンクで束ねた後のデータ点数(128)
        :param n_mfcc: MFCCの次数（デフォルト40）
        """
        
        feature_df = pd.DataFrame()   
        
        # FFT
        fft_df = self.fft(N=fft_N, bins=fft_bins)
        feature_df = fft_df
        
        # Cepstrum
        cepstrum_df = self.cepstrum(N=fft_N)
        feature_df = feature_df.join(cepstrum_df)  
        
        # Melspectrogram
        melspectrogram_df = self.melspectrogram(N=fft_N, n_mels=n_mels)
        feature_df = feature_df.join(melspectrogram_df)  
        
        # MFCC
        mfcc_df = self.librosa_mfcc(N=fft_N, n_mels=n_mels, n_mfcc=n_mfcc)
        feature_df = feature_df.join(mfcc_df)           
            
        return feature_df


class Stft:
    """このクラスは読み込んだ時系列データを解析するためのものです(scipy_stftを前提)"""
    def __init__(self, stft, fs, name):
        """
        コンストラクタ
        
        :param stft: STFT
        :param fs: サンプリングレート　Hz
        """
        
        self.__stft = stft
        self.__fs = fs
        self.__name = name
        self.__N = len(stft)
    
    @property    
    def stft(self):
        """STFTデータ"""
        return self.__stft
    
    @property    
    def fs(self):
        """サンプリングレート"""
        return self.__fs

    @property    
    def name(self):
        """時系列データの名前"""
        return self.__name
    @name.setter
    def name(self, name):
        self.__name = name
    
    @property    
    def N(self):
        return self.__N
    
    
    def extract(self, normal_stft, extract_name = "+extract"):
        """このメソッドは正常音との差分フィルタです
        
        :param stft_normal: 正常車両のStft(引く側)
        :return :サブトラクション後のStft
        """
        # 差分フィルタ後の名前
        self.__name = self.__name + extract_name
        
        stft_inspect = self.__stft.T
        stft_normal = normal_stft.__stft.T
        
        # それぞれの平均FFT算出
        stft_mean = np.zeros(self.__N)
        stft_normal_mean = np.zeros(self.__N)
        for k in range(self.__N):
            stft_mean[k] = abs(stft_inspect[:, k]).mean()
            stft_normal_mean[k] = abs(stft_normal[:, k]).mean()

        # 差分フィルタを適用
        stft_xtract = np.array(stft_inspect)
        filter = (stft_mean - stft_normal_mean) / stft_mean
        stft_xtract = (stft_inspect * filter)   # フィルタを元データにかける

        # 正常車両の方が大きかった成分は重みを1/4にする
        stft_xtract = np.where(filter < -0.05, stft_xtract / 4, stft_xtract)
        
        # 低周波成分は1/1024にカットする

        stft_xtract = np.c_[stft_xtract[:, :5] / 1024, stft_xtract[:, 5:]]
        stft_xtract = stft_xtract.T

        return Stft(stft_xtract, self.__fs, self.__name)
        
    
    def scipy_istft(self):
        """このメソッドはSTFTを逆変換し、時系列データを復元します
        
        :return data.real: 時系列データ
        """
        
        time, data = signal.istft(Zxx = self.__stft, fs = self.__fs,
                                  nperseg = self.__N, input_onesided = False)
        
        return Wavedata(self.__fs, data.real, self.__name, 0)
        

class Multiwave:
    """このクラスは読み込んだ時系列データを解析するためのものです"""
    def __init__(self, wavedata_list):
        """
        コンストラクタ
        
        :param fs: サンプリングレート　Hz
        :param norm_sound: 時系列データリスト
        :param name: この時系列データの名前
        :param time: 時間 s
        :param nchan: アレイマイクのチャンネル数
        """
        
        self.__nchan = len(wavedata_list)
        self.__fs = wavedata_list[0].fs
        self.__name = wavedata_list[0].name
        self.__time = wavedata_list[0].time
        self.__nbyte = wavedata_list[0].nbyte
        self.__norm_sound = []
        for i in range(self.__nchan):
            self.__norm_sound.append(wavedata_list[i].norm_sound)

    @property    
    def name(self):
        """時系列データの名前"""
        return self.__name
    @name.setter
    def name(self, name):
        self.__name = name
    
    @property    
    def norm_sound(self):
        """-1~1に正規化された時系列データ"""
        return self.__norm_sound
    
    @property    
    def nchan(self):
        """-1~1に正規化された時系列データ"""
        return self.__nchan
    
    @property    
    def fs(self):
        """サンプリングレート"""
        return self.__fs
    
    @property    
    def time(self):
        return self.__time
    
    @property    
    def nbyte(self):
        """1点あたりのバイト数"""
        return self.__nbyte
    
    def plot(self, ymin = -1, ymax = 1):
        """このメソッドは時系列データをグラフ表示するためのものです"""
        plt.figure(figsize=(15, self.__nchan))
        for i in range(self.__nchan):
            plt.subplot(self.__nchan, 1, i+1)
            plt.plot(self.__time, self.__norm_sound[i])
            #plt.ylim(ymin, ymax)
        plt.title(self.__name)
        plt.xlabel("Time [s]")
        plt.ylabel("Sound pressure [Pa]")
        plt.show()
        plt.close()
        
        
    def stft_plot(self, N = 2048, cmin = -100, cmax = -80):
        """このメソッドはSTFTカラーマップをグラフ表示のみ行います)
        
        :param N: STFTフレームサイズ（通常2048）
        """
        
        pxx =[]
        plt.figure(figsize=(15, self.__nchan * 2))
        for n in range(self.__nchan):
            plt.subplot(self.__nchan, 1,n+1)
            p, freqs, bins, im = plt.specgram(self.__norm_sound[n],
                                                NFFT=N, Fs=self.__fs, 
                                                noverlap=N/2, 
                                                window=np.hanning(N))
            pxx.append(10 * np.log10(p.mean(1)))
            #plt.ylim(0,20000)
            plt.clim(cmin, cmax)
            plt.colorbar()
        plt.title(self.__name + "STFT")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency[Hz]")
        plt.show()
        plt.close()
        
        # STFT結果を時間平均し、FFTグラフを表示
        plt.figure(figsize=(15, 4))
        for n in range(self.__nchan):
            plt.plot(freqs, pxx[n])
            
        plt.title(self.__name + "FFT")
        plt.xlabel("frequency [Hz]")
        plt.ylabel("spectrum[dB]")
        #plt.xlim(0,20000)
        plt.ylim(cmin - 20 ,cmax + 20)
        plt.grid(True)
        plt.show()
        plt.close()
        
        
    def fft(self, N = 2048, bins = 1024):
        """このメソッドは平均のFFTを算出し、特徴量データフレームに出力します
        
        :param N: STFTのフレーム幅(通常、20~40ms程度が良いらしい。が調検では2048を使用)
        :return 平均のFFTデータフレーム
        """
        
        n_bins = (N // 2) // bins
        index_list = []
        fft_list = np.array(())
        for n in range(self.__nchan):
            fft_data = abs(self.scipy_stft(N = N).stft[n][0:N//2]).mean(1)
            fft_data = 20 * np.log10(fft_data)
            
            # バンド幅を束ねる(n個おきのFFTデータ取り出し、平均）
            integrated_fft = fft_data[::n_bins]
            for j in range(1, n_bins):
                integrated_fft += fft_data[j::n_bins]
            integrated_fft = integrated_fft / n_bins
            
            # 各チャンネルのFFTをリストに書き込み
            fft_list = np.append(fft_list, integrated_fft)
            
            # indexに特徴量を書き込む
            freqs = (np.arange(0, N//2, n_bins) * self.__fs // N).astype(np.str)
            Hzs = np.array(["Hz_FFT_{}ch".format(n+1)] * len(freqs))
            index_list = np.append(index_list, 
                                   np.core.defchararray.add(freqs, Hzs))
            
            """           
            for i in range(0, N//2, n_bins):
                index_list.append(str(int(self.__fs/N*i)) + "Hz_FFT_" \
                                  + str(n+1) + "ch")
            """
        # FFT_dataをデータフレームに書き込み
        fft_df = pd.DataFrame(fft_list, columns = [self.__name])                  
        fft_df.index = index_list
        
        return fft_df.T
    

    def cepstrum(self, N = 2048):
        """このメソッドは平均のケプストラムを算出し、特徴量データフレームに出力します
        
        :param N: STFTのフレーム幅(通常、20~40ms程度が良いらしい。が調検では2048を使用)
        :return 平均のケプストラムデータフレーム
        """
        
        index_list = []
        cepstrum_list = np.array(())
        for n in range(self.__nchan):
            fft_data = abs(self.scipy_stft(N = N).stft[n]).mean(1)
            fft_data = 20 * np.log10(fft_data)
            cepstrum = np.real(np.fft.ifft(fft_data))[:N//2]
            cepstrum_list = np.append(cepstrum_list, cepstrum)

            for i in range(N//2):
                index_list.append(str(i) + "th_Cepstrum_" + str(n+1) + "ch")

        cepstrum_df = pd.DataFrame(cepstrum_list, columns = [self.__name])                  
        cepstrum_df.index = index_list
        
        return cepstrum_df.T
        
        
    def melspectrogram(self, N = 2048, n_mels = 40, plot=False):
        """このメソッドはメルスペクトログラムを算出し、特徴量データフレームに出力します)
        
        :param N: STFTフレームサイズ（通常2048）
        :param n_mels: N個のSTFTデータを何次元のメルフィルタバンクに束ねるか
        :param plot: メルスペクトログラムのグラフ表示有無
        :return : メルスペクトログラム特徴量データフレーム）
        """
        
        ave_melspectrogram = np.array(())
        mels_list = librosa.core.mel_frequencies(n_mels=n_mels, fmax=self.__fs/2)
        
        if plot == True:
            plt.figure(figsize=(12, self.__nchan * 2))
        for n in range (self.__nchan):       
            melspectrogram = librosa.feature.melspectrogram(y=self.__norm_sound[n], 
                                                            sr=self.__fs,
                                                            n_fft = N, 
                                                            hop_length = N//2, 
                                                            n_mels = n_mels)    
            melspectrogram = 10 * np.log10(melspectrogram)
            ave_melspectrogram = np.append(ave_melspectrogram, 
                                           melspectrogram.mean(1))
            
            if plot == True:               
                plt.subplot(self.__nchan, 1, n+1)
                librosa.display.specshow(melspectrogram, sr=self.__fs, 
                                         hop_length = N // 2, 
                                         x_axis="time", y_axis="mel")
                plt.colorbar()
        if plot == True:
            plt.title(self.__name)
            plt.show()
            
            # STFT結果を時間平均し、FFTグラフを表示
            plt.figure(figsize=(12, 4))
            for n in range(self.__nchan):
                plt.plot(mels_list, ave_melspectrogram[n])
            plt.title(self.__name)
            plt.xlabel("frequency [Hz]")
            plt.ylabel("spectrum[dB]")
            plt.xlim(0, self.__fs // 2)
            plt.show()
            
        
        index_list = []
        melspectrogram_df = pd.DataFrame(ave_melspectrogram, 
                                         columns = [self.__name])
        for n in range(self.__nchan):
            for i in range(n_mels):
                index_list.append(str(int(mels_list[i])) + "Hz_melFFT_" \
                                  + str(n+1) + "ch")
                
        melspectrogram_df.index = index_list
            
        return melspectrogram_df.T
        
    
    def mfcc(self, N=2048, n_mels=40, n_mfcc=40, delta_k=0, plot=False):
        """このメソッドはMFCCを算出し、特徴量データフレームに出力します
        
        :param N: STFTフレームサイズ（通常2048）
        :param n_mels: N個のSTFTデータを何次元のメルフィルタバンクに束ねるか
        :param n_mfcc: n_melsに束ねられたメルスペクトログラムを何次元のMFCCデータにするか
        :param plot: MFCCのグラフ表示有無
        :return : MFCC特徴量データフレーム
        """
        
        from scipy.ndimage.interpolation import shift
        
        # グラフ表示        
        if plot == True:
            plt.figure(figsize=(12, self.__nchan * 2))
        
        # 出力用のデータフレーム定義
        output_df = pd.DataFrame(columns = [self.__name])

        for n in range (self.__nchan):       
            # 単一チャンネルのMFCCデータフレーム作成(mfcc_df)
            melspectrogram = librosa.feature.melspectrogram(y=self.__norm_sound[n], 
                                                            sr=self.__fs,
                                                            n_fft = N, 
                                                            hop_length = N//2, 
                                                            n_mels = n_mels)    
            melspectrogram = 10 * np.log10(melspectrogram)
            mfccs = librosa.feature.mfcc(y=self.__norm_sound[n], sr=self.__fs, 
                                         S = melspectrogram, n_mfcc=n_mfcc)
            
            ave_mfcc = mfccs.mean(1)   # MFCCをフレーム全体で時間平均
            mfcc_df = pd.DataFrame(ave_mfcc, columns = [self.__name]) 
            
            # indexリストに名前を格納
            index_list = []
            for i in range(n_mfcc):
                index_list.append(str(i+1) + "th_MFCC_" + str(n+1) + "ch")
            mfcc_df.index = index_list
     
        
            # 単一チャンネルのMFCC1次デルタデータフレーム作成(delta_df)
            if delta_k > 0:
                numer = 0
                denom = 0
                for k in range(-delta_k, delta_k + 1):
                    numer += k * shift(mfccs, [0,k] , cval=0)   # 分子
                    denom += k ** 2                             # 分母
                delta = numer / denom
                # デルタは平均すると0になると思われるので、絶対値を特徴量として算出
                delta_ave = abs(delta[:, k:len(delta.T) - k - 1]).mean(1)
                
                # MFCC1次デルタをデータフレームに格納(delta_df)
                delta_df = pd.DataFrame(delta_ave, columns = [self.__name])
                index_list = []
                for i in range(n_mfcc):
                    index_list.append(str(i+1) + "th_MFCC_delta_" + str(n+1) + "ch")            
                delta_df.index = index_list
                
                mfcc_df = mfcc_df.append(delta_df) # delta_dfをmfcc_dfにまとめる
            
            output_df = output_df.append(mfcc_df) # 出力用Dataframeに全ch書き込む

            # 各チャンネルグラフ表示                  
            if plot == True:
                plt.subplot(self.__nchan, 1, n+1)
                librosa.display.specshow(mfccs, sr=self.__fs, 
                                         hop_length = N // 2, x_axis="time")             
                plt.colorbar()
        
        # 時間平均グラフ表示          
        if plot == True:
            plt.title(self.__name + "_MFCC")
            plt.show()
            
            # 時間平均MFCCをグラフを表示
            plt.figure(figsize=(12, 4))
            for n in range(self.__nchan):
                plt.plot(ave_mfcc[n])
            plt.title(self.__name)
            plt.xlabel("quefrency")
            #plt.ylabel("spectrum[dB]")
            plt.show()
                            
        
        return output_df.T 
    
    
    def scipy_stft(self, N = 2048):
        """このメソッドはSTFTを算出します（複素行列）
        
        :param N: STFTのフレーム幅(通常、20~40ms程度が良いらしい。が調検では2048を使用)
        :return stft: STFT結果（複素数行列）
        """
        stft_list = []
        for i in range(self.__nchan):
            freqs, time, stft = signal.stft(x = self.__norm_sound[i], 
                                            fs = self.__fs, nperseg = N, 
                                            return_onesided=False)
            stft_list.append(stft)
        
        return Multistft(stft_list, self.__fs, self.__name)
    
    
    def down_sample(self, down_name = "_down_sampling", down_n = 2):
        """
        このメソッドは時系列データをデシメーションフィルリングした後、1/2倍のダウンサンプリングします
        :param down_name: ﾀﾞｳﾝｻﾝﾌﾟﾘﾝｸﾞ後の時系列データにつける名前
        :return down_wavedata: ダウンサンプリングされたWavedata
        """
        
        down_name = self.__name + down_name
        down_fs = self.__fs // down_n

        down_wavedata_list = []
        for i in range(self.__nchan):
            down_fs = self.__fs // down_n # サンプリングタイムを変更
    
            # scipy.decimateを用いてダウンサンプリング
            down_data = signal.decimate(self.__norm_sound[i], q = down_n, 
                                        n = down_n*20+1, ftype = "fir", 
                                        zero_phase = True)
            
            """
            # 折り返しノイズを防ぐため、10kHzローパス
            lpf = signal.firwin(255, (down_fs/2) / (self.__fs / down_n))             
            # ﾀﾞｳﾝｻﾝﾌﾟﾘﾝｸﾞ
            down_data = signal.lfilter(lpf, 1, self.__norm_sound[i])[::down_n]
            """
            
            down_wavedata_list.append(Wavedata(down_fs, down_data, 
                                               down_name, self.__nbyte))
        
        return Multiwave(down_wavedata_list)
    
    def resample_poly(self, rename = "_resample", up_n = 5, down_n = 16):
        """
        このメソッドは時系列データをれサンプリングします
        :param rename: ﾘｻﾝﾌﾟﾘﾝｸﾞ後の時系列データにつける名前
        :param up_n: アップサンプリング割合（整数） self.__fs*up_nにアップサンプリング
        :param down_n: ダウンサンプリング割合（整数） self.__fs/down_nにダウンサンプリング
        :return down_wavedata: ダウンサンプリングされたWavedata
        """
        
        rename = self.__name + rename
        re_fs = self.__fs * up_n // down_n # サンプリングタイムを変更

        # scipy.decimsteを用いてダウンサンプリング
        resample = signal.resample_poly(self.__norm_sound, up_n, down_n, 
                                         axis = 1, window=('kaiser', 5.0))

        resample_wavedata_list = []        
        for i in range(self.__nchan):
            resample_wavedata = Wavedata(re_fs, resample[i], 
                                         rename, self.__nbyte)
            resample_wavedata_list.append(resample_wavedata)
        
        return Multiwave(resample_wavedata_list)
    
    def moving_ave(self, sec = 1, plot = False):
        """このメソッドは振幅の移動平均をプロットします
        
        :param sec: 移動平均の時間 
        :return ave: 振幅の移動平均
        """
        
        n = sec * self.__fs
        b=np.ones(n) / n
        
        ave = np.convolve(abs(self.__norm_sound[0]), b, mode='same')
        
        if plot == True:
            plt.figure(figsize=(15,3))
            #plt.plot(self.__time, self.__norm_sound)
            plt.plot(self.__time, ave)
            plt.title(self.__name)
            plt.xlabel("Time [s]")
            plt.show()
            
        return ave
    
    
    def belgian(self, velosity = 20, bel_name = "+belgian"):
        """このメソッドはQCTベルジャン路走行部分だけを抽出します
        
        :param velosity: テストコースの走行速度（15, 20, 30 [km/h]） 
        :param bel_name: この時系列データの名前
        :return down_wavedata: ベルジャン路部分の時系列データ
        """
        
        # 抽出後の時系列データの名前
        bel_name = self.__name + bel_name
        
        """
        # 走行開始地点抽出
        for i in range(len(self.__norm_sound)):
            if abs(self.__norm_sound [i]) > 0.05:
                bel_sound  = self.__norm_sound[i : ]   
                break
        
        # ベルジャン路終了部分抽出(走行速度に応じて秒数決め打ち）
        if velosity == 15:
            bel_sound = bel_sound[int(27 * self.__fs):int(71 * self.__fs)] 
        elif velosity == 20:
            bel_sound = bel_sound[int(23 * self.__fs):int(55 * self.__fs)] 
        elif velosity == 30:
            bel_sound = bel_sound[int(17 * self.__fs):int(37 * self.__fs)]  
        """    
            
        ma = self.moving_ave()
        start = np.where(ma > 0.095)[0][0]
        
        bel_wavedata_list = []
        for i in range(self.__nchan):
            if velosity == 15:
                bel_sound = self.__norm_sound[i][start:start + int(44 * self.__fs)] 
            elif velosity == 20:
                bel_sound = self.__norm_sound[i][start:start + int(30 * self.__fs)] 
            elif velosity == 30:
                bel_sound = self.__norm_sound[i][start:start + int(20 * self.__fs)]
        
            bel_wavedata_list.append(Wavedata(self.__fs, bel_sound, 
                                               bel_name, self.__nbyte))
    
        return Multiwave(bel_wavedata_list)
    
    
    def cut_wav(self, start_sec, end_sec):
        """このメソッドは指定した秒数部分だけを抽出し、Wavedataクラスを返します
        
        :param start_sec: 切り出し開始秒 
        :param end_sec: 切り出し終了秒
        :return : 切り出された時系列データのMultiwaveインスタンス
        """
        
        start_index = int(start_sec * self.__fs)
        end_index = int(end_sec * self.__fs)

        cut_wavedata_list = []        
        for i in range(self.__nchan):
            cut_sound = self.__norm_sound[i][start_index : end_index]
            
            cut_wavedata_list.append(Wavedata(self.__fs, cut_sound, 
                                              self.__name, self.__nbyte))
        
        return Multiwave(cut_wavedata_list)    
 
    
    
    def devide(self, t):
        """このメソッドは時系列データをt秒ごとのフレームに分割します(行：フレーム,列：時系列サンプル)
        
        :param t: 分割音声データの時間 [s]
        :return devide_data: 分割された音声データ
        """

        data_in_frame = t * self.__fs
        n_frame = int(len(self.__norm_sound[0]) / data_in_frame)
        
        multiwave_list = []
        for i in range(n_frame):
            wavedata_list = []
            # devide_wavedataリストにt秒ごとに分割されたフレームを格納
            for nchan in range(self.__nchan):
                devide_name = self.__name + "_frame_" + str(i)
                # もとの時系列データをt秒分ごとに行列に変形する(行:t秒フレーム、列：フレーム数)            
                cut_sound = self.__norm_sound[nchan][0 : data_in_frame * n_frame]
                devide_sound = cut_sound.reshape(n_frame, data_in_frame)
                
                wavedata_list.append(Wavedata(self.__fs, devide_sound[i], 
                                                     devide_name, self.__nbyte))
                
            multiwave_list.append(Multiwave(wavedata_list))
            
        return multiwave_list
       
        
    def bpf(self, fe1 = 500, fe2 = 7500, ntap = 255, bpf_name = "+BPF"):
        """このメソッドは時系列データにバンドパスフィルタをかけます
        
        :param sound: 時系列データ
        :param fe1: カットオフ周波数1 [Hz]
        :param fe2: カットオフ周波数2 [Hz]
        :param bpf_name: フィルタ後のWavedataインスタンスの名前
        :return fcenters: 中心周波数
        """
        
        # インスタンスの名前を変更
        bpf_name = self.__name + bpf_name
        
        # バンドパスフィルタをかける
        fe1 = fe1 / (self.__fs / 2.0)        # カットオフ周波数1
        fe2 = fe2 / (self.__fs / 2.0)        # カットオフ周波数2
        bpf = signal.firwin(ntap, [fe1, fe2], pass_zero = False)
        
        bpf_wavedata_list = []
        for i in range(self.__nchan):
            sound = signal.lfilter(bpf, 1, self.norm_sound[i])
        
            # 信号処理で発生したスパイクを削除
            sound = self.__del_spike(sound)
        
            bpf_wavedata_list.append(Wavedata(self.__fs, sound, bpf_name, 
                                              self.__nbyte))
    
        return Multiwave(bpf_wavedata_list)
    
    
    def __del_spike(self, sound):
        """このメソッドは時系列データにバンドパスフィルタをかけます
        
        :param sound: 時系列データ
        :return sound: スパイク削除後の時系列データ
        """

        sound_threshold = abs(sound).mean() * 5               #閾値：平均値の5倍
        sound = np.where(sound > sound_threshold, 0, sound)   #閾値以上を0に置換

        return sound
    
    
    def synthesis(self, multi, synthesis_name = "synthesis"):
        """このメソッドは2つの時系列データを合成します
        
        :param wavedata: 合成する騒音のwavedata
        :return :合成後のwavedata
        """
        
        if len(self.norm_sound[0]) <= len(multi.norm_sound[0]):
            synthe_wavedata_list = []
            for i in range(self.__nchan):
                synthesis_sound = self.norm_sound[i] \
                                + multi.norm_sound[i][0:len(self.norm_sound[0])]
                synthe_wavedata_list.append(Wavedata(self.__fs, 
                                                     synthesis_sound, 
                                                     synthesis_name, 
                                                     self.__nbyte))
        else:
            print("合成する音声データ長が、もとのデータ長より短いため、合成できません")
       
        return Multiwave(synthe_wavedata_list)

    
      
    def fast_ica(self):   
        """このメソッドはFast ICAにより、複数音源混合音声の音源分離を行います
        
        :param wavedata_list: 音源分離したいマイクアレイの時系列データリスト 
        :return : 音源分離後の時系列データリスト
        """

        from sklearn.decomposition import FastICA

        ica = FastICA(n_components = self.__nchan)
        data = np.array(())
        for i in range(self.__nchan):
            data = np.append(data, self.__norm_sound[i])
        data = data.reshape(self.__nchan, len(data) // self.__nchan).T
        data = data - data.mean()
        ica.fit(data)
        
        return ica.transform(data).T
    
    
    def write_wav_sf(self, filepath = None, filename = "write.wav"):
        """このメソッドは時系列データを24bitwavファイルに書き出すためのメソッドです"""
        
        import soundfile as sf

        sf.write(filepath + filename, np.array(self.__norm_sound).T, self.__fs, 
                 subtype="PCM_24")
        
        
    def write_hdf(self, filepath = None, filename = "write.h5"):
        """このメソッドは時系列データを24bitwavファイルに書き出すためのメソッドです"""
        
        h5file = h5py.File("filename",'w')
        dset = h5file.create_dataset('time_data', 
                                     data= np.array(self.__norm_sound).T)
        dset.attrs.create(name = 'sample_freq', data = float(self.__fs))
        h5file.flush()
        h5file.close()
        
        shutil.move(filename, filepath + filename)
    
    
    def write_feature(self, 
                      fft_N = 2048, 
                      fft_bins = 1024,
                      cep_N = 2048,
                      n_mels = 40, 
                      n_mfcc = 40):   
        """このメソッドは特徴量をpandas DataFrameに書き出します
        
        :param fft_N: FFTに用いるデータ点数(デフォルト2048)
        :param fft_bins: FFTをいくつのbinに束ねるか(デフォルト1024：束ねない)
        :param n_mels: Melspectrogramにて、メルフィルタバンクで束ねた後のデータ点数(128)
        :param n_mfcc: MFCCの次数（デフォルト40）
        """
        
        feature_df = pd.DataFrame()   
        
        # FFT
        fft_df = self.fft(N=fft_N, bins=fft_bins)
        feature_df = fft_df
        
        # Cepstrum
        cepstrum_df = self.cepstrum(N=cep_N)
        feature_df = feature_df.join(cepstrum_df)  
        
        # Melspectrogram
        melspectrogram_df = self.melspectrogram(N=fft_N, n_mels=n_mels)
        feature_df = feature_df.join(melspectrogram_df)  
        
        # MFCC
        mfcc_df = self.mfcc(N=fft_N, n_mels=n_mels, n_mfcc=n_mfcc)
        feature_df = feature_df.join(mfcc_df)

            
        return feature_df
    
        
class Multistft:
    """このクラスは読み込んだ時系列データを解析するためのものです"""
    def __init__(self, stft_list, fs, name):
        """
        コンストラクタ
        
        :param stft: STFT
        :param fs: サンプリングレート　Hz
        """
        
        self.__nchan = len(stft_list)
        self.__stft = stft_list
        self.__fs = fs
        self.__name = name
        self.__N = len(stft_list[0])
        
    @property    
    def nchan(self):
        """サンプリングレート"""
        return self.__nchan
    
    @property    
    def stft(self):
        """STFTデータ"""
        return self.__stft
    
    @property    
    def fs(self):
        """サンプリングレート"""
        return self.__fs
    
    @property    
    def name(self):
        """時系列データの名前"""
        return self.__name
    @name.setter
    def name(self, name):
        self.__name = name
    
    @property    
    def N(self):
        return self.__N

    
    def scipy_istft(self):
        """このメソッドはSTFTを逆変換し、時系列データを復元します
        
        :return data.real: 時系列データ
        """
        wavedata_list = []
        for i in range(self.__nchan):
            time, data = signal.istft(Zxx = self.__stft[i], fs = self.__fs,
                                      nperseg = self.__N, input_onesided=False)
        
            wavedata_list.append(Wavedata(self.__fs, data.real, self.__name, 
                                          0))
        
        return Multiwave(wavedata_list)
    
    
class Feature:
    """このクラスは読み込んだ時系列データを解析するためのものです"""
    def __init__(self, X, y):
        """
        コンストラクタ
        
        :param X: 特徴量データフレーム
        :param y: クラスデータフレーム
        """
        
        self.__X = X
        self.__y = y
        
        self.__X_train = np.array(())
        self.__y_train = np.array(())
        self.__X_test = np.array(())
        self.__y_test = np.array(())
        
    @property    
    def X(self):
        """"""
        return self.__X
    
    @property    
    def y(self):
        """y"""
        return self.__y
    
    @property    
    def X_train(self):
        """"""
        return self.__X_train
    
    @property    
    def y_train(self):
        """y"""
        return self.__y_train
    
    @property    
    def X_test(self):
        """"""
        return self.__X_test
    
    @property    
    def y_test(self):
        """y"""
        return self.__y_test
    
    
    
    
    def totwoclass(self):
        """このメソッドは多クラスのラベルを2クラスに変換します
        
        """

        self.__y = self.__y.where(self.__y == 0, 1)
    
    def combineclass(self):
        """このメソッドは9クラスを5クラスにまとめてリストに出力します
        
        """

        self.__y = self.__y.where(self.__y != 2, 1)
        self.__y = self.__y.where(self.__y != 3, 2)
        self.__y = self.__y.where(self.__y != 4, 2)
        self.__y = self.__y.where(self.__y != 5, 3)
        self.__y = self.__y.where(self.__y != 6, 3)
        self.__y = self.__y.where(self.__y != 7, 4)
        self.__y = self.__y.where(self.__y != 8, 4)
    
    
    def separateclass(self):
        """このメソッドは特徴量行列をクラスごとにまとめてリストに出力します
        
        :return　各クラスの特徴量リスト
        """
        
        class_list = []
        for i in range(int(self.__y.max() + 1)): # 最大のクラスまで
            class_list.append(self.__X[self.__y == i])
        
        return class_list
    
    
    def f_test(self, p_thresh=0.05):
        """このメソッドは2クラス間の特徴量の等分散性をF検定により評価します
        
        :return p_value: F値, p値のデータフレーム　<0.05で有意差あり
        """
        
        self.__y = self.__y.where(self.__y == 0, 1)
        
        class_list = self.separateclass()
        
        f = np.var(class_list[0]) / np.var(class_list[1])
        
        df0 = len(class_list[0]) - 1
        df1 = len(class_list[1]) - 1
        
        p_value = stats.f.cdf(f, df0, df1)
        p_value = pd.DataFrame(p_value, 
                               index = class_list[0].columns)
        
        f_df = pd.concat([f, p_value], axis = 1)
        f_df.columns = ["F", "F_test_p_value"]
        
        judge = pd.DataFrame(f_df.iloc[:,1] < p_thresh)
        judge.columns = ["F_test_sig.?"]

        f_df = f_df.join(judge)
    
        return f_df
    
    
    def t_test(self, equal_var = False, p_thresh=0.05):
        """このメソッドは2クラス間の特徴量に有意差があるかをt検定を用いて評価します
        
        :param equal_var: 等分散性の仮定（True:t検定, False:Welchのt検定）
        :return p_value: 統計量, p値のデータフレーム　<0.05で有意差あり
        """
        
        self.__y = self.__y.where(self.__y == 0, 1)
        
        class_list = self.separateclass()
        statistics, p_value = stats.ttest_ind(class_list[0], class_list[1], 
                                              equal_var=equal_var)
        
        t_df = pd.DataFrame([statistics, p_value], 
                            index = ["Statistics", "t_test_p_value"], 
                            columns = class_list[0].columns).T
                            
        judge = pd.DataFrame(t_df.iloc[:,1] < p_thresh)
        judge.columns = ["t_test_sig.?"]

        t_df = t_df.join(judge)
 
        return t_df
    
    
    def levene(self, center="mean", p_thresh=0.05):
        """このメソッドは3クラス間以上の特徴量の等分散性をLevene検定により評価します
        
        :param center: mean, median, trimmed
        :return p_value: 統計量, p値のデータフレーム　<0.05で有意差あり
        """
        
        class_list = self.separateclass()

        n_class = len(class_list)
        s = np.array(())
        p = np.array(())

        if n_class == 2:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.levene(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   center=center)
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 3:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.levene(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   center=center)
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 4:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.levene(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   center=center)
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 5:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.levene(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i],
                                                   center=center)
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 6:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.levene(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i],
                                                   class_list[5].iloc[:,i], 
                                                   center=center)
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 7:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.levene(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i],
                                                   class_list[5].iloc[:,i], 
                                                   class_list[6].iloc[:,i],
                                                   center=center)
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 8:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.levene(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i],
                                                   class_list[5].iloc[:,i], 
                                                   class_list[6].iloc[:,i],
                                                   class_list[7].iloc[:,i],
                                                   center=center)
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                                 
        if n_class == 9:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.levene(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i],
                                                   class_list[5].iloc[:,i], 
                                                   class_list[6].iloc[:,i],
                                                   class_list[7].iloc[:,i],
                                                   class_list[8].iloc[:,i],
                                                   center=center)
                s = np.append(s, statistics)
                p = np.append(p, p_value)
         
            
        levene_df = pd.DataFrame([s,p],
                            index = ["Levene", "Leven_p_value"], 
                            columns = class_list[0].columns).T
                                         
        judge = pd.DataFrame(levene_df.iloc[:,1] < 0.05)
        judge.columns = ["Levene_sig.?"]

        levene_df = levene_df.join(judge)
    
        return levene_df
    
    
    def anova(self):
        """このメソッドは3クラス間以上の特徴量の有意差をone-way ANOVAにより評価します
        
        :return 統計量, p値のデータフレーム　<0.05で有意差あり
        """
        
        class_list = self.separateclass()

        n_class = len(class_list)
                            
        if n_class == 2:
            statistics, p_value = stats.f_oneway(class_list[0], class_list[1])
                
        if n_class == 3:
            statistics, p_value = stats.f_oneway(class_list[0], class_list[1],
                                               class_list[2])                
        if n_class == 4:
            statistics, p_value = stats.f_oneway(class_list[0], class_list[1],
                                               class_list[2], class_list[3])                
        if n_class == 5:
            statistics, p_value = stats.f_oneway(class_list[0], class_list[1],
                                               class_list[2], class_list[3],
                                               class_list[4])
        if n_class == 6:
            statistics, p_value = stats.f_oneway(class_list[0], class_list[1],
                                               class_list[2], class_list[3],
                                               class_list[4], class_list[5])
        if n_class == 7:
            statistics, p_value = stats.f_oneway(class_list[0], class_list[1],
                                               class_list[2], class_list[3],
                                               class_list[4], class_list[5],
                                               class_list[6])
        if n_class == 8:
            statistics, p_value = stats.f_oneway(class_list[0], class_list[1],
                                               class_list[2], class_list[3],
                                               class_list[4], class_list[5],
                                               class_list[6], class_list[7])
        if n_class == 9:
            statistics, p_value = stats.f_oneway(class_list[0], class_list[1],
                                               class_list[2], class_list[3],
                                               class_list[4], class_list[5],
                                               class_list[6], class_list[7],
                                               class_list[8])
            
                
        anova_df = pd.DataFrame([statistics, p_value],
                            index = ["ANOVA", "ANOVA_p_value"], 
                            columns = class_list[0].columns).T
                                
        judge = pd.DataFrame(anova_df.iloc[:,1] < 0.05)
        judge.columns = ["Anova_sig.?"]

        anova_df = anova_df.join(judge)
    
        return anova_df

    
    def kruskal(self):
        """このメソッドは3クラス間以上の特徴量の有意差をKruskal-Wallis検定で評価します
        
        :return 統計量, p値のデータフレーム　<0.05で有意差あり
        """
        
        class_list = self.separateclass()

        n_class = len(class_list)
        s = np.array(())
        p = np.array(())

        if n_class == 2:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.kruskal(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i])
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 3:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.kruskal(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i])
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 4:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.kruskal(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i])
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 5:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.kruskal(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i])
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 6:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.kruskal(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i],
                                                   class_list[5].iloc[:,i])
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 7:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.kruskal(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i],
                                                   class_list[5].iloc[:,i], 
                                                   class_list[6].iloc[:,i])
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        if n_class == 8:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.kruskal(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i],
                                                   class_list[5].iloc[:,i], 
                                                   class_list[6].iloc[:,i],
                                                   class_list[7].iloc[:,i])
                s = np.append(s, statistics)
                p = np.append(p, p_value)
        if n_class == 9:
            for i in range(len(class_list[0].T)):
                statistics, p_value = stats.kruskal(class_list[0].iloc[:,i], 
                                                   class_list[1].iloc[:,i],
                                                   class_list[2].iloc[:,i],
                                                   class_list[3].iloc[:,i],
                                                   class_list[4].iloc[:,i],
                                                   class_list[5].iloc[:,i], 
                                                   class_list[6].iloc[:,i],
                                                   class_list[7].iloc[:,i],
                                                   class_list[8].iloc[:,i])
                s = np.append(s, statistics)
                p = np.append(p, p_value)
                
        kruskal_df = pd.DataFrame([s,p],
                            index = ["Kruskal", "Kruskal_p_value"], 
                            columns = class_list[0].columns).T
                                  
        judge = pd.DataFrame(kruskal_df.iloc[:,1] < 0.05)
        judge.columns = ["Kruskal_sig.?"]

        kruskal_df = kruskal_df.join(judge)
    
        return kruskal_df
    
    
    def tukey(self, alpha = 0.05):
        """このメソッドは3クラス以上の特徴量に対し、Tukey-Kramerの多重比較検定を行います
        
        :param alpha: 有意水準（デフォルト0.05だが、もっと少なくてもよいかも） 
        :return 多重比較検定の結果データフレーム
        """
        
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        index_list = []            # tukey_dfのindex名リスト 
        tukey_df = pd.DataFrame()  # 出力するデータフレーム
        
        #tukey_list = []            # statsmodelsモジュール形式で保存するためのリスト
        
        n_class = int(self.__y.max()) + 1  # 比較するクラス数
        
        # 比較する2クラスの名前を作成
        columns = np.array(())
        for i in range(n_class):
            start = i + 1
            for j in range(start, n_class):
                columns = np.append(columns, [i, j])
        columns = columns.reshape(len(columns) // 2, 2)
        
        # 多重比較検定
        for i in range(len(self.__X.T)):
            tukey = pairwise_tukeyhsd(endog=self.__X.iloc[:,i],
                              groups=self.__y, alpha=0.05)
            #tukey_list.append(tukey) #statsmodels形式の結果もリストに保存しておく
            
            # 平均値の差 / 標準誤差（あっているか謎）
            q_value = np.abs(tukey.meandiffs) / tukey.std_pairs
            
            # 特徴量１つ分の検定結果をデータフレームに書き出し
            one_tukey_df = pd.DataFrame([columns[:, 0], columns[:, 1], 
                                         tukey.meandiffs, tukey.std_pairs, 
                                         q_value, 
                                         np.ones(len(columns)) * tukey.q_crit,
                                         tukey.confint[:,0], 
                                         tukey.confint[:,1], 
                                         tukey.reject]).T
            
            tukey_df = tukey_df.append(one_tukey_df) # 各特徴量データフレームappend
            for n in range(len(columns)):
                index_list.append(self.__X.columns[i]) # index名をリストに書き出し
        
        # index, columnsを書き込み
        tukey_df.index = index_list
        tukey_df.columns = ["class_1", "class_2", 
                            "Mean_difference", "Std", "Q_value", "Q_thresh",
                            "Lower", "Upper", "Reject"]
            
        return tukey_df#, tukey_list

    
    def SelectKBest(self, k=40):
        """このメソッドは特徴選択を行います
        
        :param k: 選択する特徴料の次元（デフォルト40次元）
        :param selecter: 特徴選択する際に用いる基準
                        f_classif: F検定, 
                        mutual_info_classif: 相互情報量,
                        chi2: Χ２乗基準）
        :return 選択されたFeatureクラスのインスタンス
        """
        
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

        skb = SelectKBest(mutual_info_classif, k=k)
        skb.fit(self.__X, self.__y)
        X_skb = skb.transform(self.__X)
        skb.get_support()
        X_skb = self.__X.iloc[:,np.where(skb.get_support() == True)[0]]
        
        return Feature(X_skb, self.__y)
    
    
    def StandardScaler(self):
        """このメソッドは特徴量の標準化（平均0, 分散1）を行います
        
        :return 標準化されたFeatureクラスのインスタンス
        """
        
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaler.fit(self.__X)
        X_scale = scaler.transform(self.__X)
        X_scale = pd.DataFrame(X_scale, 
                               index = self.__X.index, 
                               columns = self.__X.columns)
        
        return Feature(X_scale, self.__y)
    
  
    def MinMaxScaler(self):
        """このメソッドは特徴量のスケーリング（±1）を行います
        
        :return スケーリングされたFeatureクラスのインスタンス
        """
        
        from sklearn.preprocessing import MinMaxScaler
        
        mmscaler = MinMaxScaler([-1,1])
        mmscaler.fit(self.__X)
        X_mms = mmscaler.transform(self.__X)
        X_mms = pd.DataFrame(X_mms, 
                             index = self.__X.index, 
                             columns = self.__X.columns)
        
        return Feature(X_mms, self.__y)
    

    def PCA(self, n_components=40, whiten=True):
        """このメソッドは主成分分析PCAを行います
        
        : param n_components: PCAで削減後の次元
        : param whiten: 白色化を行うかどうか（デフォルトTrue）
        :return PCAされたFeatureクラスのインスタンス
        """
        
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components = n_components, whiten=whiten)
        pca.fit(self.__X)
        X_pca = pca.transform(self.__X)
        X_pca = pd.DataFrame(X_pca, index = self.__X.index)
        
        # pca.components_
        #plt.plot(pca.explained_variance_ratio_)
        #plt.plot(np.add.accumulate(pca.explained_variance_ratio_))

        return Feature(X_pca, self.__y)

    
    def discriptives(self):
        """このメソッドは分離度、クラス内分散、クラス間分散を求めます
        
        :return 分離度、クラス内分散、クラス間分散データフレーム
        """
        
        class_list = self.separateclass()
        
        # 各特徴量全体にの平均、分散を算出
        total_num = len(self.__X)
        total_mean = self.__X.mean(0)
        total_var = self.__X.var(0)
        
        feature_test = pd.concat([total_mean, total_var], axis=1)
        columns_list = ["Total_mean", "Total_var"]
        
        total_inclass_var = 0
        total_interclass_var = 0
        for i in range(int(self.__y.max() + 1)): # 最大のクラスまで
            inclass_mean = class_list[i].mean(0) # このクラスの平均
            inclass_var = class_list[i].var(0)   # このクラスの分散
            
            # 特徴量評価データフレームに格納
            feature_test = pd.concat([feature_test, inclass_mean], axis=1)
            feature_test = pd.concat([feature_test, inclass_var], axis=1)
            
            # columns名をリストに格納
            columns_list.append("Class_" + str(i) + "_mean")
            columns_list.append("Class_" + str(i) + "_var")

            total_inclass_var += len(class_list[i]) * inclass_var #　クラス内分散用変数
            total_interclass_var += len(class_list[i]) * (inclass_mean - total_mean) ** 2

        total_inclass_var = total_inclass_var / total_num
        total_interclass_var = total_interclass_var / total_num
        sep_deg = total_interclass_var / total_inclass_var        

        feature_test = pd.concat([feature_test, total_interclass_var], axis=1)
        feature_test = pd.concat([feature_test, total_inclass_var], axis=1)
        feature_test = pd.concat([feature_test, sep_deg], axis=1)
        
        columns_list.append("Interclass_var")
        columns_list.append("Inclass_var")
        columns_list.append("Separation_degree")
        
        feature_test.columns = columns_list
        
        return feature_test
    
    
    def separation_degree(self):
        """このメソッドは多次元で分離度、クラス内分散、クラス間分散を求めます
        
        :return 分離度、クラス内分散、クラス間分散データフレーム
        """
        
        class_list = self.separateclass()
        
        # 特徴量全体の平均ベクトル、データ数を算出
        total_num = len(self.__X)     # 全体のデータ数
        total_mean = self.__X.mean(0) # 全体の平均ベクトル

        # 変数（１クラス内分散、各クラス内分散の平均、クラス間分散）初期化
        inclass_var = 0
        inclass_var_sum = 0
        interclass_var_sum = 0
        
        # 戻り値用のデータフレーム
        separation_df = pd.DataFrame()
        index_list = []
       
        # 全クラスごとに分散算出
        for i in range(int(self.__y.max() + 1)): 
            # ある１クラスのクラス内分散算出
            inclass_dis2_sum = 0
            for n in range(len(class_list[i])):
                inclass_dis = class_list[i].iloc[n,:] - class_list[i].mean(0)
                inclass_dis2 = np.dot(inclass_dis, inclass_dis) # 1データの差の2乗
                inclass_dis2_sum += inclass_dis2           #　差の2乗合計
            inclass_var = inclass_dis2_sum / len(class_list[i]) # クラス内分散
            
            inclass_var_sum += inclass_dis2_sum       # クラス内分散×データ数を合計
            
            # ある1クラスと全体平均の差
            interclass_dis = class_list[i].mean(0) - total_mean
            interclass_dis2 = np.dot(interclass_dis, interclass_dis) # 距離の2乗
            # 全体平均との差のクラス間分散重みづけ和
            interclass_var_sum += len(class_list[i]) * interclass_dis2
            
            # 参考 ある1クラスとクラス0(正常車両)のクラス間距離
            #dis_to_normal = class_list[i].mean(0) - class_list[0].mean(0)
            #dis_to_normal_scalar = np.dot(dis_to_normal, dis_to_normal)
            
            temp_df = pd.DataFrame([np.linalg.norm(interclass_dis), inclass_var])
            #temp_df = temp_df.append(pd.DataFrame([dis_to_normal_scalar]))
            separation_df = separation_df.append(temp_df)
            
            index_list.append("Class_" + str(i) + "_interclass_dis")
            index_list.append("Class_" + str(i) + "_inclass_var")
            #index_list.append("Class_" + str(i) + "_dis_to_normal")
            
        # データ数で割った後、分離度算出
        inclass_var_ave = inclass_var_sum / total_num
        interclass_var_ave = interclass_var_sum / total_num
        sep_deg = interclass_var_ave / inclass_var_ave
        
        temp_df = pd.DataFrame([interclass_var_ave, inclass_var_ave, sep_deg]) 
        separation_df = separation_df.append(temp_df)

        index_list.append("interclass_var")
        index_list.append("inclass_var")
        index_list.append("separation_degree")

        separation_df.index = index_list
            
        return separation_df.T


    def StratifiedShuffleSplit(self, n_splits=1, train_size=0.5):
        """このメソッドは特徴量、ラベルを同じ割合で訓練用、テスト用に分割します
        
        : param n_splits: 調査中
        : param train_size: 訓練データの割合（0.5, 0.8等）
        : param test_size: テストデータの割合（0.5, 0.2等）
        """
        
        from sklearn.model_selection import StratifiedShuffleSplit
        
        ss = StratifiedShuffleSplit(n_splits=n_splits, 
                            train_size=train_size, 
                            test_size = 1.0 - train_size)
        
        # 学習データとテストデータのインデックスを作成
        train_index, test_index = next(ss.split(self.__X, self.__y))
        
        # 学習データとテストデータに分割
        self.__X_train = self.__X.iloc[train_index]
        self.__X_test  = self.__X.iloc[test_index] 
        self.__y_train = self.__y.iloc[train_index] 
        self.__y_test  = self.__y.iloc[test_index]
        

    def KNeighborsClassifier(self, n_neighbors=10):
        """このメソッドはK最近傍法によるクラス分類を行います
        
         : param n_neighbors: 何個目までの近傍データを使用するか
        """
        
        from sklearn import neighbors

        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(self.__X_train, self.__y_train)

        y_pred = clf.predict(self.__X_test)
        cmat = self.__classification_report(y_pred, name="K Nearest Neighbors")
        
        return cmat, clf.score(self.__X_test, self.__y_test)
    
    
    def RandomForestClassifier(self, n_estimators=50):
        """このメソッドはランダムフォレストによるクラス分類を行います
        
        : param n_estimators: 何個の木を用いて分類を行うか
        """
        
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators = n_estimators)
        clf.fit(self.__X_train, self.__y_train)
        
        y_pred = clf.predict(self.__X_test)
        cmat = self.__classification_report(y_pred, name="Random Forest")
        
        return cmat, clf.score(self.__X_test, self.__y_test)
    
    
    def LogisticRegression(self, C=1):
        """このメソッドはロジスティック回帰を用いてクラス分類を行います
        
        : param C: 学習パラメータ（通常C>1）
        """
        
        from sklearn import linear_model
        
        clf = linear_model.LogisticRegression(C=C)
        clf.fit(self.__X_train, self.__y_train)
        
        y_pred = clf.predict(self.__X_test)
        cmat = self.__classification_report(y_pred, name="Logistic Regression")
        
        return cmat, clf.score(self.__X_test, self.__y_test)
    
    
    def SVC(self, kernel = "rbf", C=1):
        """このメソッドはSVMを用いてクラス分類を行います
        
        : param kernel: linearカーネル or rbf kernel
        : param C: 学習パラメータ（通常C>1）
        """
        
        from sklearn.svm import SVC

        clf = SVC(C=C, kernel=kernel, probability=True)
        clf.fit(self.__X_train, self.__y_train)

        y_pred = clf.predict(self.__X_test)
        cmat = self.__classification_report(y_pred, name="SVM " + kernel + "_kernel")
        
        return cmat, clf.score(self.__X_test, self.__y_test)
    
    
    def MLP(self, activation='relu', alpha=1, 
                            hidden_layer_sizes=(100, 100, 100, 100, 10), 
                            max_iter=2000):
        """このメソッドはMLPを用いてクラス分類を行います
        
        : param activation: activation function(relu, sigmoid etc.)
        : param alpha: 学習パラメータ
        : param hidden_layer_sizes: 隠れ層の数
        : param max_iter: 繰り返し数
        """
        
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(activation=activation, alpha=alpha, 
                            hidden_layer_sizes=hidden_layer_sizes, 
                            max_iter=max_iter)
        
        clf.fit(self.__X_train, self.__y_train)

        y_pred = clf.predict(self.__X_test)
        cmat = self.__classification_report(y_pred, name="MLP")
        
        return cmat, clf.score(self.__X_test, self.__y_test)
    
    
    
    def OneClassSVM(self, kernel = "rbf", nu=0.5, gamma=0.1, NG_data=True):
        """このメソッドはSVMを用いてクラス分類を行います
        
        : param kernel: linearカーネ or rbf kernel
        : param C: 学習パラメータ（通常C>1）
        : param NG_data: self.__X_trainにNGデータを含むか（Trueの場合は含む）
        """
        
        from sklearn.svm import OneClassSVM
        
        train_data = self.__X_train
        
        # self.__X_trainにNGデータが含まれる場合は、正常データのみを学習に使用する
        if NG_data == True:
            train_data = self.__X_train[self.__y_train == 0]

        clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        clf.fit(train_data)

        y_pred = clf.predict(self.__X_test) # 1:normal -1:abnormal
        y_pred = (y_pred - 1) // -2         # 0:normal 1:abnormalに変換
        
        cmat = self.__classification_report(y_pred, name="OneClassSVM")

        accuracy = (cmat[0, 0] + cmat[1,1]) / cmat.sum()
        
        return cmat, accuracy
    
    
    def __classification_report(self, y_pred, name):
        """このメソッドはロジスティック回帰を用いてクラス分類を行います
        
        : param C: 学習パラメータ（通常C>1）
        """
        
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report     
        
        cmat = confusion_matrix(self.__y_test, y_pred)
    
        #n_precision = cmat[1,1] / (cmat[0,1] + cmat[1,1])
        #TN_rate = cmat[1,1] / (cmat[1,0] + cmat[1,1])
        
        return cmat
    

if __name__ == "__main__":    
    """
    # 調検時の12次元MFCCを読み込み
    csv_data = np.loadtxt("C:\\Users\\yui_sudo\\Documents\\source\\異音テスト\\2sec_12mfcc_4ch.csv", delimiter=",")
    csv_data = csv_data[np.where(csv_data[:,48]==20)]
    y = csv_data[:, 49]
    y[y>0] = 1
    X = csv_data[:, :48]
    
    y = pd.DataFrame(y, columns = ["Class"])
    X = pd.DataFrame(X)
    
    columns_list = []
    for n in range(4):
        for i in range(12):
            columns_list.append(str(i+1) + "th_MFCC_" + str(n+1) + "ch")
    X.columns = columns_list
    
    index_list = []
    for i in range(len(X)):
        index_list.append("No." + str(i))
    X.index = index_list
    y.index = index_list
    """
    
    """
    # スピーカーで流したステップワゴンデータの読み込み
    directory = "D:\\180227define_jab\\" 
    filename = "condition.csv"
    
    csv_data = np.loadtxt(directory + filename, delimiter=",")
    belgian = []
    flat = []
    for i in range(1):
    #for i in range(len(csv_data)):
        if csv_data[i, 1] ==37:
            test_no = str(int(csv_data[i, 0]))
            multi = WavfileOperate(directory + "00" + test_no 
                                           + "-AAA-01 (Rec01)\\" + test_no + ".wav")
            multi = Multiwave(multi.wavedata)
            belgian.append(multi)
        
        else:
            test_no = str(int(csv_data[i, 0]))
            multi = WavfileOperate(directory + "00" + test_no 
                                           + "-AAA-01 (Rec01)\\" + test_no + ".wav")
            multi = Multiwave(multi.wavedata)
            flat.append(multi)    
    """
    
    normal = WavfileOperate("D:\\170825QCT\\wav\\00030-AAA-01 (Rec01)\\Channel001_001.wav", logger = 1).wavedata.down_sample().belgian()
    abnormal = WavfileOperate("D:\\170825QCT\\wav\\00194-AAA-01 (Rec01)\\Channel001_001.wav", logger = 1).wavedata.down_sample().belgian()

    normal.name = "No.030"
    abnormal.name = "No.194"
    
    m_normal = WavfileOperate("D:\\180227define_jab\\00296-AAA-01 (Rec01)\\296.wav").multiwave.down_sample().belgian()
    m_abnormal = WavfileOperate("D:\\180227define_jab\\00324-AAA-01 (Rec01)\\324.wav").multiwave.down_sample().belgian()
    
    m_normal.name = "No.296"
    m_abnormal.name = "No.324"
    
    m_normal.fft()
    
    
    """
    #QCTテストコースの全データを読み込み 
    csv_data = np.loadtxt("D:\\170825QCT\\wav\\170825QCT.csv", delimiter=",")
    X = pd.DataFrame()
    for n in range(len(csv_data)):
        test_no = str(int(csv_data[n, 0]))
        if len(test_no) == 2:
            test_no = "0" + test_no
        velosity = int(csv_data[n, 1])
        
        if velosity == 20:
            multi = WavfileOperate("D:\\170825QCT\\wav\\00" + test_no + \
                                  "-AAA-01 (Rec01)\\" + test_no + ".wav").multiwave  
            multi = multi.down_sample()
            multi.name = test_no
            devide = multi.devide(2)
            for i in range(len(devide)):
                x= pd.DataFrame()
                x = devide[i].write_feature(fft_N=2048,fft_bins=1024, 
                                              cep_N=2048,
                                              n_mels=40, n_mfcc=40)
                
                y = pd.DataFrame([csv_data[n,2]], index = [devide[i].name], columns = ["Class"])
                #y = pd.DataFrame([csv_data[n,3]], index = [devide[i].name], columns = ["Class"])
                x = x.join(y)
                X = X.append(x)
            print(devide[i].name)
    y = X.iloc[:, len(X.T) - 1]
    #X = X.iloc[:, :len(X.T) - 1]
    X.to_csv("feature.csv")
    """
    
    # csvから特徴量読み込み
    #data = pd.read_csv("feature.csv", delimiter=",", index_col=0)
    #X = data.iloc[:, :len(data.T) - 1]
    #y = data.iloc[:,  len(data.T) - 1]
    
    #X = data.iloc[:, 8714:8716] # 11th, 12th MFCC 1chのみ取り出す
    #y = data.iloc[:, len(data.T) - 1]
    
    #F = Feature(X, y)
    #F.combineclass() # 9クラスを５クラスにまとめる
    #F.totwoclass()   # 多クラスを2クラスにまとめる
    """
    # 分離度による特徴量の評価
    discriptive = F.discriptives()
    discrip_sort=discriptive.sort_values(by=["Separation_degree"], 
                                         ascending=False)
    """
    # 大津の手法による多次元での分離度評価(9クラス)
    #sep_deg_9 = F.separation_degree()
    
    #levene = F.levene()
    #kruskal = F.kruskal()
    #tukey = F.tukey()
    
    #F.totwoclass()   # 多クラスを2クラスにまとめる
    #f = F.f_test()
    #t = F.t_test()
    
    # 大津の手法による多次元での分離度評価
    #sep_deg_2 = F.separation_degree()
    
    #F = F.SelectKBest(k=40)
    """
    # スケーリングの検討検討
    F_stand = F.StandardScaler()
    stand_sep = F_stand.discriptives()
    stand_sort=stand_sep.sort_values(by=["Separation_degree"], ascending=False)
    
    F_min_max = F.MinMaxScaler()
    min_max_sep = F_min_max.discriptives()
    min_max_sort=min_max_sep.sort_values(by=["Separation_degree"], ascending=False)
    """
    
    """
    # PCAの検討
    F_pca = F.PCA(whiten=True)
    pca_sep = F_pca.discriptives()
    pca_sort=pca_sep.sort_values(by=["Separation_degree"], ascending=False)
    """

    
    # 分類機の検討
    #F = F.MinMaxScaler()
    #F.StratifiedShuffleSplit(train_size=0.5)
    #F.OneClassSVM()
    #print("Logistic Regression\n", F.LogisticRegression(), "\n")
    """
    print("SVC\n", F.SVC(kernel="linear"), "\n")
    print("MLP\n", F.MLP(), "\n")
    print("Kneighbor\n", F.KNeighborsClassifier(n_neighbors=5), "\n")
    print("Random Forest\n", F.RandomForestClassifier(n_estimators=50), "\n")
    """