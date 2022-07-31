'''
Projeto Inicial - Classificação de Sequências
Projeto: reconhecimento de locutor
Autores: Tatiane Balbinot, Wilson Cosmo
Data: 22/04/2022

Baseado no artigo: https://www.gosmar.eu/machinelearning/2020/05/25/neural-networks-and-speech-recognition/
Adaptado para o Dataset: https://github.com/Jakobovski/free-spoken-digit-dataset
Esse script é somente para plotar os gráficos de 12 amostras (2 de cada classe) e suas respectivas visualizações dos MFCC
'''
#início dos imports:
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy import signal
import scipy
from python_speech_features import mfcc
from python_speech_features import logfbank
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#fim dos imports

soundfile = os.listdir('./dataset') #lê os nomes dos arquivos de um subdiretório /dataset
data=[] #variável para armazenar as instâncias
for i in soundfile:
    (rate,sig) = wav.read('./dataset/'+i) #carrega as informações de cada arquivo .wav
    data.append(sig)

na = len(data) #numero total de instâncias
nc = 6 #numero de classes
tt = '' #variável para gerar o título dos gráficos
for bb in range(nc*2):
    plt.plot(data[bb*25]) #plotagem de 12 instâncias
    tt = 'Visualização da instância número #' + str((bb*25)+1)
    plt.title(tt)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.show()

X=[] #variavel de armazenamento dos valores MFCC das instâncias
for i in range(na): #aquisição dos valores MFCC para cada instâncias
    mfcc_feat = mfcc(data[i],rate,nfft=512)
    X.append(mfcc_feat)
X = np.array(X) #formatação de X no formato matriz np

for bb in range(nc*2): #plotagem do mfcc de 12 instâncias
    ig, ax = plt.subplots()
    mfcc_data= np.swapaxes(X[bb*25], 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
    tt = 'Visualização dos MFCC da instância número #' + str((bb*25)+1)
    ax.set_title(tt)
    plt.show()
