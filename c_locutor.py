'''
Projeto Inicial - Classificação de Sequências
Projeto: reconhecimento de locutor
Autores: Tatiane Balbinot, Wilson Cosmo
Data: 22/04/2022

Baseado no artigo: https://www.gosmar.eu/machinelearning/2020/05/25/neural-networks-and-speech-recognition/
Adaptado para o Dataset: https://github.com/Jakobovski/free-spoken-digit-dataset
Foram utilizados apenas as 300 primeiras amostras do dataset, correspondentes aos 6 locutores pronuciando o dígito "zero" em inglês.
Os arquivos foram renomeados adicionando um dígito de 0 a 5 no início de cada arquivo, que servirá para identificar o locutor:
Codificação: 0 = 'george', 1 = 'jackson', 2 = 'lucas', 3 = 'nicolas', 4 = 'theo', 5 = 'yweweler'
'''

#início dos imports:
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import HTML
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
import scipy
from python_speech_features import mfcc
from python_speech_features import logfbank
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, LSTM
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.optimizers import gradient_descent_v2
#fim dos imports

def rel(r): #função que associa o numero ao locutor correspondente
    n = '-'
    if r == 0:
        n = 'george'
    elif r == 1:
        n = 'jackson'
    elif r == 2:
        n = 'lucas'
    elif r == 3:
        n = 'nicolas'
    elif r == 4:
        n = 'theo'
    else:
        n = 'yweweler'
    return n

soundfile = os.listdir('./amostras') #lê os nomes dos arquivos de um subdiretório /amostras
data=[]
for i in soundfile:
    (rate,sig) = wav.read('./amostras/'+i) #carrega as informações de cada arquivo .wav
    data.append(sig)

size = 48
X=[] #variavel de armazenamento dos valores MFCC das amostras
for i in range(len(data)): #aquisição dos valores MFCC para cada amostra
    mfcc_feat = mfcc(data[i],rate,nfft=512)
    mfcc_feat = np.resize(mfcc_feat, (size,13))
    X.append(mfcc_feat)
X = np.array(X) #formatação de X no formato matriz np

#set the target label
y = [i[0] for i in soundfile] #definição dos locutores de acordo com o primeiro dígito no nome do arquivo
Y = pd.get_dummies(y) #conversão da variável categórica para variável indicador (pandas)
Y = np.array(Y) #formatação de Y no formato matriz np

model = Sequential() #modelo da rede
#Camadas de convolução
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(size, 13,1)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
#Flattening
model.add(Flatten(input_shape=(size, 13,1)))
#1st fully connected Neural Network hidden-layer
model.add(Dense(64))
model.add(Dropout(0.16))
model.add(Activation('relu'))
#2nd fully connected Neural Network hidden-layer
model.add(Dense(64))
model.add(Dropout(0.12))
model.add(Activation('relu'))
#Output layer
model.add(Dense(6))
model.add(Activation('softmax'))
model.summary()

sgd = gradient_descent_v2.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True) #otimizador para a compilação do modelo
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) #compilação do modelo
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25) #divisão do dataset em treino e teste, 75% para treino e 25% para teste
x_train = x_train.reshape(-1, size, 13, 1)
x_test = x_test.reshape(-1, size, 13, 1)
history = model.fit(x_train, y_train, epochs=18, batch_size=32, validation_split=0.2, shuffle=True) #treinamento do modelo
model.evaluate(x_test, y_test, verbose=2) #avaliação do modelo

nn = np.arange(0, 75) #vetor auxiliar para plotagem de gráfico, representa o número de amostras
ny = np.arange(0, 75) #vetor auxiliar para plotagem de gráfico, representa os valores reais do dataset de treinamento
for n1 in nn:
    max_index = np.argmax(y_test[n1])
    ny[n1] = max_index
plt.plot(ny, linestyle = 'dotted', color = 'b') #os valores reais são plotados como uma linha azul pontilhada

for n1 in nn:
    pred = model.predict(x_test[int(n1)].reshape(-1,size,13,1))
    plt.plot(n1, pred.argmax(), 'o', color = 'r') #os valores previstos pela rede são representados no gráfico como pontos vermelhos

plt.title("Comparação dos valores reais com os valores previstos") #título do gráfico
plt.xlabel("Número de amostras (conjunto de teste)")
plt.ylabel("Valores reais (-) x Predições da rede (o)")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show() #exibição do gráfico

#início do modo de teste manual
print('\n\n------------------------------------------------------------')
print('Modo de teste manual')
aa = ''
while aa != 'n': #o usuário insere uma posição do conjunto de treinamento e visualiza o seu valor real e o previsto pela rede
    print('------------------------------------------------------------')
    print('\nInsira uma posição do conjunto de treinamento (0-74): ')
    ii = input()
    y_test[int(ii)]
    y_rv = np.argmax(y_test[int(ii)])
    print("\nValor real: ", y_rv, ", locutor real: ", rel(y_rv))
    pred = model.predict(x_test[int(ii)].reshape(-1,size,13,1))
    print("\nValor previsto: ", pred.argmax(), ", locutor previsto: ", rel(pred.argmax()))
    print("\nMatriz de probabilidade:\n")
    for k in range(6): #exibição da matriz de probabilidade gerada pela rede para estimar o valor
        print(pred[0,k])

    print('\n\nRealizar novo teste? (n para cancelar): ')
    aa = input() #o programa encerra ao digitar 'n'
