'''
Projeto Inicial - Classificação de Sequências
Projeto: reconhecimento de locutor
Autores: Tatiane Balbinot, Wilson Cosmo
Data: 05/05/2022

Script com a finalidade de carregar uma porção específica do 'free spoken digit dataset' dataset
Apenas as intâncias referentes à pronuncia de 'zero' foram carregadas e identificadas de 0 a 5 de acordo com a classe correspondente (locutor)
'''
import urllib.request
import os

base_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/blob/master/recordings/"
end_url = ".wav?raw=true"
n = 0
c1 = "0_george_"
c2 = "0_jackson_"
c3 = "0_lucas_"
c4 = "0_nicolas_"
c5 = "0_theo_"
c6 = "0_yweweler_"

try:
    os.mkdir("dataset")
    print("Diretório 'dataset' criado.")
except OSError as error:
    print(error)

print("Dataset original: ", base_url)
print("Iniciando o download do dataset.")

for n in range(50):
    url_t = base_url+c1+str(n)+end_url
    urllib.request.urlretrieve(url_t, "./dataset/0_george_"+str(n)+".wav")

print("'george' carregado.")

for n in range(50):
    url_t = base_url+c2+str(n)+end_url
    urllib.request.urlretrieve(url_t, "./dataset/1_jackson_"+str(n)+".wav")

print("'jackson' carregado.")


for n in range(50):
    url_t = base_url+c3+str(n)+end_url
    urllib.request.urlretrieve(url_t, "./dataset/2_lucas_"+str(n)+".wav")

print("'lucas' carregado.")

for n in range(50):
    url_t = base_url+c4+str(n)+end_url
    urllib.request.urlretrieve(url_t, "./dataset/3_nicolas_"+str(n)+".wav")

print("'nicolas' carregado.")

for n in range(50):
    url_t = base_url+c5+str(n)+end_url
    urllib.request.urlretrieve(url_t, "./dataset/4_theo_"+str(n)+".wav")

print("'theo' carregado.")

for n in range(50):
    url_t = base_url+c6+str(n)+end_url
    urllib.request.urlretrieve(url_t, "./dataset/5_yweweler_"+str(n)+".wav")

print("'yweweler' carregado.")
print("Todos os arquivos carregados.")
