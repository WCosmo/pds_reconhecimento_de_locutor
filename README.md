# PDS - Reconhecimento de locutor
Projeto Inicial - Classificação de Sequências, reconhecimento de locutor
- Autores: Tatiane Balbinot, Wilson Cosmo
- Baseado no artigo: https://www.gosmar.eu/machinelearning/2020/05/25/neural-networks-and-speech-recognition/

Como realizar os testes:
- Executar "s_imports.py" para instalar todas as bibliotecas presentes no algoritmo;
- Executar "s_dataset.py" para carregar a porção do "free spoken digit dataset" nescessária para a atual abordagem;
- Será carregado 300 instâncias correspondentes a 6 pessoas pronuciando a palavra "zero";
- Os arquivos foram renomeados adicionando um dígito de 0 a 5 no início de cada arquivo, que servirá para identificar o locutor
- Codificação: 0 = 'george', 1 = 'jackson', 2 = 'lucas', 3 = 'nicolas', 4 = 'theo', 5 = 'yweweler'
- Dataset original completo: https://github.com/Jakobovski/free-spoken-digit-dataset

Sobre os algoritmos de teste: 
- O script "c_locutor_zero_padding.py" compara as instâncias realizando zero padding com base no tamanho da maior instância presente;
- O script "c_locutor_truncation.py" compara as instâncias realizando truncation com base no tamanho da menor instância presente;
- Ambos geram gráficos de Acurácia, Erro e Matriz de Confusão para visualização e comparação dos resultados;
- O script "graphic_plot.py" tem como propósito somente a visualização de algumas instâncias (2 de cada classe);
- São plotados a instância ao longo do tempo, o espectrograma e uma visualização dos seus MFCC. 

