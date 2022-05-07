'''
Projeto Inicial - Classificação de Sequências
Projeto: reconhecimento de locutor
Autores: Tatiane Balbinot, Wilson Cosmo
Data: 05/05/2022

Script com a finalidade de intalar todas as bibliotecas necessárias para a execução dos códigos principais.
A instalação ocorre por meio do pip.
'''
import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python_speech_features'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'IPython'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'librosa'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'keras'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])

reqs = subprocess.check_output ([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

print(installed_packages)
