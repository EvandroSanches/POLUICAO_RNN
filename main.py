from keras.models import Sequential, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from statistics import stdev, mean
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


#Variaveis de controle
batch_size = 32 # - Define batch de treinamento
epochs = 5 # - Define epocas de treinamento
timesteps = 168 # - Define a quantidade de horas anteriores que serão usados para treino e previsão
horas = 72 # - Define quantos horas a frente serão previstos

def CarregaDados(caminho, mode):
    #Carregando e tratando base de dados
    dados = pd.read_csv(caminho)
    dados = dados.dropna()
    dados = dados.drop('No', axis=1)
    dados = dados.drop('cbwd', axis=1)
    dados = dados.drop('year', axis=1)
    dados = dados.drop('month', axis=1)
    dados = dados.drop('day', axis=1)
    dados = dados.drop('hour', axis=1)

    #Normalizando dados
    normalizador = MinMaxScaler()
    dados_normalizados = normalizador.fit_transform(dados)
    dados = np.asarray(dados)

    #Estruturando dados de treinamento de acordo com as variaveis de controle (batch, timesteps, feature)
    if mode.lower() == 'treinamento':

        previsao = []
        poluicao_real = []
        for i in range(timesteps, dados.shape[0]):
            previsao.append(dados_normalizados[ i - timesteps: i, :])
            poluicao_real.append(dados_normalizados[i, 0])

        previsao = np.asarray(previsao)
        poluicao_real = np.asarray(poluicao_real)
        poluicao_real = np.expand_dims(poluicao_real, axis=1)

        return previsao, poluicao_real

    #Estruturando dados de teste com autoalimentação
    elif mode.lower() == 'previsao':

        try:
            modelo = load_model('Modelo.0.1')
        except:
            print('Treino o modelo primeiro!!!')

        previsao = []
        poluicao_real = []

        for i in range(timesteps, (timesteps+horas)):
            poluicao_real.append(dados[i , 0])
            if i == dados.shape[0]:
                break

        for i in range(timesteps, (timesteps+horas)):
            previsao.append(dados_normalizados[i - timesteps : i, :])
            x = np.asarray(previsao)
            if i == dados_normalizados.shape[0]:
                break
            else:
                result = modelo.predict(x)
                dados_normalizados[i, 0] = result[0]


        previsao = np.asarray(previsao)
        poluicao_real = np.asarray(poluicao_real)
        poluicao_real = np.expand_dims(poluicao_real, axis=1)

        normalizador_previsao = MinMaxScaler()
        poluicao_real = normalizador_previsao.fit_transform(poluicao_real)
        joblib.dump(normalizador_previsao,'normalizador')


        return previsao, poluicao_real

#Criando rede neural
def CriaRede():
    modelo = Sequential()

    modelo.add(LSTM(units=100, return_sequences=True, input_shape = (timesteps, 7)))
    modelo.add(Dropout(0.4))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.4))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.4))

    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.4))

    modelo.add(LSTM(units=80))
    modelo.add(Dropout(0.4))

    modelo.add(Dense(units=1, activation='linear'))

    modelo.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return modelo

def Treinamento():
    #Carregando dados de treino e teste
    previsores, poluicao_real = CarregaDados('poluicao.csv', 'treinamento')
    previsores_teste, poluicao_real_teste = CarregaDados('poluicao_teste.csv', 'treinamento')


    modelo = CriaRede()

    #Definindo Callbacks
    ers = EarlyStopping(monitor='val_loss', patience=2, verbose=1, min_delta= 1e-10 )
    rlp = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1)
    mc = ModelCheckpoint(filepath='Modelo.0.1', save_best_only=True, verbose=1 )

    result = modelo.fit(previsores, poluicao_real, batch_size=batch_size, epochs=epochs, callbacks=[ers, rlp, mc], validation_data=(previsores_teste, poluicao_real_teste) )
    modelo.save('Modelo.0.1')

    #Relatorio de treinamento
    media = mean(result.history['val_mae'])
    desvio = stdev(result.history['val_mae'])

    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('Epocas')
    plt.ylabel('LOSS')
    plt.legend(['Treino', 'Teste'])
    plt.show()

    plt.plot(result.history['mae'])
    plt.plot(result.history['val_mae'])
    plt.title('Relação de perda\nMédia:'+str(media)+'\nDesvio Padrão:'+str(desvio))
    plt.xlabel('Epocas')
    plt.ylabel('Poluição')
    plt.legend(['Treino', 'Teste'])
    plt.show()

def Previsao(caminho, mode):
    #Carregando dados de teste
    previsores, poluicao_real = CarregaDados(caminho, mode)

    modelo = load_model('Modelo.0.1')

    result = modelo.predict(previsores)

    normalizador = joblib.load('normalizador')
    result = normalizador.inverse_transform(result)
    poluicao_real = normalizador.inverse_transform(poluicao_real)

    #Relatório de previsão
    plt.plot(result)
    plt.plot(poluicao_real)
    plt.title('Nivel de Poluição na China')
    plt.xlabel('Horas')
    plt.ylabel('Poluição')
    plt.legend(['Previsão', 'Poluição Real'])
    plt.show()

#Previsao(caminho do arquivo de dados no formato csv, modo de carregamento de dados)
#Modo de carregamento de dados
# - Treinamento : modelo ira carregar dados normaliza-los e estruturar dividindo em variaveis 'previsores' e 'preco_real' para comparação e analise
# - Previsao : modelo ira carregar dados normaliza-los, estruturar com divisão de variaveis 'previsores' e 'preco_real', porém,
# a variavel 'previsores' irá se autoalimentar com as previsões, permitindo prever o preço além dos dados ja conhecidos
# Exemplo: se possuir dados dos ultimos 100 dias no modo treinamento só sera possivel prever o preço do dia 101. No modo
# previsão é possivel prever além
# o modo Treinamento performou muito bem, porém, o modo previsão performou muito mal, a previsão recursiva estabiliza os dados, sendo assim muito dificil
# prever grandes avarias

Previsao('poluicao_teste.csv', 'treinamento')
