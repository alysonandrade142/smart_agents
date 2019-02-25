# -*- coding: utf-8 -*-
import numpy as np
from time import sleep, time
from math import sqrt
import matplotlib.pyplot as plt

class SOM:
    wNodes = None  # Peso dos Nos

    alpha0 = None  # Taxa de Aprendizado
    sigma0 = None  # Raio
    dataIn = None  # Dados
    grid = None

    def __init__(self, dataIn, grid=[10, 10], alpha=0.1, sigma=None):
        dim = dataIn.shape[1]
        self.wNodes = np.random.uniform (-1, 1, [grid[0], grid[1], dim])
        plt.scatter (self.wNodes[0], self.wNodes[1])
        plt.show ()
        self.alpha0 = alpha
        if (sigma is None):
            self.sigma0 = max (grid) / 2.0
        else:
            self.sigma0 = sigma

        self.dataIn = np.asarray (dataIn)
        self.grid = grid

    def train(self, maxIt=100, verbose=True, analysis=False, timeSleep=0.5):
        nSamples = self.dataIn.shape[0]
        m = self.wNodes.shape[0]  # linha
        n = self.wNodes.shape[1]  # coluna

        # Processamento (TEMPO)
        timeCte = (maxIt / np.log (self.sigma0))
        if analysis:
            print ('timeCte = ', timeCte)

        timeInit = 0
        timeEnd = 0

        for epc in range (maxIt):
            alpha = self.alpha0 * np.exp (-epc / timeCte)
            sigma = self.sigma0 * np.exp (-epc / timeCte)

            if verbose:
                print ('Epoca: ', epc, ' - Tempo de Processamento: ', (timeEnd - timeInit) * (maxIt - epc), ' seg')

            timeInit = time ()

            for k in range (nSamples):

                # Nó Vencedor
                matDist = self.distance (self.dataIn[k, :], self.wNodes)

                posWin = self.getWinNodePos (matDist)

                deltaW = 0
                h = 0

                for i in range (m):
                    for j in range (n):
                        # Distância Entre Dois Nos
                        dNode = self.getDistanceNodes ([i, j], posWin)

                        # Região de Vizinhança
                        h = np.exp ((-dNode ** 2) / (2 * sigma ** 2))

                        # Atualizando os Pesos
                        deltaW = (alpha * h * (self.dataIn[k, :] - self.wNodes[i, j, :]))
                        self.wNodes[i, j, :] += deltaW

                        if analysis:
                            print ('Epoca = ', epc)
                            print ('Amostra = ', k)
                            print ('-------------------------------')
                            print ('alpha = ', alpha)
                            print ('sigma = ', sigma)
                            print ('h = ', h)
                            print ('-------------------------------')
                            print ('No vencedor = [', posWin[0], ', ', posWin[1], ']')
                            print ('No atual = [', i, ', ', j, ']')
                            print ('Distancia entre Nos = ', dNode)
                            print ('deltaW = ', deltaW)
                            print ('wNode antes = ', self.wNodes[i, j, :])
                            print ('wNode depois = ', self.wNodes[i, j, :] + deltaW)
                            print ('\n')
                            sleep (timeSleep)

            timeEnd = time ()

    # Método para calcular a distância entre a entrada e seus pesos
    def distance(self, a, b):
        return np.sqrt (np.sum ((a - b) ** 2, 2, keepdims=True))

    # Método que retorna a distância entre dois nós

    def getDistanceNodes(self, n1, n2):
        n1 = np.asarray (n1)
        n2 = np.asarray (n2)
        return np.sqrt (np.sum ((n1 - n2) ** 2))

    # Método que retorna a posição do Nó vencedor
    def getWinNodePos(self, dists):
        arg = dists.argmin ()
        m = dists.shape[0]
        return arg // m, arg % m

    # Método que retornar o centróide dos dados de entrada
    def getCentroid(self, data):
        data = np.asarray (data)
        N = data.shape[0]
        centroids = list ()

        for k in range (N):
            matDist = self.distance (data[k, :], self.wNodes)
            centroids.append (self.getWinNodePos (matDist))

        return centroids

    # Método para salvar os Pesos
    def saveTrainedSOM(self, fileName='SOMTreinado.csv'):
        np.savetxt (fileName, self.wNodes)

    # Método para da um load nos Pesos já treinado - Utilizar para Teste
    def setTrainedSOM(self, fileName):
        self.wNodes = np.loadtxt (fileName)


import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

data = [['Bom Dia como você está?', 'Eu também estou.',
         'Olá', 'Como vai você?', 'Eu estou bem'],
        ['Eu estou bem, e você?', 'Que bom.',
         'Olá.', 'Bem, e você?', 'Legal.']]

entrada = [['Bom Dia como você está?', 'Eu também estou.',
            'Olá', 'Como vai você?', 'Eu estou bem'],
           ['Eu estou bem, e você?', 'Que bom.',
            'Olá.', 'Bem, e você?', 'Legal.']]

result = []
t1 = []
t2 = []
vectorizer = HashingVectorizer(n_features=5)

for i in range (len (entrada) - 1):

    for j in range (len (entrada[i])):
        t1 = data[i][j] + data[i + 1][j]
        t2 = entrada[i][j] + entrada[i + 1][j]
        vector1 = vectorizer.transform ([t1]).toarray()
        vector2 = vectorizer.transform ([t2]).toarray()
        distance = 0.0
        for k in range (len (vector1)):
            distance += (vector1[k] - vector2[k]) ** 2
            result.append(str(sqrt(sum(distance))))
            print(distance)
        # print(fuzz.token_sort_ratio (t1, t2))
    i += 1
entrada.append (result)
print(entrada)

resData = np.asarray (entrada)
print(resData)

vector = vectorizer.transform(resData.flatten())

# summarize encoded vector
# print(vector.shape)
# print(vector.toarray())

s = SOM (vector.toarray (), [20, 30], alpha=0.3)
#plt.imshow (s.wNodes)

s.train (maxIt=30)

plt.scatter (s.wNodes[0], s.wNodes[1])
plt.show ()
