import numpy as np

class ponto_instancia:
    def __init__(self,x,y, linha):
        self.x = x
        self.y = y
        self.x_network = x
        self.y_network = linha - y - 1
        self.estante = 0
        self.corredor = 0
        self.reward_est = 0

def mundo_16x16():
    # GRADE 16 X 16
    linha = 16
    coluna = 16
    mundo = [[ponto_instancia(x,y,linha) for y in range(linha)] for x in range(coluna)]
    # Adicionando corredores e estantes sem pontuação
    for i in range(coluna):
        for j in range(linha):
            if i%2 != 0:
                mundo[i][j].estante = 1
                mundo[i][j].reward_est = 0
                mundo[i][j].corredor = 0
            else:
                mundo[i][j].estante = 0
                mundo[i][j].reward_est = 0
                mundo[i][j].corredor = 1

    # Adicionando corredores horizontais
    linha_corredor = [0, 5, 10, 15]
    for j in linha_corredor:
        for i in range(coluna):
                mundo[i][j].estante = 0
                mundo[i][j].reward_est = 0
                mundo[i][j].corredor = 1

    # Colocando estantes desejadas
    e_i = [1,1,1,1,1,3,3,3,3,3,3,3,3,3,5,5,5,5,5,7,7,7,7,7,7,7,7,9,9,9,9,9,9,11,11,11,11,11,11,11,11,11,13,13,13,13,13,13,13,13,15,15,15,15,15]
    e_j = [1,3,4,12,14,1,4,6,7,9,11,12,13,14,1,4,6,13,14,1,2,4,6,8,9,12,13,6,8,11,12,13,14,1,2,4,6,7,8,9,12,14,1,2,3,8,9,11,12,14,1,6,8,12,14]
    for k in range(len(e_i)):
        i = e_i[k]
        j = e_j[k]
        mundo[i][j].estante = 1
        mundo[i][j].reward_est = 1
        mundo[i][j].corredor = 0
    return mundo

def mundo_21x17():
    # GRADE 16 X 16
    linha = 21
    coluna = 17
    mundo = [[ponto_instancia(x,y,linha) for y in range(linha)] for x in range(coluna)]
    # Adicionando corredores e estantes sem pontuação
    for i in range(coluna):
        for j in range(linha):
            if i%2 != 0:
                mundo[i][j].estante = 1
                mundo[i][j].reward_est = 0
                mundo[i][j].corredor = 0
            else:
                mundo[i][j].estante = 0
                mundo[i][j].reward_est = 0
                mundo[i][j].corredor = 1

    # Adicionando corredores horizontais
    linha_corredor = [0, 5, 10, 15]
    for j in linha_corredor:
        for i in range(coluna):
                mundo[i][j].estante = 0
                mundo[i][j].reward_est = 0
                mundo[i][j].corredor = 1

    # Colocando estantes desejadas
    e_j = [2,6,7,8,9,11,14,17,19,1,12,13,16,17,19,1,2,4,6,7,8,9,13,14,16,17,18,1,2,3,4,11,12,13,19,1,2,12,14,16,17,18,2,4,6,8,11,13,16,19,1,3,4,6,7,11,12,13,17,2,3,7,11,12,13,17,18]
    e_i = [1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7,9,9,9,9,9,9,9,11,11,11,11,11,11,11,11,13,13,13,13,13,13,13,13,13,15,15,15,15,15,15,15,15]
    for k in range(len(e_i)):
        i = e_i[k]
        j = e_j[k]
        mundo[i][j].estante = 1
        mundo[i][j].reward_est = 1
        mundo[i][j].corredor = 0
    return mundo