from page_rank import *

import matplotlib.pyplot as plt
import numpy as np

def main():
    #Matrice d'adjacence donnée (on ordonne les liens dans l'ordre alphabétique):
    matrix = []
    with open("MatriceAdjacence.csv") as file_matrice :
        for line in file_matrice.readlines():
            line = line.strip().split(",")
            matrix.append([float(val) for val in line])

    A = np.array(matrix)

    #Construire le vecteur de personnalisation
    with open("VecteurPersonnalisation_Groupe2.csv") as file:
        file.readline()
        valeurs = file.readline()
    valeurs = valeurs.strip().split(',')
    valeurs = [float(val) for val in valeurs]
    
    v = np.array(valeurs)

    #======Le code commence ici======
    alpha = 0.9
    v1 = pageRankLinear(A, alpha,v)
    v2 = pageRankPower(A, alpha, v)
    v3 = randomWalk(A,alpha, v)
    

if __name__ == '__main__':
    main()
