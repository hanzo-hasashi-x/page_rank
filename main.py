from page_rank import *

import matplotlib.pyplot as plt
import numpy as np


def randomWalkSimulation(A, alpha, v, steps_num=10_000, walkers=50):
    errors = []

    P = get_transition_matrix(A)
    exactPageRank = pageRankLinear(A, alpha, v)

    n = A.shape[0]
    avg_scores = np.zeros(n)

    walkers_gen = [randomWalker(A, alpha, v) for _ in range(walkers)]

    for step in range(steps_num):
        for w in walkers_gen:
            avg_scores += next(w)

        avg_scores /= walkers

        mean_error = np.linalg.norm(avg_scores - exactPageRank, 1) / n
        errors.append(mean_error)

    plt.semilogy(errors)
    plt.xlabel("Steps")
    plt.ylabel("Mean error (log scale)")
    plt.title("Random Walk PageRank convergence")
    plt.show()

    return avg_scores

if __name__ == '__main__':
    #Matrice d'adjacence donnée (on ordonne les liens dans l'ordre alphabétique):
    A = np.array([[0,5,0,0,0,0,0,3,0,0],
                  [3,0,1,0,0,0,0,0,2,0],
                  [0,0,0,2,0,0,0,0,5,3],
                  [0,0,3,0,0,0,0,0,0,3],
                  [0,0,0,5,0,4,0,0,0,0],
                  [0,0,0,0,2,0,5,0,0,0],
                  [0,0,0,0,0,2,0,0,3,0],
                  [0,0,0,0,0,0,2,0,0,0],
                  [1,4,0,0,0,0,0,4,0,4],
                  [0,0,0,0,4,1,0,0,2,0]])
    
    #Construire le vecteur de personnalisation
    with  open("VecteurPersonnalisation_Groupe2.csv") as file:
        file.readline()
        valeurs = file.readline()
    valeurs = valeurs.strip().split(',')
    valeurs = [float(val) for val in valeurs]
    v = np.array(valeurs)

    print(pageRankLinear(A,0.7,v))
    print(pageRankPower(A, 0.7, v))
    print(randomWalkSimulation(A, 0.7, v))
   
    
