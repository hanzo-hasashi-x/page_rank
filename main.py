from page_rank import *

import matplotlib.pyplot as plt
import numpy as np


def randomWalkSimulation(A, alpha, v, steps_num=10_000):
    """Return PageRank scores and outputting the graph of mean error evolution

    Args:
        A (`np.matrix`):
            an adjacency matrix of a directed, weighted, regular graph G
        alpha (`float`):
            a teleportation parameter between [0,1]
        v (`np.array`):
            a personalization vector
    Returns:
        page_rank_scores (`np.array`): 
            scores of personalized PageRank algorithm via random walk
    """
    errors = []

    P = get_transition_matrix(A)
    exactPageRank = pageRankLinear(A, alpha, v)

    random_walker = randomWalker(A, alpha, v)

    for _ in range(steps_num):
        scores = next(random_walker)

        mean_error = 1 / len(P) * np.abs(scores - exactPageRank).sum()
        errors.append(mean_error)

    plt.semilogy(errors) # log scaled graph
    plt.show()

    return scores

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
    
   
    
