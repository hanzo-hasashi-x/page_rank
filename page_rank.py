import numpy as np
import matplotlib.pyplot as plt


def get_transition_matrix(A : np.array) -> np.array:
    """
    Retourne la matrice de probabilité de transition P d'après la matrice d'adjacence A, avec
    `P[i][j]` - La probabilité de se déplacer du noeud `i` vers le noeud `j` <=> w_ij/w_i. 

    Args:
        A (`np.matrix`):
            une matrice d'adjacence d'un graphe dirigé, pondéré et simple
    Returns:
        P (`np.array`):
            la matrice de probabilité de transition associée au graphe décrit par A.
    """
    n = A.shape[0]
    P = np.zeros(A.shape, dtype=float)

    for i in range(n):
        w_i = A[i].sum()

        if w_i == 0:  # dangling node 
            #(noeud pendant; dans ce cas on attribue une probabilité uniforme de se téléporter ailleurs)
            for j in range(n):
                P[i, j] = 1 / n
        else:
            for j in range(n):
                P[i, j] = A[i, j] / w_i

    return P

def get_google_matrix(P : np.array, alpha : float , v : np.array) -> np.array:
    """
    Retourne la matrice Google, alpha, et vecteur de personnalisation, avec
    `G[i][j]` - La probabilité de se déplacer d'un noeud `i` vers un noeud `j`,
                en prenant en compte la personnalisation 

    Args:
        P (`np.matrix`):
            une matrice de probabilité de transition d'un graphe simple, dirigé et pondéré
        alpha (`float`):
            le paramètre de téléportaion entre [0,1]
        v (`np.array`):
            le vecteur de personnalisation
    Returns:
        G (`np.array`):
            La matrice Google correspondante
    """

    n = P.shape[0]
    return alpha * P + (1 - alpha) * np.outer(np.ones(n), v)

def pageRankLinear(A : np.array, alpha : float, v : np.array, show=True ) -> np.array:
    """Retourne les scores PageRank en résolvant le système linéaire.
    
    Pour alpha < 1, résout `(I - alpha*P)^T * x = (1 - alpha) * v`.
    Pour alpha = 1, résout le système `(I - P)^T * x = 0` sous la contrainte `e^Tx = 1`.
    Args:
        A (`np.matrix`):
            une matrice d'adjacence d'un graphe simple, dirigé et pondéré
        alpha (`float`):
            le paramètre de téléportation entre [0,1]
        v (`np.array`):
            le vecteur de personnalisation
        show (`bool`):
           par défaut est `True`, affiche ou non les scores.

    Returns:
        scores (`np.array`):
            les scores PageRank personnalisés exacts.
          

    """
    P = get_transition_matrix(A) 
    n = A.shape[0]
    I = np.eye(n)

    if alpha != 1:
        # Système standard : (I - alpha*P)^T * x = (1 - alpha) * v
        coeff_matrix = (I - alpha * P).T 
        b_vector = (1 - alpha) * v
        
        scores = np.linalg.solve(coeff_matrix, b_vector)
        
    else:
        # Cas alpha = 1 : Système (I - P)^T * x = 0 et sum(x) = 1
        coeff_matrix = (I - P).T
        
        # On ajoute la ligne de contrainte [1, 1, ..., 1] à la position 0
        e = np.ones(n)
        A_system = np.insert(coeff_matrix, 0, e, axis=0) #
        
        #Le vecteur cible, càd [1, 0, 0, ...]
        b_system = np.zeros(n + 1)
        b_system[0] = 1
        
        #Vu qu'on a une matrice rectangle, on utilise cette fonction qui renvoie un tuple compliqué (x, residuals, rank, s).
        scores, _, _, _ = np.linalg.lstsq(A_system, b_system, rcond=None)

    if show:
        print("===========================")
        print("[SYSTEME LINEAIRE] Vecteur de scores PageRank final :\n", scores)
    return scores

def pageRankPower(A: np.array, alpha:float, v: np.array) -> np.array:
    """
    Args:
        A (`np.matrix`):
            une matrice d'adjacence d'un graphe simple, dirigé et pondéré
        alpha (`float`):
            le paramètre de téléportation entre [0,1]
        v (`np.array`):
            le vecteur de personnalisation
    Returns:
        scores (`np.array`):
            les scores PageRank personnalisés approximés (exacts à 1e-8).
    """
 
    n = A.shape[0]
    P = get_transition_matrix(A)
    G = get_google_matrix(P,alpha, v)
    print("[POWER METHOD]")
    print("===========================")
    print("Matrice d'adjacence : \n", A )
    print("===========================")
    print("Matrice de probabilité de transition : \n", P)
    print("===========================")
    print("Matrice Google : \n", G)
    print("===========================")

    # initial vector
    x = np.full(n, 1 / n)

    epsilon = 1e-8
    max_iter = 1000
    x_new = x.T
    for i in range(max_iter):
        if i < 4 :
            print("Vecteur x à l'itération", i, ':\n', x)
        x_new = x@G 
        # norme L1 (classique pour PageRank)
        diff = np.linalg.norm(x_new - x, 1)
        if diff < epsilon:
            it = i
            break
        x = x_new
    print('\n')
    print("[POWER METHOD] Vecteur de scores PageRank final (obtenu après", it," itérations) :\n", x)
    return x
   
def randomWalker(A:np.array, alpha:float, v:np.array):

    """Retourne un generator, qui à chaque étape retourne les scores PageRank actuels
    Args:
        A (`np.matrix`):
            une matrice d'adjacence d'un graphe simple, dirigé et pondéré
        alpha (`float`):
            le paramètre de téléportation entre [0,1]
        v (`np.array`):
            le vecteur de personnalisation
    Yields:
        page_rank_scores (`np.array`): 
            scores PageRank personnalisés via la simulation de marche aléatoire
        """
    step = 0
    curr_node = 0

    P = get_transition_matrix(A)
    scores = np.zeros(v.shape, dtype=float)

    v = v[..., np.newaxis]
    G = alpha * P + (1-alpha) * np.ones(v.shape) @ v.T

    G_F = G.cumsum(1) # vector of partition functions to pass to other nodes
    while True:
        # get next random chosen node
        prob = np.random.rand() # normal distribution sample
        curr_node = np.where(G_F[curr_node] > prob)[0][0]

        # step update
        scores[curr_node] += 1
        step += 1

        yield scores / step # returns scores such as their sum = 1

def randomWalk(A:np.array, alpha:float, v:np.array) -> np.array:
    """Return PageRank scores and outputting the graph of mean error evolution
    Args:
        A (`np.matrix`):
            une matrice d'adjacence d'un graphe simple, dirigé et pondéré
        alpha (`float`):
            le paramètre de téléportation entre [0,1]
        v (`np.array`):
            le vecteur de personnalisation
    Returns:
        scores (`np.array`):
            les scores PageRank personnalisés approximés avec la simulation de marche aléatoire.
    """
    steps_num = 100000
    P = get_transition_matrix(A)
    exactPageRank = pageRankLinear(A, alpha, v, False)

    random_walker = randomWalker(A,alpha, v )

    errors = []
    for _ in range(steps_num):
        scores = next(random_walker)

        mean_error = 1 / len(P) * np.abs(scores - exactPageRank).sum()
        errors.append(mean_error)

    print("===========================")
    print("[RANDOM WALK] Vecteur de scores approximés PageRank final", scores)

    plt.semilogy(errors)
    plt.xlabel("Steps")
    plt.ylabel("Mean error (log scale)")
    plt.title("Random Walk PageRank convergence")
    plt.show()
   
    return scores
