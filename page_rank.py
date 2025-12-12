import numpy as np

def get_transition_matrix(W):
    """Returns transition matrix (P) based on graph weighted, adjacency matrix

    `P[i][j]` - the probability to move from node `i` to node `j`

    Args:
        W (`np.matrix`):
            an adjacency matrix of a directed, weighted, regular graph G
    Returns:
        P (`np.array`):
            transition matrix of `W`
    """
    P = np.zeros(W.shape)
    for i in range(W.shape[0]):
        w_i = W[i].sum()

        if w_i == 0: # dangling node
            P[i, i] = 1.
        else:
            for j in range(W.shape[1]):
                w_ij = W[i, j]
                P[i, j] = w_ij / w_i

    return P

def pageRankLinear(A, alpha, v):
    """Returns the PageRank scores by solving the linear system equation

    By solving equation of Markov chain: `x^T A = x^T`
    Transition matrix on vector must remain the same.
    That means that transition matrix became constant, stabilized-converged

    By placing Google matrix in this equation, and developing it
    `G = aP + (1-a)ev^T`

    <=> x^T (aP+(1-a)ev^T) = x^T
    <=> ax^TP + (1-a)x^Tev^T = x^T
    <=> x^T - ax^TP = (1-a)v^T # because x^T * e = 1
    <=> x^T(I - aP) = (1-a)v^T # because x^T * I = x^T | shapes: (1,n) x (n,n) = (1,n)
    <=> (I - aP)^Tx = (1-a)v # just get transpose on both sides

    Args:
        A (`np.matrix`):
            an adjacency matrix of a directed, weighted, regular graph G
        alpha (`float`):
            a teleportation parameter between [0,1]
        v (`np.array`):
            a personalization vector
    Returns:
        page_rank_scores (`np.array`): scores of personalized PageRank algorithm
                                       it's an exact solution, and not approximation
    """

    # get transform matrix
    P = get_transition_matrix(A)
    I = np.eye(A.shape[0])

    coeff_matrix = (I - alpha * P).T
    b_vector = (1 - alpha) * v

    # because of very deep math explanation, such equation has always solution
    scores = np.linalg.solve(coeff_matrix, b_vector)
    return scores

def pageRankPower(A, alpha, v):
    """
    Args:
        A (`np.matrix`):
            an adjacency matrix of a directed, weighted, regular graph G
        alpha (`float`):
            a teleportation parameter between [0,1]
        v (`np.array`):
            a personalization vector
    Returns:
        page_rank_scores (`np.array`): scores of personalized PageRank algorithm via power method
    """

    return pageRankLinear(A, alpha, v)

def randomWalk(A, alpha, v):
    """
    Args:
        A (`np.matrix`):
            an adjacency matrix of a directed, weighted, regular graph G
        alpha (`float`):
            a teleportation parameter between [0,1]
        v (`np.array`):
            a personalization vector
    Returns:
        page_rank_scores (`np.array`): scores of personalized PageRank algorithm via power method
    """

    return pageRankLinear(A, alpha, v)
