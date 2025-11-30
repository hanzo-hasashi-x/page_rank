import numpy as np

def get_transform_matrix(W):
    """ - the probability to move from node `i` to node `j`

    Args:
        W (`np.matrix`):
            an adjacency matrix of a directed, weighted, regular graph G
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
    P = get_transform_matrix(A)

    # from Google matrix: G = aP + (1-a)ev.T

    coeff_matrix = (np.eye(A.shape[0]) - alpha * P).T
    b_vector = (1 - alpha) * v

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
