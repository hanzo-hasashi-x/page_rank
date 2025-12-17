import numpy as np
def get_transition_matrix(A):
    n = A.shape[0]
    P = np.zeros(A.shape, dtype=float)

    for i in range(n):
        w_i = A[i].sum()

        if w_i == 0:  # dangling node
            for j in range(n):
                P[i, j] = 1 / n
        else:
            for j in range(n):
                P[i, j] = A[i, j] / w_i

    return P
def get_google_matrix(P, alpha, v):
    n = P.shape[0]
    return alpha * P + (1 - alpha) * np.outer(np.ones(n), v)


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

def pageRankPower(A: np.array, alpha: float, v: np.array):
    """
    Args:
        A (np.array):
            adjacency matrix
        alpha (float):
            teleportation parameter in [0,1]
        v (np.array):
            personalization vector (sum = 1)
    Returns:
        np.array: PageRank scores
    """

    n = A.shape[0]
    P = get_transition_matrix(A)
    G = get_google_matrix(P, alpha, v)

    # initial vector
    x = np.full(n, 1 / n)

    epsilon = 1e-8
    max_iter = 1000

    for _ in range(max_iter):
        x_new = G.T @ x
        # norme L1 (classique pour PageRank)
        diff = np.linalg.norm(x_new - x, 1)
        if diff < epsilon:
            break
        x = x_new

    return x

        



   

def randomWalker(A, alpha, v):
    """Returns generator, that at every step returns current PageRank scores

    Args:
        A (`np.matrix`):
            an adjacency matrix of a directed, weighted, regular graph G
        alpha (`float`):
            a teleportation parameter between [0,1]
        v (`np.array`):
            a personalization vector
    Yields:
        page_rank_scores (`np.array`): 
            scores of personalized PageRank algorithm via random walk simulation
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

def randomWalk(A, alpha, v):
    """Returns approximative PageRank scores by simulating random walker for 10_000 steps

    Args:
        A (`np.matrix`):
            an adjacency matrix of a directed, weighted, regular graph G
        alpha (`float`):
            a teleportation parameter between [0,1]
        v (`np.array`):
            a personalization vector
    Returns:
        page_rank_scores (`np.array`): 
            final scores of personalized PageRank algorithm via random walk of 10_000 steps
    """
    steps_num = 1000_000
    random_walker = randomWalker(A, alpha, v)

    for _ in range(steps_num):
        scores = next(random_walker)

    return scores
