import numpy as np
import networkx as nx
from pandas import DataFrame
import sys


def _special_scaler(inv_dist_matrix, allow_neighbours=True):
    """Scales a matrix to unit variance and zero mean,
    optionally ignoring neighbour values"""

    non_diag_vals = np.tril_indices_from(inv_dist_matrix, k=-1)
    mat_vals = inv_dist_matrix.values[non_diag_vals]

    if not allow_neighbours:
        mat_vals = mat_vals[mat_vals != mat_vals.max()]
        # This is for masking the neighbours.
        # The maximum values not along the diagonal are by definition
        # the neighbours.

    mean_vals = mat_vals.mean()
    std_vals = mat_vals.std()
    # We exclude the diagonal when calculating mean and std

    inv_dist_matrix = (inv_dist_matrix - mean_vals) / std_vals
    return inv_dist_matrix


def _calc_pairing_cost(cost_matrix, pairings):
    """Utility function to calculate the sum of all pairs"""
    return sum(cost_matrix.values[i, j] for i, j in pairings)


def _make_pairs_common(optimization_func, collaborator_network,
                       knowledge_matrix, output_fh, alpha, allow_neighbours,
                       n_rounds, update_network):
    """Closure which calculates the pairs for a given round.  Can use
    with different optimization_func that calculate the maximum
    bipartite graph"""
    if output_fh is None:
        output_fh = sys.stdout

    all_iter_couples = []

    if output_fh:
        output_fh.write('Person 1, Person 2, Cost: \n')

    G = collaborator_network.copy()
    distance_matrix = calc_distance_matrix(G)
    cost_orig = cost_function(distance_matrix, knowledge_matrix, alpha,
                              allow_neighbours)
    idx = cost_orig.index

    for i in range(n_rounds):
        couples_idx = optimization_func(cost_orig)
        total_round_cost = _calc_pairing_cost(cost_orig, couples_idx)
        print 'Round %d: %.9g' % (i, total_round_cost)
        couples = [(idx[c[0]], idx[c[1]]) for c in couples_idx]

        for c1, c2 in couples_idx:
            pair_cost = cost_orig.values[c1, c2]
            if output_fh:
                output_fh.write('%s , %s, %.4f\n' % (idx[c1], idx[c2],
                                                     pair_cost))
            cost_orig.values[c1, c2] = 100  # Don't use inf
            cost_orig.values[c2, c1] = 100  # Don't use inf
            # No longer a legal pairing
            all_iter_couples.append((idx[c1], idx[c2]))
            if update_network:
                G.add_edges_from(couples)
                distance_matrix = calc_distance_matrix(G)
                cost_orig = cost_function(distance_matrix, knowledge_matrix,
                                          alpha, allow_neighbours)
                # We have to re-calculate the costs now.
        output_fh.write('\n\n')
    return all_iter_couples


def calc_distance_matrix(G, max_distance=None):
    """Returns a matrix containing the shortest distance
    between all nodes in a network

    Parameters
    ----------
    G : graph
       A NetworkX graph

    max_distance : float or None, optional (default='None')
       The maximum possible distance value in the network.
       If None, max_distance is the longest shortest path between
       two nodes of the network (the graph eccentricity)

    Returns
    -------
    dist_matrix : NumPy array
      An NxN numpy array.

    Notes
    -----
    Along the diagonal, the values are all 0.
    Unconnected nodes have a distance of max_distance to other nodes.
    """

    # Network (collaborator) Distance
    dist_matrix = nx.all_pairs_shortest_path_length(G)
    dist_matrix = DataFrame(dist_matrix, index=G.nodes(), columns=G.nodes())
    if max_distance is None:
        max_distance = float(dist_matrix.max().max())
    dist_matrix = dist_matrix.fillna(max_distance)
    # The unconnected ones are infinitely far from the rest
    diag_idx = np.diag_indices(len(dist_matrix), ndim=2)
    dist_matrix.values[diag_idx] = 0
    return dist_matrix


def cost_function(distance_matrix, knowledge_matrix, alpha=0.95,
                  allow_neighbours=False):
    """Returns a matrix containing the cost all possible pairings.

    Parameters
    ----------
    distance_matrix : NumPy array like
       An NxN NumPy array containing the shortest path distance
       between all pairs

    knowledge_matrix : NumPy array like
       An NxN NumPy array containing a knowledge measure between
       all pairs that we wish to minimize

    alpha : float, optional (default=0.95)
       A scalar value that governs the trade off between minimizing
       the knowledge_matrix and the inverse of the distance_matrix

    Returns
    -------
    cost : NumPy array
      An NxN numpy array containing the cost of each possible pairings

    Notes
    -----
    Before combining the different matrix, they are standardized
    to zero mean and unit variance.
    """

    inv_distance = 1.0 / (distance_matrix)
    inv_distance = _special_scaler(inv_distance, allow_neighbours)
    knowledge_matrix = _special_scaler(knowledge_matrix)

    cost = ((1-alpha)*inv_distance)
    cost.add(alpha*(knowledge_matrix), fill_value=1)

    if not allow_neighbours:
        cost[distance_matrix == 1] = 100
        # Using np.inf causes issues with minimization.
    return cost


def make_pairs_with_nx(collaborator_network, knowledge_matrix,
                       output_fh=None, alpha=0.95, allow_neighbours=False,
                       n_rounds=5, update_network=False):
    """Calculates the optimal pairs for a given knowledge_matrix and
    collaborator network.  The cost function which is minimized is:

    .. math::
     cost = \frac{alpha}{S} + (1-alpha)*(K),

    where `S` is the matrix of shortest paths between all nodes in the
    collaborator network and `K` is the knowledge similarity.

    Parameters
    ----------
    collaborator_network : graph
       A networkx Graph

    knowledge_matrix : NumPy array like
       An NxN NumPy array containing a knowledge measure between
       all pairs that we wish to minimize

    output_fh : filehandle, optional (default=sys.stdout)
       A file-handle like object where to write out the pairs.
       If you want to suppress output, just open os.devnull
       and write to there.

    alpha : float, optional (default=0.95)
       A scalar value that governs the trade off between minimizing
       the knowledge_matrix and the inverse of the distance_matrix

    allow_neighbours : bool, optional (default=False)
        Whether or not pairs between people who are neighbours on the
        collaborator network are permitted.  If set to False, a very
        large penalty is applied to those pairs.  For very highly connected
        networks, you will still get pairs between neighbours where it is
        impossible to avoid them.

    n_rounds : int, optional (default=5)
        The number of rounds for which we calculate the pairs.

    update_network : bool, optional (default=False)
        Whether or not to update the network at each round.  If true,
        we update the collaborator network after each round, treating
        each interaction between pairs as a new edge.  Warning, setting
        this to True and allow_neighbours to False is not advised for highly
        connected networks.

    Returns
    -------
    pairs : list
      A list of n lists (where n is the number of rounds) which contains all
      the pairs for a given round in a tuple.

    Notes
    -----
    Before combining the different matrix, they are standardized
    to zero mean and unit variance.
    """

    def nx_optimizer(cost_orig, **kwargs):
        cost = np.rint(-cost_orig*1e9).astype('int64')
        edges = [(u, v, cost.values[u, v]) for u in
                 range(cost.shape[0]) for v in range(u)]
        UG = nx.Graph().to_undirected()
        UG.add_weighted_edges_from(edges)
        couples_dict = nx.max_weight_matching(UG, maxcardinality=True)
        return [(k, couples_dict[k]) for k in
                couples_dict if k < couples_dict[k]]
    return _make_pairs_common(nx_optimizer, collaborator_network,
                              knowledge_matrix, output_fh, alpha,
                              allow_neighbours, n_rounds, update_network)
