import networkx as nx
import json
from scipy.spatial.distance import rogerstanimoto
from pandas import DataFrame, read_csv
from itertools import permutations
from collections import defaultdict
from pairings import make_pairs_with_nx


class distance_calculator(object):
    """Abstract class to give a contract on how DistanceCalculator class"""
    def get_distance_matrix(self):
        raise NotImplementedError("Should have implemented this")


class mutual_compatibility_calculator(distance_calculator):
    def __init__(self, dist_fcn, known_methods_filename,
                 wanted_methods_filename, basedir=None):
        self.methods_i_know = read_csv(known_methods_filename,
                                       index_col=0)
        self.methods_i_want = read_csv(wanted_methods_filename,
                                       index_col=0)
        self.index = self.methods_i_want.columns
        self.dist_fcn = dist_fcn

    def get_distance_matrix(self):
        method_dissimilarity = defaultdict(dict)
        for you_know, i_want in permutations(self.index, 2):
            method_dissimilarity[i_want][you_know] = \
                self.dist_fcn(self.methods_i_want[i_want],
                              self.methods_i_know[you_know])

        method_dissimilarity = DataFrame(method_dissimilarity,
                                         index=self.index, columns=self.index)
        # you are as distant as possible from yourself
        method_dissimilarity = method_dissimilarity.fillna(0)
        return method_dissimilarity


class knowledge_similarity_calculator(distance_calculator):
    def __init__(self, dist_fcn, known_methods_filename, basedir=None):
        self.methods_i_know = read_csv(known_methods_filename,
                                       index_col=0)
        self.index = self.methods_i_know.columns
        self.dist_fcn = dist_fcn

    def get_distance_matrix(self):
        knowledge_similarity = defaultdict(dict)
        for you_know, i_know in permutations(self.index, 2):
            knowledge_similarity[i_know][you_know] = \
                1-self.dist_fcn(self.methods_i_know[i_know],
                                self.methods_i_know[you_know])
        knowledge_similarity = DataFrame(knowledge_similarity, index=self.index,
                                         columns=self.index)
        # Your knowledge similarity to yourself is 100%
        knowledge_similarity = knowledge_similarity.fillna(1)
        return knowledge_similarity


class speed_dating:
    def __init__(self, network_filename, distance_calculator,
                 output_fh=None):
        self.network = nx.read_edgelist(network_filename)
        self.distance_calculator = distance_calculator
        self.output_fh = output_fh

    def getPairs(self, alpha, n_rounds, allow_neighbours=False):
        distance_matrix = self.distance_calculator.get_distance_matrix()
        return make_pairs_with_nx(self.network,
                                  distance_matrix,
                                  output_fh=self.output_fh,
                                  alpha=alpha,
                                  allow_neighbours=False,
                                  n_rounds=n_rounds)

if __name__ == "__main__":
    settings_fh = open('Settings.json')
    settings = json.load(settings_fh)

    # Settings:
    dist_fcn = rogerstanimoto
    network_fn = settings['interactions']
    wanted_methods_fn = settings['wanted_methods']
    known_methods_fn = settings['known_methods']
    n_rounds = settings['n_rounds']
    alpha = settings['alpha']

    knowledge_similarity_output = settings['knowledge_similarity_output']
    if knowledge_similarity_output is not None:
        knowledge_similarity_fh = open(knowledge_similarity_output, 'w')

    mutual_compatibility_output = settings['mutual_compatibility_output']
    if knowledge_similarity_output is not None:
        mutual_compatibility_fh = open(knowledge_similarity_output, 'w')

    meth_diss_calc = mutual_compatibility_calculator(dist_fcn, known_methods_fn,
                                                     wanted_methods_fn)
    know_sim_calc = knowledge_similarity_calculator(dist_fcn, known_methods_fn)

    speed_dating_with_mutual_comp = speed_dating(network_fn, meth_diss_calc,
                                                 mutual_compatibility_output)
    pairings_for_meth_diss = speed_dating_with_mutual_comp.getPairs(alpha,
                                                                    n_rounds)
    speed_dating_with_know_sim = speed_dating(network_fn, know_sim_calc)
    pairings_for_know_sim = speed_dating_with_know_sim.getPairs(alpha,
                                                                n_rounds)
