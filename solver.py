import networkx as nx
import parse
import utils
import random
import os
from help_horizon import lock

def solve(G, T=None, cost=float('inf')):
    """
    Args:
        G: networkx.Graph
        T: the best known dominating-set tree. None if unknown
        cost: the cost of the best known T

    Returns:
        T: networkx.Graph
    """
    tries, max_tries = 0, 4*len(list(G.nodes))
    while tries < max_tries:
        nodes = list(G.nodes)
        random.shuffle(nodes)
        i = 0
        while not nx.is_dominating_set(G, nodes[:i]) or not nx.is_connected(G.subgraph(nodes[:i])):
            i += 1
        new_T, new_cost = min(((t, utils.average_pairwise_distance_fast(t)) for t in gen_candidates(G, nodes[:i])), key=lambda p: p[1])
        if new_cost < cost:
            T, cost = new_T, new_cost
            tries = 0
        else:
            tries += 1
    return T, cost


def gen_candidates(G, nodes, length=10):
    '''Return an iterable over a set of candidate minimum routing cost
       trees in G given a set of vertices NODES.

       Currently gives shortest-path trees from every vertex.
    '''
    subG = G.subgraph(nodes)
    for node in nodes:
        _, paths = nx.algorithms.shortest_paths.weighted.single_source_dijkstra(subG, node)
        T = nx.Graph()
        T.add_nodes_from(nodes)
        for p in paths.values():
            for i in range(len(p) - 1):
                T.add_edge(p[i], p[i + 1], weight=G[p[i]][p[i + 1]]['weight'])
        yield T

def solve_file(files):
    infile, outfile = files
    G, T, cost = parse.read_input_file(infile), None, float('inf')
    if os.path.exists(outfile):
        T = parse.read_output_file(outfile, G)
        cost = utils.average_pairwise_distance_fast(T)
    new_T, new_cost = solve(G, T, cost)
    if new_cost < cost:
        if lock is not None:
            lock.acquire()
        parse.write_output_file(new_T, outfile)
        print(f"New minimum found for {infile}, with cost {new_cost}.")
        if lock is not None:
            lock.release()
