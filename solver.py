import networkx as nx
import parse
import utils
import random
import os
from help_horizon import LOCK

def solve(G, T=None, cost=float('inf'), multiplier=0.5):
    """
    Args:
        G: networkx.Graph
        T: the best known dominating-set tree. None if unknown
        cost: the cost of the best known T

    Returns:
        T: networkx.Graph
    """
    tries, max_tries = 0, int(multiplier*len(list(G.nodes)))
    while tries < max_tries:
        if cost == 0:
            break
        nodes = list(G.nodes)
        random.shuffle(nodes)
        i = 0
        while not nx.is_dominating_set(G, nodes[:i]) or not nx.is_connected(G.subgraph(nodes[:i])):
            i += 1
        new_T, new_cost = min((pick_leaves(G, t, utils.average_pairwise_distance_fast(t)) for t in gen_candidates(G, nodes[:i])),
                key=lambda p: p[1])
        if new_cost < cost:
            T, cost = new_T, new_cost
            tries = 0
        else:
            tries += 1
    return T, cost


def gen_candidates(G, nodes):
    '''Return an iterable over a set of candidate minimum routing cost
       trees in G given a set of vertices NODES.

       Currently gives shortest-path trees from every vertex after campos(G).
    '''
    subG = G.subgraph(nodes)
    yield campos(subG)
    for node in nodes:
        _, paths = nx.algorithms.shortest_paths.weighted.single_source_dijkstra(subG, node)
        T = nx.Graph()
        T.add_nodes_from(nodes)
        for p in paths.values():
            for i in range(len(p) - 1):
                T.add_edge(p[i], p[i + 1], weight=G[p[i]][p[i + 1]]['weight'])
        yield T


def pick_leaves(G, T, cost):
    '''Given a tree T and a graph G, recursively picks off leaves to find the minimum possible
    routing cost subtree of T

    Returns the best T and its cost.
    '''
    def pick_leaves_helper(G, T, edges, i):
        if not utils.is_valid_network(G, T):
            return float("inf"), []
        leafset = set(leaf for leaf, neighbor in edges)
        new_leaf_edges = [(n, list(T[n])[0]) for n in T.nodes if n not in leafset and len(T[n]) == 1]
        edges = edges + new_leaf_edges
        cost, nodes_to_remove = utils.average_pairwise_distance_fast(T), []
        for k in range(i, len(edges)):
            leaf, neighbor = edges[k]
            w = T[leaf][neighbor]['weight']
            T.remove_node(leaf)
            new_cost, additional_nodes = pick_leaves_helper(G, T, edges, k + 1)
            if new_cost < cost:
                cost, nodes_to_remove = new_cost, [leaf] + additional_nodes
            T.add_edge(leaf, neighbor, weight=w)
        return cost, nodes_to_remove
    new_cost, nodes_to_remove = pick_leaves_helper(G, T, [], 0)
    if new_cost < cost:
        return T.subgraph(T.nodes - set(nodes_to_remove)), new_cost
    else:
        return T, cost


def mindex(L, key=lambda x: x):
    if not L:
        raise ValueError
    i, min_index, min_key = 1, 0, key(L[0])
    while i < len(L):
        new_key = key(L[i])
        if new_key < min_key:
            min_index, min_key = i, new_key
        i += 1
    return min_index

def campos(G):
    T = nx.Graph()
    total_weight, num_edges, s, m = 0, 0, {n: 0 for n in G.nodes}, {n: 0 for n in G.nodes}
    for u, v, d in G.edges(data=True):
        w = d['weight']
        total_weight += w
        s[u] += w
        s[v] += w
        m[u] = max(m[u], w)
        m[v] = max(m[v], w)
        num_edges += 1
    mean = total_weight / num_edges
    std_dev = (sum((d['weight'] - mean)**2 for u, v, d in G.edges(data=True))/(num_edges - 1))**0.5
    if std_dev/mean < 0.4 + 0.005*(len(G) - 10):
        C_4 = C_5 = 1
    else:
        C_4 = 0.9
        C_5 = 0.1
    w, cf, sp_max, f = dict(), dict(), 0, None
    for v in G.nodes:
        w[v] = cf[v] = float('inf')
        spv = 0.2*len(G[v]) + 0.6*(len(G[v])/s[v]) + 0.2*(1/m[v])
        if spv > sp_max:
            sp_max = spv
            f = v
    w[f] = cf[f] = 0
    L = [f]
    wd, jsp, p = {n: float('inf') for n in G.nodes}, {n: 0 for n in G.nodes}, {n: None for n in G.nodes}
    T.add_node(f)
    while L:
        u = L.pop(mindex(L, key=lambda v: (wd[v], -jsp[v])))
        for v in G[u]:
            if v not in T:
                wdt = C_4 * G[u][v]['weight'] + C_5 * (cf[u] + G[u][v]['weight'])
                jspt = (len(G[u]) + len(G[v])) + (len(G[v]) + len(G[u]))/(s[v] + s[u])
                if wdt < wd[v]:
                    wd[v], jsp[v] = wdt, jspt
                    p[v] = u
                elif wdt == wd[v] and jspt >= jsp[v]:
                    jsp[v] = jspt
                    p[v] = u
                if v not in L:
                    L.append(v)
        if u != f:
            T.add_edge(u, p[u], weight=G[u][p[u]]['weight'])
    return T


def solve_file(files):
    infile, outfile = files
    G, T, cost = parse.read_input_file(infile), None, float('inf')
    if os.path.exists(outfile):
        try:
            T = parse.read_output_file(outfile, G)
        except:
            print(f"{outfile} could not be read.")
            return None
        cost = utils.average_pairwise_distance_fast(T)
    new_T, new_cost = solve(G, T, cost)
    if new_cost < cost:
        parse.write_output_file(new_T, outfile)
        if LOCK is not None:
            LOCK.acquire()
        print(f"New minimum found for {infile}, with cost {new_cost}.")
        if LOCK is not None:
            LOCK.release()
    else:
        if LOCK is not None:
            LOCK.acquire()
        print(f"No new minimum found for {infile}.")
        if LOCK is not None:
            LOCK.release()
