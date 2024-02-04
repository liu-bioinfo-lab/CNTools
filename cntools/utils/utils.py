import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def cns_info(ds, n_cns, cns):
    data = ds.data
    info_ent, info_pat, info_size = {i: [] for i in range(n_cns)}, np.zeros(n_cns), np.zeros(n_cns)
    for sample in data:
        for image in data[sample]:
            graph = nx.Graph(data[sample][image].edge_indices)
            graph.add_nodes_from([i for i in range(data[sample][image].n_cells)])
            for edge in graph.edges:
                if cns[sample][image][edge[0]] != cns[sample][image][edge[1]]:
                    graph.remove_edge(edge[0], edge[1])
            for i in nx.connected_components(graph):
                info_ent[cns[sample][image][list(i)[0]]] += data[sample][image].cts_oh[list(i)].tolist()
                info_pat[cns[sample][image][list(i)[0]]] += 1
                info_size[cns[sample][image][list(i)[0]]] += len(i)
    for i in info_ent:
        if not info_ent[i]:
            info_ent[i] = 0
        else:
            p = np.array(info_ent[i]).mean(axis=0)
            p = p[p.nonzero()]
            info_ent[i] = -(p * np.log2(p)).sum()
    info_ent = np.array(list(info_ent.values()))
    print(f'Entropy: {(info_ent * info_size / info_size.sum()).sum():.3f}, Size: {sum(info_size) / sum(info_pat):.2f}')


def delauney_edges(ds):
    data = ds.data
    edges = [[], []]
    i = 0
    for sample in data:
        for image in data[sample]:
            edges[0] += [edge[0] + i for edge in data[sample][image].edge_indices]
            edges[1] += [edge[1] + i for edge in data[sample][image].edge_indices]
            i += data[sample][image].n_cells
    edges = csr_matrix((np.ones(len(edges[0]), dtype=int), (edges[0], edges[1])), shape=(i, i))
    return edges
