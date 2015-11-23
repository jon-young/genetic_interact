#!/usr/bin/env python

"""
Visualize interactions between clusters as networks

Created: 19 November 2015
"""

import itertools
import matplotlib.pyplot as plt
import networkx as nx


def btw_complex_net(pairs, clust2genes, gene2idx, adjMat):
    """Clusters are protein complexes
    INPUT: 1.) pairs <- [(complex 1 name, complex 2 name)]"""
    fig = plt.figure()
    
    # assemble network edges
    for p in pairs:
        cmplx1genes = clust2genes[p[0]]
        cmplx2genes = clust2genes[p[1]]

        # get interaction edges
        edges = list()
        for genePair in itertools.product(cmplx1genes, cmplx2genes):
            i = gene2idx[genePair[0]]
            j = gene2idx[genePair[1]]
            if adjMat[i,j] == 1:
                edges.append((genePair[0], genePair[1]))

        # draw network
        plt.clf()
        G_gi = nx.Graph()
        G_c1 = nx.Graph()
        G_c2 = nx.Graph()

        for e in edges:
            G_gi.add_edge(e[0], e[1])
        for gene in cmplx1genes:
            G_c1.add_node(gene)
        for gene in cmplx2genes:
            G_c2.add_node(gene)

        pos1 = nx.circular_layout(G_c1)
        pos2 = nx.circular_layout(G_c2)
        pos2.update((k, v+2) for k,v in pos2.items())
        
        posGI = dict()
        for gene in G_gi.nodes():
            if gene in pos1:
                posGI[gene] = pos1[gene]
            else:
                posGI[gene] = pos2[gene]

        nx.draw_networkx_nodes(G_c1, pos=pos1, alpha=0.25)
        nx.draw_networkx_labels(G_c1, pos=pos1)
        nx.draw_networkx_nodes(G_c2, pos=pos2, alpha=0.25)
        nx.draw_networkx_labels(G_c2, pos=pos2)
        nx.draw_networkx_edges(G_gi, pos=posGI)

        plt.title(p[0] + ' and ' + p[1])
        plt.axis('off')
        plt.show()
        
        plt.waitforbuttonpress()

