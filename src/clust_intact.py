#!/usr/bin/env python

"""
COMMAND LINE ARGS: "genetic interaction type" "organism (i.e. human, yeast)"

For a given species and genetic interaction type, compute statistics about 
the significance of interactions between a pair of gene clusters. A cluster 
could be genes tightly connected in a functional gene network. 

Created: 2 October 2015

HARD-CODED PARAMETERS:
    1.) organism common names and filenames
    2.) functional network directory
    3.) BIOGRID directory
    4.) columns in BIOGRID file to read
    5.) AUC upper and lower limits for predictability
    6.) # of cluster pairs for network visualization
    7.) write to JSON
"""

import bisect
import collections
import csv
import itertools
import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pyuserfcn
import random
import scipy.special
import scipy.stats as stats
import sys
from networkx.readwrite import json_graph
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from sklearn import metrics


def process_func_net(netwkFile):
    """Get nodes and edge weights of functional net and store in dictionary
    RETURNS <dict> {(gene ID1, gene ID2): LLS}"""
    node2edgewt = dict()
    netwkDir = os.path.join('..', '..', 'DataDownload', 'FunctionalNet', '')
    if not os.path.exists(netwkDir + netwkFile):
        netwkDir = os.path.join('..', 'data', '')
    scoreCol = 2
    for line in open(netwkDir + netwkFile):
        tokens = line.split('\t')
        node2edgewt[(tokens[0], tokens[1])] = float(tokens[scoreCol])
    return node2edgewt


def assign_gene_indices(node2edgewt):
    """Assign indices to genes to facilitate adjacency matrix creation
    ARGUMENTS: <dict> {(gene node 1, gene node 2): LLS}
    RETURNS <dict> {gene: matrix index}"""
    allGenes = set()
    for vx in node2edgewt.keys():
        allGenes.update(vx)
    geneList = sorted(allGenes)
    gene2idx = {gene: i for i, gene in enumerate(geneList)}
    return gene2idx


def build_netwk_adj_matrix(node2edgewt, gene2idx):
    """ARGUMENTS:
        1.) <dict> {(gene node 1, gene node 2): LLS}
        2.) <dict> {gene: matrix index}
    RETURNS <ndarray> adjacency matrix"""
    adjMat = np.zeros((len(gene2idx), len(gene2idx)))
    for vx in node2edgewt.keys():
        adjMat[gene2idx[vx[0]], gene2idx[vx[1]]] = node2edgewt[vx]
        adjMat[gene2idx[vx[1]], gene2idx[vx[0]]] = node2edgewt[vx]
    return adjMat


def read_biogrid(experimentSys, filename):
    """Read BIOGRID genetic interactions and return dictionary converting each 
    interactor ID to its genetic interaction partners
    NOTE: the correct column numbers need to be specified beforehand: 1 & 2 
    for Entrez, 5 & 6 for systematic name, 7 & 8 for official symbol
    ARGUMENTS: 
        1.) <string> type of genetic interaction
        2.) <string> BIOGRID file name
    RETURNS <dict> {seed Entrez: {interactors' Entrez}}"""
    seedSets = collections.defaultdict(set)
    allGenes = set()
    fileDir = os.path.join('..', 'data', '')
    experimentalSysColNum = 11
    for line in open(fileDir + filename):
        tokens = line.split('\t')
        if tokens[experimentalSysColNum] == experimentSys:
            seedSets[tokens[7]].add(tokens[8])
            seedSets[tokens[8]].add(tokens[7])
            allGenes.update([tokens[7], tokens[8]])
    return seedSets


def seed_set_predictability(gene2idx, adjMat, seedSets):
    """For each seed gene, measure its predictability for genetic interaction by 
    AUC. Also return a dictionary converting the seed gene (provided it is in 
    the network) to the set of its genetic interaction partners that are also 
    in the network.
    ARGUMENTS:
        1.) <dict> {gene: adjacency matrix index}
        2.) <ndarray> adjacency matrix
        3.) <dict> {seed gene: {genetic interactors}}
    RETURNS:
        1.) <list> [(seed AUC, seed gene)] ascend sort
        2.) <dict> {seed gene in func net: [interactors in func net]}
    """
    seedAUC = list()
    seed2interactors = dict()
    idx2gene = pyuserfcn.invert_dict(gene2idx)
    numCols = adjMat.shape[1]
    for seedGene in seedSets.keys():
        if seedGene in gene2idx:
            interactors = seedSets[seedGene]
            seedIdx = [gene2idx[i] for i in interactors if i in gene2idx]
            if len(seedIdx) > 0:
                llsSum = np.sum(adjMat[seedIdx,:], axis=0)
                trueLabels = np.zeros(numCols, dtype=np.int)
                trueLabels[seedIdx] = 1
                auc = metrics.roc_auc_score(trueLabels, llsSum)
                seedAUC.append((auc, seedGene))
                seed2interactors[seedGene] = [idx2gene[x] for x in seedIdx]
    seedAUC.sort()
    return seedAUC, seed2interactors


def get_predictive_seeds(seedAUC, lowerLim, upperLim):
    """Create a list of predictive seed genes
    INPUT:
        1.) <list> [(seed AUC, seed gene)] ascend sort
        2.) <float> minimum AUC of seed set
        3.) <float> maximum AUC of seed set
    RETURNS <list> [genes] ascend sort by AUC"""
    predictiveSeeds = list()
    for p in seedAUC:  # p=(AUC, gene)
        if p[0] >= lowerLim and p[0] < upperLim:
            predictiveSeeds.append(p[1])
    print('Number of predictive seed genes:', len(predictiveSeeds), '\n')
    return predictiveSeeds


def jaccard(s1, s2):
    """Calculate Jaccard index between 2 sets
    INPUT: 1.) & 2.) <set>
    RETURN <float>"""
    if isinstance(s1, set) and isinstance(s2, set):
        return len(s1 & s2)/len(s1 | s2)
    elif isinstance(s1, frozenset) and isinstance(s2, frozenset):
        return len(s1 & s2)/len(s1 | s2)
    else:
        print('Both arguments to Jaccard must be sets. Exiting...\n')
        sys.exit()


def combine_sets(setOfSets):
    """Combine various sets if they have enough in common measured by Jaccard
    INPUT & OUTPUT: <set> elements of set are frozensets"""
    minJaccard = 0.5
    print('The minimum Jaccard index is', minJaccard, '\n')
    # the following 17 lines are the "do" part of a "do-while" loop
    newSetOfSets = set()  # DO START
    deleteLater = set()
    for pair in itertools.combinations(setOfSets, 2):
        if jaccard(pair[0], pair[1]) >= minJaccard:
            pairUnion = pair[0] | pair[1]
            if pairUnion == pair[0]:
                newSetOfSets.add(pairUnion)
                deleteLater.add(pair[1])
            elif pairUnion == pair[1]:
                newSetOfSets.add(pairUnion)
                deleteLater.add(pair[0])
            else:
                newSetOfSets.add(pairUnion)
                deleteLater.update((pair[0], pair[1]))
        else:
            newSetOfSets.update((pair[0], pair[1]))
    newSetOfSets = {x for x in newSetOfSets if x not in deleteLater}  # DO END
    while len(newSetOfSets) < len(setOfSets):
        setOfSets = newSetOfSets
        newSetOfSets = set()
        deleteLater = set()
        for pair in itertools.combinations(setOfSets, 2):
            if jaccard(pair[0], pair[1]) >= minJaccard:
                pairUnion = pair[0] | pair[1]
                if pairUnion == pair[0]:
                    newSetOfSets.add(pairUnion)
                    deleteLater.add(pair[1])
                elif pairUnion == pair[1]:
                    newSetOfSets.add(pairUnion)
                    deleteLater.add(pair[0])
                else:
                    newSetOfSets.add(pairUnion)
                    deleteLater.update((pair[0], pair[1]))
            else:
                newSetOfSets.update((pair[0], pair[1]))
        newSetOfSets = {x for x in newSetOfSets if x not in deleteLater}
    # remove sets that are subsets of other sets
    deleteLater = set()
    for pair in itertools.combinations(setOfSets, 2):
        if pair[0].issubset(pair[1]):
            deleteLater.add(pair[0])
        elif pair[1].issubset(pair[0]):
            deleteLater.add(pair[1])
        else:
            pass
    return {s for s in setOfSets if s not in deleteLater}


def get_interacting_pairs(seeds, seed2interactors):
    """Assemble all the genetic interactions for genes in the network into a 
    single set.
    INPUT:
        1.) <iterable> seed genes
        2.) <dict> {seed gene: [interactor genes]}
    RETURNS <set> {(seed, interactor), (interactor, seed)}"""
    interactPairs = set()
    for seed in seeds:
        for interactor in seed2interactors[seed]:
            interactPairs.update([(seed, interactor), (interactor, seed)])
    return interactPairs


def interaction_stats(interactPairs, id2set, gene2idx, adjMat):
    """Get statistics of interactions between predictive seed sets
    INPUT:
        1.) <set> {(seed, interactor), (interactor, seed)}
        2.) <dict> {int: frozensets}
        3.) <dict> {gene: matrix index}
        4.) <ndarray> adjacency matrix
    RETURN <dict> {(int,int): p-val}"""
    numGenes = len(set(itertools.chain.from_iterable(interactPairs)))
    p = (len(interactPairs)/2)/scipy.special.binom(numGenes, 2)

    print('The background probability is', p, '\n')  # CHECKING

    pvals = dict()
    for idPair in itertools.combinations(id2set.keys(), 2):
        set1st = id2set[idPair[0]]
        set2nd = id2set[idPair[1]]
        
        # remove intersection & genes w/o functional partners
        set1 = set1st - set2nd
        set2 = set2nd - set1st
        deleteSet = set()
        idx1 = [gene2idx[g] for g in set1]
        for g in set1:
            if np.sum(adjMat[gene2idx[g], idx1], axis=0) == 0:
                deleteSet.add(g)
        set1st = {g for g in set1 if g not in deleteSet}
        deleteSet = set()
        idx2 = [gene2idx[g] for g in set2]
        for g in set2:
            if np.sum(adjMat[gene2idx[g], idx2], axis=0) == 0:
                deleteSet.add(g)
        set2nd = {g for g in set2 if g not in deleteSet}
        
        count = 0
        num1stSet, num2ndSet = len(set1st), len(set2nd)
        for genePair in itertools.product(set1st, set2nd):
            if genePair in interactPairs:
                count += 1
        n = num1stSet * num2ndSet
        pvals[idPair] = stats.binom.pmf(count,n,p) + stats.binom.sf(count,n,p)
    return pvals


def interaction_stats_2(seeds, seed2intacts, id2set, gene2idx, adjMat):
    """Compute significance of cluster interactions by comparing with random 
    draws from the interaction network
    INPUT:
        1.) <list> [predictive seed genes]
        2.) <dict> {seed: [interactors]}
        3.) <dict> {int: {genes}}
        4.) <dict> {gene: index for functional net adjacency matrix}
        5.) <ndarray> functional net adjacency matrix
    RETURNS <dict> {(int, int): score}"""
    # create interaction network
    allGenes = set(seeds)
    for s in seeds:
        allGenes.update(seed2intacts[s])
    numGenes = len(allGenes)
    labels = sorted(allGenes)
    intactDf = pd.DataFrame(np.zeros((numGenes, numGenes), dtype=np.int), 
                            index=labels, columns=labels)
    for s in seeds:
        for i in seed2intacts[s]:
            intactDf.loc[s,i] = 1
            intactDf.loc[i,s] = 1

    numRand = 100
    scores = dict()
    for idPair in itertools.combinations(id2set.keys(), 2):
        set1st = id2set[idPair[0]]
        set2nd = id2set[idPair[1]]
        
        # remove intersection and genes w/o functional partners
        set1 = set1st - set2nd
        set2 = set2nd - set1st
        deleteSet = set()
        idx1 = [gene2idx[g] for g in set1]
        for g in set1:
            if np.sum(adjMat[gene2idx[g], idx1], axis=0) == 0:
                deleteSet.add(g)
        list1st = [g for g in set1 if g not in deleteSet]
        deleteSet = set()
        idx2 = [gene2idx[g] for g in set2]
        for g in set2:
            if np.sum(adjMat[gene2idx[g], idx2], axis=0) == 0:
                deleteSet.add(g)
        list2nd = [g for g in set2 if g not in deleteSet]

        # randomization
        num1stSet = len(list1st)
        num2ndSet = len(list2nd)
        randIntactCounts = list()
        for i in range(numRand):
            randset1 = random.sample(allGenes, num1stSet)
            randset2 = random.sample(allGenes, num2ndSet)
            numIntacts = intactDf.loc[randset1, randset2].sum().sum()
            randIntactCounts.append(numIntacts)
        randIntactCounts.sort()

        # calculate actual counts
        actualCount = intactDf.loc[list1st, list2nd].sum().sum()
        rank = bisect.bisect_left(randIntactCounts, actualCount)
        scores[idPair] = (numRand - rank + 1)/(numRand + 1)

    return scores


def net_edges(node2edgewt, pair2pval, id2set, intactPairs, gene2idx, adjMat):
    """Get network edges
    INPUTS:
        1.) <dict> {(gene ID1, gene ID2): LLS}
        2.) <dict> {(int,int): p-val}
        3.) <dict> {int: frozensets}
        4.) <set> {(seed, interactor), (interactor, seed)}
        5.) <dict> {gene: matrix index}
        6.) <ndarray> adjacency matrix
    RETURN <list> [(node, node, edge weight)]"""
    # multiple hypothesis correction
    pvals = np.array(list(pair2pval.values()))
    rejected, pvalsCor = fdrcorrection0(pvals, alpha=0.10)
    numSig = np.sum(rejected)
    print('Number of significant interacting pairs (10% FDR):', numSig, '\n')
    ##pvalsSig = np.sort(pvals[rejected])
    ##numExam = 1
    ##if numExam == 'all':
    ##    pvalCutoff = pvalsSig[pvalsSig.size - 1]
    ##else:
    ##    pvalCutoff = pvalsSig[numExam - 1]
    minpval = np.sort(pvals)[0]  # NEW
    print('Minimum p-value =', minpval, '\n')  # NEW
    
    edges = list()
    for idPair in pair2pval.keys():
        if pair2pval[idPair] <= minpval:  # CHANGED
            set1st = id2set[idPair[0]]
            set2nd = id2set[idPair[1]]
            
            # remove intersection & genes w/o functional partners
            set1 = set1st - set2nd
            set2 = set2nd - set1st
            deleteSet = set()
            idx1 = [gene2idx[g] for g in set1]
            for g in set1:
                if np.sum(adjMat[gene2idx[g], idx1], axis=0) == 0:
                    deleteSet.add(g)
            set1st = {g for g in set1 if g not in deleteSet}
            deleteSet = set()
            idx2 = [gene2idx[g] for g in set2]
            for g in set2:
                if np.sum(adjMat[gene2idx[g], idx2], axis=0) == 0:
                    deleteSet.add(g)
            set2nd = {g for g in set2 if g not in deleteSet}
            
            # get interaction edges
            for genePair in itertools.product(set1st, set2nd):
                if genePair in intactPairs:
                    edges.append((genePair[0], genePair[1], -1))
            
            # get functional edges
            for genePair in itertools.combinations(set1st, 2):
                if genePair in node2edgewt:
                    edges.append((genePair[0], genePair[1], node2edgewt[genePair]))
                elif (genePair[1], genePair[0]) in node2edgewt:
                    revPair = (genePair[1], genePair[0])
                    edges.append((genePair[1], genePair[0], node2edgewt[revPair]))
                else:
                    pass
            for genePair in itertools.combinations(set2nd, 2):
                if genePair in node2edgewt:
                    edges.append((genePair[0], genePair[1], node2edgewt[genePair]))
                elif (genePair[1], genePair[0]) in node2edgewt:
                    revPair = (genePair[1], genePair[0])
                    edges.append((genePair[1], genePair[0], node2edgewt[revPair]))
                else:
                    pass
    return edges


def plot_network(edges, experimentSys, organism, saveFlag=0):
    """Use NetworkX to save a visualization of the gene network
    INPUT: 
        1.) <list> [(node, node, edge weight)]
        2.) <string> genetic interaction type
        3.) <string> organism"""
    G_gi = nx.Graph()
    G_f = nx.Graph()
    for e in edges:
        if e[2] < 0:  # genetic interaction
            G_gi.add_edge(e[0], e[1])
        else:
            G_f.add_edge(e[0], e[1], weight=e[2], style='solid')
    weights = [G_f[u][v]['weight'] for u,v in G_f.edges()]
    style_f = [G_f[u][v]['style'] for u,v in G_f.edges()]
    pos_f = nx.spring_layout(G_f)
    nx.draw_networkx(G_f, pos=pos_f, node_size=0, alpha=0.5, width=weights, 
                     style=style_f, font_size=10)
    nx.draw_networkx_edges(G_gi, pos=pos_f, width=0.5, style='dashed')

    if saveFlag:
        writeDir = os.path.join('..', 'results', organism+'IntactClust', '')
        writeFilename = organism + ''.join(experimentSys.split(' ')) + 'Net.svg'
        plt.savefig(writeDir+writeFilename)
    
    plt.axis('off')
    plt.show()


def plot_1_network(pair2pval, pvalCutoff, id2set, gene2idx, adjMat, intactPairs, node2edgewt, experimentSys, organism):
    count = 1
    for idPair in pair2pval.keys():
        if pair2pval[idPair] <= pvalCutoff:
            edges = list()
            set1st = id2set[idPair[0]]
            set2nd = id2set[idPair[1]]
            
            # remove intersection and genes w/o functional partners
            set1 = set1st - set2nd
            set2 = set2nd - set1st
            deleteSet = set()
            idx1 = [gene2idx[g] for g in set1]
            for g in set1:
                if np.sum(adjMat[gene2idx[g], idx1], axis=0) == 0:
                    deleteSet.add(g)
            set1st = {g for g in set1 if g not in deleteSet}
            deleteSet = set()
            idx2 = [gene2idx[g] for g in set2]
            for g in set2:
                if np.sum(adjMat[gene2idx[g], idx2], axis=0) == 0:
                    deleteSet.add(g)
            set2nd = {g for g in set2 if g not in deleteSet}
            
            # get interaction edges
            for genePair in itertools.product(set1st, set2nd):
                if genePair in intactPairs:
                    edges.append((genePair[0], genePair[1], -1))
            
            # get functional edges
            for genePair in itertools.combinations(set1st, 2):
                if genePair in node2edgewt:
                    edges.append((genePair[0], genePair[1], node2edgewt[genePair]))
                elif (genePair[1], genePair[0]) in node2edgewt:
                    revPair = (genePair[1], genePair[0])
                    edges.append((genePair[1], genePair[0], node2edgewt[revPair]))
                else:
                    pass
            for genePair in itertools.combinations(set2nd, 2):
                if genePair in node2edgewt:
                    edges.append((genePair[0], genePair[1], node2edgewt[genePair]))
                elif (genePair[1], genePair[0]) in node2edgewt:
                    revPair = (genePair[1], genePair[0])
                    edges.append((genePair[1], genePair[0], node2edgewt[revPair]))
                else:
                    pass
            
            # plot network of a cluster pair
            plt.figure()
            G_gi = nx.Graph()
            G_f = nx.Graph()
            for e in edges:
                if e[2] < 0:  # genetic interaction
                    G_gi.add_edge(e[0], e[1])
                else:
                    G_f.add_edge(e[0], e[1], weight=e[2], style='solid')
            weights = [G_f[u][v]['weight'] for u,v in G_f.edges()]
            style_f = [G_f[u][v]['style'] for u,v in G_f.edges()]
            pos_f = nx.spring_layout(G_f)
            nx.draw_networkx(G_f, pos=pos_f, node_size=0, alpha=0.5, width=weights, 
                             style=style_f, font_size=10, font_color='b')
            nx.draw_networkx_edges(G_gi, pos=pos_f, width=0.5, style='dashed')
        
            writeDir = os.path.join('..', 'results', organism+'IntactClust', '')
            writeFilename = organism + ''.join(experimentSys.split(' ')) + 'Net-' + str(count) + '.png'
            count += 1
            
            plt.axis('off')
            plt.savefig(writeDir+writeFilename)
            plt.close()


def write_json(edges, experimentSys, organism):
    """Write network to JSON format for visualization with D3.js
    INPUT: 
        1.) <list> [(node, node, edge weight)]
        2.) <string> genetic interaction type
        3.) <string> organism"""
    G = nx.Graph()
    for e in edges:
        if e[2] < 0:
            G.add_edge(e[0], e[1], weight=0.5, style='dashed')
        else:
            G.add_edge(e[0], e[1], weight=e[2], style='solid')
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    styles = [G[u][v]['style'] for u,v in G.edges()]
    for n in G:
        G.node[n]['name'] = n
    d = json_graph.node_link_data(G)  # node-link format to serialize
    writeFile = organism + ''.join(experimentSys.split(' ')) + '.json'
    json.dump(d, open(os.path.join('..', 'd3', writeFile), 'w'))


def main():
    experimentSys = sys.argv[1]
    organism = sys.argv[2].strip().lower()
    org2files = {'human': ('H6Net_CC.net', 
                           'HumanNet2_adj_matrix.npy', 
                           'BIOGRID-3.4.127-human.txt'),
                 'yeast': ('yeastnet2.gene.txt', 
                           'YeastNet2_adj_matrix.npy', 
                           'BIOGRID-3.4.127-for-yeastnetv2.txt'),
                 'fly': ('FlyNetEntrez-noNull.txt',
                         'FlyNet_adj_matrix.npy',
                         'BIOGRID-3.4.127-fly.txt')}
    if organism not in org2files:
        print('Organism not found.')
        print('Available organisms: human, yeast')
        sys.exit()
    netwkFile = org2files[organism][0]
    matrixFile = org2files[organism][1]
    biogridFile = org2files[organism][2]
    node2edgewt = process_func_net(netwkFile)
    gene2idx = assign_gene_indices(node2edgewt)
    matrixPath = os.path.join('..', 'data', matrixFile)
    try:
        adjMat = np.load(matrixPath)
    except:
        print('Network file not found. Creating network matrix...\n')
        adjMat = build_netwk_adj_matrix(node2edgewt, gene2idx)
        np.save(matrixPath, adjMat)
    print('Number of genes in functional network:', len(gene2idx.keys()), '\n')
    seedSets = read_biogrid(experimentSys, biogridFile)
    seedAUC, seed2intacts = seed_set_predictability(gene2idx, adjMat, seedSets)
    seedGenes = [x[1] for x in seedAUC]
    intactPairs = get_interacting_pairs(seedGenes, seed2intacts)
    
    lowerAUC = 0.9
    upperAUC = 1.0
    predictiveSeeds = get_predictive_seeds(seedAUC, lowerAUC, upperAUC)
    
    setOfSets = set()
    for seed in predictiveSeeds:
        setOfSets.add( frozenset(set(seed2intacts[seed])) )
    combinedSets = combine_sets(setOfSets)
    print('Number of combined sets using Jaccard:', len(combinedSets), '\n')

    id2set = {i: fs for i,fs in enumerate(combinedSets)}

    ##pvals = interaction_stats(intactPairs, id2set, gene2idx, adjMat)
    pvals = interaction_stats_2(predictiveSeeds, seed2intacts, id2set, 
                                gene2idx, adjMat)
    print('Number of set pairs:', len(pvals), '\n')

    # multiple hypothesis correction
    p_values = np.array(list(pvals.values()))
    rejected, pvalsCor = fdrcorrection0(p_values, alpha=0.10)
    numSig = np.sum(rejected)
    print('Number of significant interacting pairs (10% FDR):', numSig, '\n')
    minpval = np.sort(p_values)[0]
    print('Minimum p-value =', minpval, '\n')
    plt.hist(p_values, bins=math.sqrt(p_values.size), log=True)
    plt.xlabel('p-value')
    plt.ylabel('Count')
    plt.title(experimentSys)
    plt.show()
    
    ##edges = net_edges(node2edgewt, pvals, id2set, intactPairs, gene2idx, adjMat)
    ##plot_network(edges, experimentSys, organism)
    plot_1_network(pvals, minpval, id2set, gene2idx, adjMat, intactPairs, node2edgewt, experimentSys, organism)

    writeToJSON = 0
    if writeToJSON:
        write_json(edges, experimentSys, organism)


if __name__=="__main__":
    main()

