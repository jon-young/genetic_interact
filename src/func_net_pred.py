#!/usr/bin/env python

"""
Predict genetic interactions using a functional gene network 
and known interactions

HARD-CODED PARAMETERS:
    1.) functional network file path
    2.) save or load adjacency matrix
    3.) BIOGRID file
    4.) columns in BIOGRID file to read
    5a.) how many seed sets to write out (AUC)
    5b.) filepath for Entrez-to-name conversion
    5c.) entrez2names[ ][0] vs entrez2names[ ]
"""

import bisect
import collections
import matplotlib.pyplot as plt
import numpy
import pickle
import pyuserfcn
import sys
from sklearn import metrics


def process_func_net():
    """
    Get nodes and edge weights of HumanNet and store in dictionary
    RETURNS:
        <dict> {(Entrez ID1, Entrez ID2): LLS}
    """
    node2edgewt = dict()
    netwk = '../data/FlyNetEntrez-noNull.txt'
    scoreCol = 2
    for line in open(netwk):
        tokens = line.split('\t')
        node2edgewt[(tokens[0], tokens[1])] = float(tokens[scoreCol])
    return node2edgewt


def get_entrez_indices(node2edgewt):
    """
    Assign indices to Entrez IDs to facilitate adjacency matrix creation
    ARGUMENTS:
        <dict> {(Entrez ID1, Entrez ID2): LLS}
    RETURNS:
        <dict> {Entrez ID: matrix index}
    """
    entrezSet = set()
    entrez2idx = dict()
    for vx in node2edgewt.keys():
        entrezSet.update(vx)
    entrezList = sorted([int(n) for n in entrezSet])
    for n,entrezID in enumerate(entrezList):
        entrez2idx[str(entrezID)] = n
    return entrez2idx


def build_netwk_adj_matrix(node2edgewt, entrez2idx):
    """
    ARGUMENTS:
        <dict> {(Entrez ID1, Entrez ID2): LLS}
        <dict> {Entrez ID: matrix index}
    RETURNS:
        <ndarray> adjacency matrix"""
    adjMat = numpy.zeros((len(entrez2idx),len(entrez2idx)))
    for vx in node2edgewt.keys():
        adjMat[entrez2idx[vx[0]], entrez2idx[vx[1]]] = node2edgewt[vx]
        adjMat[entrez2idx[vx[1]], entrez2idx[vx[0]]] = node2edgewt[vx]
    return adjMat


def read_known_interact():
    """
    For files containing only the interacting gene pairs on each line
    (each gene in a column)
    RETURNS:
        <dict> {seed Entrez: {interactors' Entrez}}
    """
    seedSets = collections.defaultdict(set)
    allGenes = set()
    filePath = os.path.join('..', 'data', 'user_gen', 
            'intact_gene_intact_human.entrez')
    for line in open(filePath):
        tokens = line.rstrip().split('\t')
        seedSets[tokens[0]].add(tokens[1])
        seedSets[tokens[1]].add(tokens[0])
        allGenes.update([tokens[0], tokens[1]])
    print('Number of genes in interactions:', len(allGenes))
    return seedSets


def read_biogrid(experimentSys):
    """
    Read BIOGRID genetic interactions and return dictionary converting each 
    interactor ID to its genetic interaction partners
    NOTE: the correct column numbers need to be specified beforehand
    ARGUMENTS:
        <string> type of genetic interaction
    RETURNS:
        <dict> {seed Entrez: {interactors' Entrez}}
    """
    seedSets = collections.defaultdict(set)
    allGenes = set()
    filepath = '../data/BIOGRID-3.4.127-for-yeastnetv2.txt'
    experimentalSysColNum = 11
    for line in open(filepath):
        tokens = line.split('\t')
        if tokens[experimentalSysColNum] == experimentSys:
            seedSets[tokens[5]].add(tokens[6])
            seedSets[tokens[6]].add(tokens[5])
            allGenes.update([tokens[5], tokens[6]])
    print('Number of genes in interactions:', len(allGenes))
    return seedSets


def seed_set_predictability(entrez2idx, adjMat, seedSets):
    """
    For each seed gene, measure its predictability for genetic interaction by 
    AUC. Also return a dictionary converting the seed gene (provided it is in 
    the network) to the set of its genetic interaction partners that are also 
    in the network.
    ARGUMENTS:
        1.) <dict> {Entrez: adjacency matrix index}
        2.) <ndarray> adjacency matrix
        3.) <dict> {seed Entrez: {interactors' Entrez}}
    RETURNS:
        1.) <list> [(seed AUC, seed Entrez)]
        2.) <dict> {seed Entrez in func net: {interactors' Entrez in func net}}
    """
    seedAUC = list()
    seed2interactors = dict()
    idx2entrez = pyuserfcn.invert_dict(entrez2idx)
    numCols = adjMat.shape[1]
    for seedGene in seedSets.keys():
        if seedGene in entrez2idx:
            interactors = seedSets[seedGene]
            seedIdx = [entrez2idx[i] for i in interactors if i in entrez2idx]
            if len(seedIdx) > 0:
                llsSum = numpy.sum(adjMat[seedIdx,:], axis=0)
                trueLabels = numpy.zeros(numCols, dtype=numpy.int)
                trueLabels[seedIdx] = 1
                fpr,tpr,thresholds = metrics.roc_curve(trueLabels, llsSum)
                seedAUC.append( (metrics.auc(fpr, tpr), seedGene) )
                seed2interactors[seedGene] = [idx2entrez[x] for x in seedIdx]
    seedAUC.sort()
    return seedAUC, seed2interactors


def plot_aucs(seedAUC):
    """
    ARGUMENTS: <list> [(seed AUC, seed Entrez)]
    """
    aucs = [t[0] for t in seedAUC]
    pos = numpy.arange(1, len(aucs)+1)
    plt.barh(pos, aucs, height=1.0, align='center')
    plt.ylim([0, len(aucs)+1])
    ax = plt.axes()
    ##ax.get_yaxis().set_visible(False)
    ##ax.set_yticklabels([])
    ##plt.tick_params(axis='x', labelsize='24')
    plt.xlabel('AUC')
    plt.show()


def write_seeds_text(seed2interactors, seedAUC, expmntSys):
    """
    Writes out seed genes and their interactors in the following format:
    AUC = %.2f seed gene symbol: interactor symbols
    ARUGMENTS: 
        1.) <dict> {seed Entrez in func net: {interactors' Entrez in func net}}
        2.) <list> [(seed AUC, seed Entrez)]
        3.) <string> type of genetic interaction
    """
    AUCcutoff = 0.9
    aucs = [v[0] for v in seedAUC]  # sorted in ascending order
    cutIndex = len(aucs) - bisect.bisect_left(aucs, AUCcutoff)
    dataPath = '/work/jyoung/PyPickle/flyentrez2names.p'
    entrez2names = pickle.load(open(dataPath, 'rb'))
    expmntSys = expmntSys.replace(' ', '')
    outPath = '../results/' + expmntSys + '_seed_sets.txt'
    outFile = open(outPath, 'w')
    for i in range(1, cutIndex+1):
        outFile.write('AUC = %.2f\t' %seedAUC[-i][0])
        outFile.write(entrez2names[seedAUC[-i][1]] + ': ')
        intactEntrez = seed2interactors[seedAUC[-i][1]]
        if len(intactEntrez) > 1:
            for j in range(len(intactEntrez)-1):
                outFile.write(entrez2names[intactEntrez[j]] + ', ')
        outFile.write(entrez2names[intactEntrez[len(intactEntrez)-1]] + '\n')
    outFile.close()


def main():
    experimentSys = sys.argv[1]
    node2edgewt = process_func_net()
    entrez2idx = get_entrez_indices(node2edgewt)
    ##adjMat = build_netwk_adj_matrix(node2edgewt, entrez2idx)
    matrixPath = '../data/FlyNet_adj_matrix.npy'
    ##numpy.save(matrixPath, adjMat)
    adjMat = numpy.load(matrixPath)
    print('Number of genes in functional network:', len(entrez2idx.keys()))
    ##seedSets = read_known_interact()
    seedSets = read_biogrid(experimentSys)
    seedAUC, seed2interactors = seed_set_predictability(entrez2idx, 
            adjMat, seedSets)
    print('Number of seed sets:', len(seedAUC))
    write_seeds_text(seed2interactors, seedAUC, experimentSys)
    ##plot_aucs(seedAUC)


if __name__=="__main__":
    main()

