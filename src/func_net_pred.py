#!/usr/bin/env python

"""
Predict genetic interactions using a functional gene network 
and known interactions

HARD-CODED PARAMETERS:
    1.) functional network file path
    2.) save or load adjacency matrix
    3.) BIOGRID file
    4.) columns in BIOGRID file to read
    5a.) whether to perform gene name conversion
    5b.) filepath for Entrez-to-name conversion
"""

import bisect
import collections
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pyuserfcn
import sys
from sklearn import metrics


def process_func_net():
    """
    Get nodes and edge weights of functional net and store in dictionary
    RETURNS:
        <dict> {(gene ID1, gene ID2): LLS}
    """
    node2edgewt = dict()
    netwk = '/work/jyoung/DataDownload/FunctionalNet/yeastnet2.gene.txt'
    scoreCol = 2
    for line in open(netwk):
        tokens = line.split('\t')
        node2edgewt[(tokens[0], tokens[1])] = float(tokens[scoreCol])
    return node2edgewt


def assign_gene_indices(node2edgewt):
    """
    Assign indices to genes to facilitate adjacency matrix creation
    ARGUMENTS:
        <dict> {(gene node 1, gene node 2): LLS}
    RETURNS:
        <dict> {gene: matrix index}
    """
    allGenes = set()
    for vx in node2edgewt.keys():
        allGenes.update(vx)
    geneList = sorted(allGenes)
    gene2idx = {gene: i for i, gene in enumerate(geneList)}
    return gene2idx


def build_netwk_adj_matrix(node2edgewt, gene2idx):
    """
    ARGUMENTS:
        <dict> {(gene node 1, gene node 2): LLS}
        <dict> {gene: matrix index}
    RETURNS:
        <ndarray> adjacency matrix
    """
    adjMat = np.zeros((len(gene2idx), len(gene2idx)))
    for vx in node2edgewt.keys():
        adjMat[gene2idx[vx[0]], gene2idx[vx[1]]] = node2edgewt[vx]
        adjMat[gene2idx[vx[1]], gene2idx[vx[0]]] = node2edgewt[vx]
    return adjMat


def read_known_interact(filePath):
    """
    For files containing only the interacting gene pairs on each line
    (each gene in a column)
    RETURNS:
        <dict> {seed gene: {interactors' genes}}
    """
    seedSets = collections.defaultdict(set)
    allGenes = set()
    for line in open(filePath):
        tokens = line.rstrip().split('\t')
        seedSets[tokens[0]].add(tokens[1])
        seedSets[tokens[1]].add(tokens[0])
        allGenes.update([tokens[0], tokens[1]])
    return seedSets


def read_biogrid(experimentSys):
    """
    Read BIOGRID genetic interactions and return dictionary converting each 
    interactor ID to its genetic interaction partners
    NOTE: the correct column numbers need to be specified beforehand: 1 & 2 
    for Entrez, 5 & 6 for systematic name, 7 & 8 for official symbol
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
            seedSets[tokens[7]].add(tokens[8])
            seedSets[tokens[8]].add(tokens[7])
            allGenes.update([tokens[7], tokens[8]])
    print('Number of genes in interactions:', len(allGenes))
    return seedSets


def seed_set_predictability(gene2idx, adjMat, seedSets):
    """
    For each seed gene, measure its predictability for genetic interaction by 
    AUC. Also return a dictionary converting the seed gene (provided it is in 
    the network) to the set of its genetic interaction partners that are also 
    in the network.
    ARGUMENTS:
        1.) <dict> {gene: adjacency matrix index}
        2.) <ndarray> adjacency matrix
        3.) <dict> {seed gene: {genetic interactors}}
    RETURNS:
        1.) <list> [(seed AUC, seed gene)]
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


def plot_aucs(seedAUC, experimentSys):
    """
    ARGUMENTS: <list> [(seed AUC, seed Entrez)]
    """
    aucs = [t[0] for t in seedAUC]
    pos = np.arange(1, len(aucs)+1)
    plt.barh(pos, aucs, height=1.0, align='center')
    plt.ylim([0, len(aucs)+1])
    ax = plt.axes()
    ax.set_yticklabels([])
    plt.xlabel('AUC')
    plt.ylabel('Seed sets')
    plt.tight_layout()
    ##saveName = '../results/' + experimentSys + '.svg'
    ##plt.savefig(saveName, bbox_inches='tight')
    plt.show()


def write_seeds_text(seed2interactors, seedAUC, expmntSys):
    """
    NOTE: If conversion needed, then it must be done before calling function
    Writes out seed genes and their interactors in the following format:
    AUC = %.2f seed gene symbol: interactor symbols
    ARUGMENTS: 
        1.) <dict> {seed gene in func net: [interactors in func net]}
        2.) <list> [(seed AUC, seed gene)]
        3.) <string> type of genetic interaction
    """
    AUCcutoff = 0.9
    aucs = [v[0] for v in seedAUC]  # sorted in ascending order
    cutIndex = len(aucs) - bisect.bisect_left(aucs, AUCcutoff)
    expmntSys = expmntSys.replace(' ', '')
    outPath = '../results/' + expmntSys + '_seed_sets.txt'
    outFile = open(outPath, 'w')
    for i in range(1, cutIndex+1):
        outFile.write('AUC = %.2f\t' %seedAUC[-i][0])
        outFile.write(seedAUC[-i][1] + ': ')
        interactors = seed2interactors[seedAUC[-i][1]]
        if len(interactors) > 1:
            for j in range(len(interactors)-1):
                outFile.write(interactors[j] + ', ')
        outFile.write(interactors[len(interactors)-1] + '\n')
    outFile.close()


def report_predictions(gene2idx, adjMat, seedSets):
    """
    Return predicted genetic interactors of a seed gene. If any gene ranks 
    higher in LLS than any seed gene interactor, then it is itself a 
    predicted interactor. 
    INPUT:
        1.) <dict> {gene: adjacency matrix index}
        2.) <ndarray> adjacency matrix
        3.) <dict> {seed gene: {genetic interactors}}
    OUTPUT:
        1.) <list> [(seed AUC, seed gene)]
        2.) <dict> {seed gene: [predicted interactors]}
    """
    seedAUC = list()
    seed2pred = dict()
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
                llsSort, trueSort = zip(*sorted(list(zip(llsSum, trueLabels)), 
                    reverse=True))
                trueSort = np.array(trueSort)
                maxTrue = np.amax(np.where(trueSort == 1))
                sortIdx = np.argsort(llsSum)[::-1]
                geneIdx = sortIdx[:maxTrue][np.logical_not(trueSort[:maxTrue])]
                predForSeed = [idx2gene[idx] for idx in geneIdx]
                auc = metrics.roc_auc_score(trueLabels, llsSum)
                seedAUC.append((auc, seedGene))
                seed2pred[seedGene] = predForSeed
    seedAUC.sort()
    return seedAUC, seed2pred


def eval_performance(seedAUC, seed2pred, experimentSys):
    """
    Use ROC to evaluate performance of genetic interactors predicted from known 
    interactions from a given time period. Interactions from later time periods 
    comprise the gold-standard benchmark. 
    INPUT:
        1.) <sorted list> [(seed AUC, seed gene)]
        2.) <dict> {seed gene: [predicted interactors]}
    """
    dataFolder = '/work/jyoung/genetic_interact/data/yeast_time_split/'
    fileSuffix = '-' + ''.join(experimentSys.split()) + '.txt'
    pre2007file = 'pre2007' + fileSuffix
    pre2007 = read_known_interact(dataFolder + pre2007file)
    post2009file = 'post2009' + fileSuffix
    post2009 = read_known_interact(dataFolder + post2009file)
    aucs, seedGenes = zip(*seedAUC)
    lowerLim = 0.90
    upperLim = 1.0
    start = bisect.bisect_left(aucs, lowerLim)
    end = bisect.bisect_left(aucs, upperLim)
    y_true = list()
    for i in range(start, end):
        seed = seedAUC[i][1]
        if seed in post2009:
            for pred in seed2pred[seed]:
                if pred in post2009[seed]:
                    y_true.append(1)
                else:
                    y_true.append(0)
                    ##if seed in pre2007:
                    ##    if pred not in pre2007[seed]:
                    ##        y_true.append(0)
                    ##else:
                    ##    y_true.append(0)
        else:
            y_true.append(0)
    y_score = np.array(range(1, len(y_true)+1))
    y_true = np.array(y_true)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    rocAuc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' %rocAuc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Performance of predictions for ' + experimentSys)
    plt.legend(loc='lower right')
    plt.show()


def main():
    experimentSys = sys.argv[1]
    node2edgewt = process_func_net()
    gene2idx = assign_gene_indices(node2edgewt)
    ##adjMat = build_netwk_adj_matrix(node2edgewt, gene2idx)
    matrixPath = '../data/YeastNet2_adj_matrix.npy'
    ##np.save(matrixPath, adjMat)
    adjMat = np.load(matrixPath)
    print('Number of genes in functional network:', len(gene2idx.keys()))
    ##seedSets = read_biogrid(experimentSys)
    ##seedAUC, seed2interactors = seed_set_predictability(gene2idx, adjMat, 
    ##        seedSets)
    CONVERSFLAG = 0
    if CONVERSFLAG:
        converSource = '/work/jyoung/PyPickle/humanEntrez2names.p'
        id1toid2 = pickle.load(open(converSource, 'rb'))
        # NOTE: converSource should be based from network
        seedAUC = [(p[0], id1toid2[p[1]]) for p in seedAUC]
        seed2interactors = {id1toid2[k]: list({id1toid2[x] 
            for x in seed2interactors[k]}) for k in seed2interactors} 
    ##write_seeds_text(seed2interactors, seedAUC, experimentSys)
    dataFolder = '/work/jyoung/genetic_interact/data/yeast_time_split/'
    incl0709file = 'incl0709' + '-' + ''.join(experimentSys.split()) + '.txt'
    incl0709 = read_known_interact(dataFolder + incl0709file)
    seedAUC, seed2pred = report_predictions(gene2idx, adjMat, incl0709)
    print('Number of seed sets:', len(seedAUC))
    ##plot_aucs(seedAUC, experimentSys)
    eval_performance(seedAUC, seed2pred, experimentSys)


if __name__=="__main__":
    main()

