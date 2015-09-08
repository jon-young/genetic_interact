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
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pyuserfcn
import scipy.stats as stats
import sys
from sklearn import metrics


def process_func_net():
    """Get nodes and edge weights of functional net and store in dictionary
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
    """Assign indices to genes to facilitate adjacency matrix creation
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
    """ARGUMENTS:
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
    """For files containing only the interacting gene pairs on each line
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
    """Read BIOGRID genetic interactions and return dictionary converting each 
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
    """For each seed gene, measure its predictability for genetic interaction by 
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
    """ARGUMENTS: <list> [(seed AUC, seed Entrez)]"""
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
    """NOTE: If conversion needed, then it must be done before calling function
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


def read_SGA(interactionType):
    """"""
    queryGeneNameCol = 1
    arrayGeneNameCol = 3
    scoreCol = 4  # genetic interaction score (epsilon)
    pvalCol = 6
    gene2tested = collections.defaultdict(set)
    interactingPairs = set()
    folder = '/work/jyoung/DataDownload/Yeast_SGA/'
    os.chdir(folder)
    for line in open('sgadata_costanzo2009_rawdata_101120.txt'):
        tokens = line.rstrip().split('\t')
        gene2tested[tokens[queryGeneNameCol]].add(tokens[arrayGeneNameCol])
        gene2tested[tokens[arrayGeneNameCol]].add(tokens[queryGeneNameCol])
        if not math.isnan(float(tokens[scoreCol])):
            if float(tokens[scoreCol]) > 0:
                sign = 'positive'
                cutoff = 0.16
            else:
                sign = 'negative'
                cutoff = 0.12
            if sign == interactionType.strip().lower():
                score = math.fabs(float(tokens[scoreCol]))
                pval = float(tokens[pvalCol])
                if score > cutoff and pval < 0.05:
                    interactingPairs.add(frozenset({tokens[queryGeneNameCol], 
                        tokens[arrayGeneNameCol]}))
    return gene2tested, interactingPairs


def eval_time_split_pred(gene2idx, adjMat, seedSets):
    """Using the Costanzo (2010) dataset as a gold standard benchmark. Note that 
    this necessitates that the genetic interaction type is either 'Negative 
    Genetic' or 'Positive Genetic'."""
    seedScores = list()
    gene2tested, interactPairs = read_SGA('positive')
    idx2gene = pyuserfcn.invert_dict(gene2idx)
    numCols = adjMat.shape[1]
    for seedGene in seedSets.keys():
        # check if seed gene in network and if included in SGA
        if seedGene in gene2idx and seedGene in gene2tested:
            interactors = seedSets[seedGene]
            seedIdx = [gene2idx[i] for i in interactors if i in gene2idx]
            if len(seedIdx) > 0:
                llsSum = np.sum(adjMat[seedIdx, :], axis=0)
                trueLabels = np.zeros(numCols, dtype=np.int)
                trueLabels[seedIdx] = 1
                predictiveAUC = metrics.roc_auc_score(trueLabels, llsSum)
                # now examine performance of LLS predictability
                y_true = list()
                y_score = list()
                for i in range(len(llsSum)):
                    interactorGene = idx2gene[i]
                    if interactorGene in interactors:  # ignore if used to predict
                        pass
                    elif interactorGene in gene2tested[seedGene]:
                        if {seedGene, interactorGene} in interactPairs:
                            y_true.append(1)
                        else:
                            y_true.append(0)
                        y_score.append(llsSum[i])
                    else:  # not tested in SGA
                        pass
                try:
                    performanceAUC = metrics.roc_auc_score(y_true, y_score)
                    seedScores.append((seedGene, predictiveAUC, performanceAUC))
                except:
                    pass
    # plot performance by AUC of seed genes
    seedScores.sort(key=lambda f:f[1])
    seedGeneList, xaxis, yaxis = zip(*seedScores)
    scc, sig = stats.spearmanr(xaxis, yaxis)
    plt.plot(xaxis, yaxis, 'o')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Predictability of seed set')
    plt.ylabel('Performance of seed set by AUC')
    plt.title('Spearman correlation coefficient = %f' %scc)
    plt.tight_layout()
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
    ##plot_aucs(seedAUC, experimentSys)
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
    fileSuffix = '-' + ''.join(experimentSys.split()) + '.txt'
    incl0709file = 'incl0709' + fileSuffix
    incl0709 = read_known_interact(dataFolder + incl0709file)
    eval_time_split_pred(gene2idx, adjMat, incl0709)


if __name__=="__main__":
    main()

