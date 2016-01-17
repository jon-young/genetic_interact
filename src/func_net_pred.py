#!/usr/bin/env python

"""
Predict genetic interactions using a functional gene network 
and known interactions

HARD-CODED PARAMETERS:
    1.) functional network file path
    2.) BIOGRID file
    3.) columns in BIOGRID file to read
    4a.) whether to perform gene name conversion
    4b.) filepath for Entrez-to-name conversion
"""

import bisect
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pyuserfcn
import sys
from sklearn import metrics


def process_func_net(netwkFile):
    """Get nodes and edge weights of functional net and store in dictionary
    RETURNS:
        <dict> {(gene ID1, gene ID2): LLS}
    """
    node2edgewt = dict()
    #netwkDir = os.path.join('..', '..', 'DataDownload', 'FunctionalNet', '')
    netwkDir = os.path.join('..', 'data', '')
    scoreCol = 2
    for line in open(netwkDir + netwkFile):
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


def read_biogrid(experimentSys, filename):
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
    fileDir = os.path.join('..', 'data', '')
    experimentalSysColNum = 11
    for line in open(fileDir + filename):
        tokens = line.split('\t')
        if tokens[experimentalSysColNum] == experimentSys:
            seedSets[tokens[1]].add(tokens[2])
            seedSets[tokens[2]].add(tokens[1])
            allGenes.update([tokens[1], tokens[2]])
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
    folder = os.path.join('..', '..', 'DataDownload', 'Yeast_SGA', '')
    fname = 'sgadata_costanzo2009_rawdata_101120.txt'
    for line in open(folder + fname):
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


def eval_time_split_pred(typeForEval, gene2idx, adjMat, seedSets):
    """Using the Costanzo (2010) dataset as a gold standard benchmark. Note that 
    this necessitates that the genetic interaction type is either 'Negative 
    Genetic' or 'Positive Genetic'."""
    y_true_all = list()
    y_score_all = list()
    gene2tested, interactPairs = read_SGA(typeForEval)
    idx2gene = pyuserfcn.invert_dict(gene2idx)
    numCols = adjMat.shape[1]
    predAucCut = 0.0
    for seedGene in seedSets.keys():
        # check if seed gene in network and if included in SGA
        if seedGene in gene2idx and seedGene in gene2tested:
            interactors = seedSets[seedGene]
            seedIdx = [gene2idx[i] for i in interactors if i in gene2idx]
            if len(seedIdx) > 0:
                llsSum = np.sum(adjMat[seedIdx, :], axis=0)
                trueLabels = np.zeros(numCols, dtype=np.int)
                trueLabels[seedIdx] = 1
                predAuc = metrics.roc_auc_score(trueLabels, llsSum)
                if predAuc >= predAucCut:
                    # now make true/false calls on interactions
                    y_true = list()
                    y_score = list()
                    for i in range(len(llsSum)):
                        interactorGene = idx2gene[i]
                        if interactorGene in interactors:
                            pass  # ignore if used to predict
                        elif interactorGene in gene2tested[seedGene]:
                            if {seedGene, interactorGene} in interactPairs:
                                y_true.append(1)
                            else:
                                y_true.append(0)
                            y_score.append(llsSum[i])
                        else:  # not tested in SGA
                            pass
                    y_true_all.extend(y_true)
                    y_score_all.extend(y_score)
    y_true_all = np.array(y_true_all)
    y_score_all = np.array(y_score_all)
    # test y_true_all and y_score_all are same size
    if y_true_all.size == y_score_all.size:
        print('y_true_all and y_score_all are the same size.')
        print('Size of y_true_all:', y_true_all.size)
        print('Size of y_score_all:', y_score_all.size, '\n')
    else:
        print('y_true_all and y_score_all are NOT the same size. Exiting...\n')
        sys.exit()
    # plot performance by AUC of seed genes
    fpr, tpr, threshROC = metrics.roc_curve(y_true_all, y_score_all)
    auc = metrics.auc(fpr, tpr)
    prec, rec, threshPR = metrics.precision_recall_curve(y_true_all, y_score_all)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label='ROC curve (area=%0.2f)' %auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC performance of predictions for %s genetic' %typeForEval)
    plt.legend(loc='lower right')
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall of predictions for %s genetic' %typeForEval)
    plt.show()


def main():
    experimentSys = sys.argv[1]
    organism = sys.argv[2].strip().lower()
    org2files = {'human': ('H6Net_CC.net', 
                           'HumanNet2_adj_matrix.npy', 
                           'BIOGRID-3.4.127-human.txt'),
                 'yeast': ('yeastnet2.gene.txt', 
                           'YeastNet2_adj_matrix.npy', 
                           'BIOGRID-3.4.127-for_yeastnetv2.txt')}
    if organism not in org2files:
        print('Organism not found.')
        print('Available organisms: human, yeast\n')
        sys.exit()
    netwkFile = org2files[organism][0]
    matrixFile = org2files[organism][1]
    biogridFile = org2files[organism][2]
    node2edgewt = process_func_net(netwkFile)
    gene2idx = assign_gene_indices(node2edgewt)
    print('\nNumber of genes in functional network:', len(gene2idx.keys()), '\n')
    matrixPath = os.path.join('..', 'data', 'YeastNet2_adj_matrix.npy')
    try:
        adjMat = np.load(matrixPath)
    except:
        print('Network file not found. Creating network matrix...\n')
        adjMat = build_netwk_adj_matrix(node2edgewt, gene2idx)
        np.save(matrixPath, adjMat)
    ##seedSets = read_biogrid(experimentSys, biogridFile)
    ##seedAUC, seed2intacts = seed_set_predictability(gene2idx, adjMat, seedSets)
    ##plot_aucs(seedAUC, experimentSys)
    CONVERSFLAG = 0
    if CONVERSFLAG:
        converSource = '/work/jyoung/PyPickle/humanEntrez2names.p'
        id1toid2 = pickle.load(open(converSource, 'rb'))
        # NOTE: converSource should be based from network
        seedAUC = [(p[0], id1toid2[p[1]]) for p in seedAUC]
        seed2interactors = {id1toid2[k]: list({id1toid2[x] 
            for x in seed2interactors[k]}) for k in seed2interactors} 
    ##write_seeds_text(seed2intacts, seedAUC, experimentSys)
    dataFolder = os.path.join('..', 'data', 'yeast_time_split', '')
    fileSuffix = '-' + ''.join(experimentSys.split()) + '.txt'
    incl0709file = 'incl0709' + fileSuffix
    incl0709 = read_known_interact(dataFolder + incl0709file)
    if organism == 'yeast':
        typeForEval = experimentSys.split()[0].lower()
        if typeForEval == 'negative' or typeForEval == 'positive':
            eval_time_split_pred(typeForEval, gene2idx, adjMat, incl0709)
        else:
            print('Interaction type for prediction performance evaluation') 
            print('must be either "Negative Genetic" or "Positive Genetic."')
            print('No validation performed. Exiting...\n')
            sys.exit()
    else:
        print('For validation, organism must be yeast. Exiting...\n')
        sys.exit()


if __name__=="__main__":
    main()

