#!/usr/bin/env python

"""
COMMAND LINE ARGS: "genetic interaction type" "organism (i.e. human, yeast)"

For a given species and genetic interaction type, compute statistics about 
the significance of interactions between a pair of gene clusters. A cluster 
could be genes tightly connected in a functional gene network. 

Created: 10 September 2015

HARD-CODED PARAMETERS:
    1.) organism common names and filenames
    2.) functional network directory
    3.) BIOGRID directory
    4.) columns in BIOGRID file to read
    5.) AUC upper and lower limits for predictability
"""

import collections
import csv
import itertools
import numpy as np
import os
import pyuserfcn
import scipy.special
import scipy.stats as stats
import sys
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from sklearn import metrics


def process_func_net(netwkFile):
    """Get nodes and edge weights of functional net and store in dictionary
    RETURNS:
        <dict> {(gene ID1, gene ID2): LLS}
    """
    node2edgewt = dict()
    netwkDir = os.path.join('..', '..', 'DataDownload', 'FunctionalNet', '')
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
    RETURNS: <list> [genes] ascend sort by AUC
    """
    predictiveSeeds = list()
    for p in seedAUC:  # p=(AUC, gene)
        if p[0] >= lowerLim and p[0] < upperLim:
            predictiveSeeds.append(p[1])
    print('Number of predictive seed genes:', len(predictiveSeeds), '\n')
    return predictiveSeeds


def jaccard(s1, s2):
    """Calculate Jaccard index between 2 sets
    INPUT: 1.) & 2.) <set>
    RETURN: <float>"""
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
    while len(newSetOfSets) != len(setOfSets):
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
    return setOfSets


def get_interacting_pairs(seeds, seed2interactors):
    """Assemble all the genetic interactions for genes in the network into a 
    single set.
    INPUT:
        1.) <iterable> seed genes
        2.) <dict> {seed gene: [interactor genes]}
    RETURNS: <set> {(seed, interactor), (interactor, seed)}"""
    interactPairs = set()
    for seed in seeds:
        for interactor in seed2interactors[seed]:
            interactPairs.update([(seed, interactor), (interactor, seed)])
    return interactPairs


def interaction_stats(interactPairs, setOfSets):
    """Get statistics of interactions between predictive seed sets
    INPUT:
        1.) <dict> {seed gene: [interactor genes]}
        2.) <list> [genes] ascend sort by AUC
        3.) <set> {(seed, interactor), (interactor, seed)}
    RETURNS: <list> [(1st set cnts, 2nd set cnts, num interacts, p-value)]"""
    numGenes = len(set(itertools.chain.from_iterable(interactPairs)))
    p = (len(interactPairs)/2)/scipy.special.binom(numGenes, 2)
    results = list()
    for setPair in itertools.combinations(setOfSets, 2):
        count = 0
        num1stSet, num2ndSet = len(setPair[0]), len(setPair[1])
        for genePair in itertools.product(setPair[0], setPair[1]):
            if genePair in interactPairs:
                count += 1
        n = num1stSet * num2ndSet
        pval = stats.binom.pmf(count, n, p) + stats.binom.sf(count, n, p)
        results.append((num1stSet, num2ndSet, count, pval))
    return results


def main():
    experimentSys = sys.argv[1]
    organism = sys.argv[2].strip().lower()
    org2files = {'human': ('H6Net_CC.net', 
                           'HumanNet2_adj_matrix.npy', 
                           'BIOGRID-3.4.127-human.txt'),
                 'yeast': ('yeastnet2.gene.txt', 
                           'YeastNet2_adj_matrix.npy', 
                           'BIOGRID-3.4.127-for-yeastnetv2.txt')}
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
    interactPairs = get_interacting_pairs(seedGenes, seed2intacts)
    
    lowerAUC = 0.8
    upperAUC = 1.0
    predictiveSeeds = get_predictive_seeds(seedAUC, lowerAUC, upperAUC)
    
    setOfSets = set()
    for seed in predictiveSeeds:
        setOfSets.add( frozenset(set(seed2intacts[seed])) )
    combinedSets = combine_sets(setOfSets)

    results = interaction_stats(interactPairs, combinedSets)
    print('Number of set pairs:', len(results), '\n')

    # check for statistical significance
    pvals = [x[3] for x in results]
    rejected, pvalsCor = fdrcorrection0(pvals)
    numSig = np.sum(rejected)
    print('Number of significant interacting pairs (5% FDR):', numSig, '\n')


if __name__=="__main__":
    main()

