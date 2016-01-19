#!/usr/bin/env python
"""
Predict genetic interactions using a functional gene network 
and known interactions

@author: jyoung
"""

import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
from sklearn import metrics


def setup_filepaths(organism):
    """Setup full file paths for functional net and BIOGRID"""
    if organism == 'cerevisiae':
        biogridpath = os.path.join('..', 'data', 
                'BIOGRID-3.4.130-yeast-post2006.txt')
        fnetpath = os.path.join('..', 'data', 'YeastNetDataFrame.pkl')
    elif organism == 'sapiens':
        biogridpath = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
                'BIOGRID-ORGANISM-3.4.130.tab2', 
                'BIOGRID-ORGANISM-Homo_sapiens-3.4.130.tab2.txt')
        fnetpath = os.path.join('..', 'data', 'HumanNetDataFrame.pkl')
    elif organism == 'melanogaster':
        biogridpath = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
                'BIOGRID-ORGANISM-3.4.130.tab2', 
                'BIOGRID-ORGANISM-Drosophila_melanogaster-3.4.130.tab2.txt')
        fnetpath = os.path.join('..', 'data', 'FlyNetDataFrame.pkl')
    else:
        print('ORGANISM NOT FOUND! Exiting...')
        sys.exit()

    return biogridpath, fnetpath


def determine_col(organism, gene):
    """Determine which gene column in the BIOGRID file to read"""
    entrezRegEx = re.compile(r'\d+')
    if organism == 'cerevisiae':
        sysNameRegEx = re.compile(r'Y[A-Z][A-Z]\d+')
        ofcSymRegEx = re.compile(r'[A-Z]+')
    elif organism == 'sapiens':
        sysNameRegEx = re.compile(r'\w+')
        ofcSymRegEx = re.compile(r'[A-Za-z]+.')
    else:  # organism == 'melanogaster'
        sysNameRegEx = re.compile(r'Dmel_.')
        ofcSymRegEx = re.compile(r'\w+')
    
    if entrezRegEx.match(gene) is not None:
        colName = 'Entrez Gene Interactor A'
    elif sysNameRegEx.match(gene) is not None:
        colName = 'Systematic Name Interactor A'
    elif ofcSymRegEx.match(gene) is not None:
        colName = 'Official Symbol Interactor A'
    else:
        print('ERROR: Unable to match ID type! Exiting...')
        sys.exit()

    return colName


def read_biogrid(biogridpath, experimentSys, colName):
    """Read BIOGRID genetic interactions and return dictionary converting each 
    interactor ID to its genetic interaction partners"""
    seedSets = collections.defaultdict(set)
    biogridfile = open(biogridpath)
    header = biogridfile.readline().split('\t')
    geneColNum = header.index(colName)
    expSysColNum = header.index('Experimental System')
    for line in biogridfile:
        tokens = line.split('\t')
        if tokens[expSysColNum] == experimentSys:
            seedSets[tokens[geneColNum]].add(tokens[geneColNum + 1])
            seedSets[tokens[geneColNum + 1]].add(tokens[geneColNum])

    return seedSets


def seed_set_predictability(funcNetDf, seedSets):
    """For each seed gene, measure its predictability of genetic interactions 
    by AUC. Also return a dictionary converting the seed gene (provided it is 
    in the network) to the set of its known genetic interaction partners."""
    seedAUC = list()
    seed2interactors = dict()
    for seedGene in seedSets.keys():
        interactors = [gene for gene in seedSets[seedGene] 
                if gene in funcNetDf.index]
        if len(interactors) > 0:
            llsSum = funcNetDf.loc[interactors,:].sum(axis=0)
            trueLabels = pd.Series([0]*llsSum.size, index=llsSum.index)
            trueLabels.loc[interactors] = 1
            auc = metrics.roc_auc_score(trueLabels, llsSum)
            seedAUC.append((auc, seedGene))
            seed2interactors[seedGene] = interactors
    seedAUC.sort()

    return seedAUC, seed2interactors


def plot_aucs(seedAUC):
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
    plt.show()


def main():
    if len(sys.argv) < 3:  # validate input parameters
        print('Usage: python {} <organism>'\
                ' <genetic interaction>'.format(sys.argv[0]))
        sys.exit()
    
    organism = sys.argv[1]
    intactType = sys.argv[2]
    
    biogridpath, fnetpath = setup_filepaths(organism)
    funcNetDf = pd.read_pickle(fnetpath)

    numNodes = len(funcNetDf.columns)
    print('\nNumber of genes in functional network: {}'.format(numNodes))

    geneExample = funcNetDf.columns[0]
    colName = determine_col(organism, geneExample)
    seedSets = read_biogrid(biogridpath, intactType, colName)
    seedAUC, seed2intacts = seed_set_predictability(funcNetDf, seedSets)
    
    print('Number of seed sets: {}\n'.format(len(seedAUC)))

    plot_aucs(seedAUC)


if __name__=="__main__":
    main()

