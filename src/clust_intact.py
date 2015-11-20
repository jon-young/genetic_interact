# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:11:40 2015

@author: jyoung
"""

import bisect
import itertools
import numpy as np
import os.path
import geneintactmatrix
import genesets


def setup_filepaths(clustType, organism):
    """Establish full paths for input gene set file"""
    if clustType == 'protein complexes':
        if organism == 'cerevisiae':
            filepath = os.path.join('..', '..', 'DataProcessed', 'Sc_prot_cmplx_Hart2007.txt')
        elif organism == 'pombe':
            filepath = os.path.join('..', '..', 'DataProcessed', 'Sp_prot_cmplx_Ryan2013.txt')
        else:
            filepath = ''
    else:  # clusters from functional net
        if organism == 'cerevisiae':
            filepath = ''
        else:
            filepath = ''
    
    return filepath


def between_interact_stats(clust2genes, adjMat, gene2idx):
    """Calculate significane of interactions between clusters"""
    genes = list(gene2idx.keys())    
    NUMRAND = 10000
    scores = dict()
    
    for clustPair in itertools.combinations(clust2genes.keys(), 2):
        allRandCounts = list()        
        # randomization
        for n in range(NUMRAND):
            shufdict = dict(zip(np.random.permutation(genes), gene2idx.values()))
            randIdx1 = [shufdict[g] for g in clust2genes[clustPair[0]]]
            randIdx2 = [shufdict[g] for g in clust2genes[clustPair[1]]]
            allRandCounts.append(np.sum(adjMat[np.ix_(randIdx1, randIdx2)]))
        allRandCounts.sort()
        
        # calculate actual counts
        clustIdx1 = [gene2idx[g] for g in clust2genes[clustPair[0]]]
        clustIdx2 = [gene2idx[g] for g in clust2genes[clustPair[1]]]
        actualCount = np.sum(adjMat[np.ix_(clustIdx1, clustIdx2)])
        
        rank = bisect.bisect_left(allRandCounts, actualCount)
        scores[clustPair] = (NUMRAND - rank + 1)/(NUMRAND + 1)
    
    return scores


def main():
    print('\nChoose from the following organisms (enter species name):')
    print('1) cerevisiae')
    print('2) pombe')
    print('3) melanogaster')
    print('4) sapiens')
    organism = input()
    clustType = input('\nUse "functional net" or "protein complexes" for clusters?\n')

    clustFile = setup_filepaths(clustType, organism)
    clust2genes = genesets.process_file(clustFile)
    
    intactType = input('Enter type of genetic interaction:\n')
    adjMat, gene2idx = geneintactmatrix.make_adj(organism, intactType)
    
    pvals = between_interact_stats(clust2genes, adjMat, gene2idx)


if __name__=="__main__":
    main()
