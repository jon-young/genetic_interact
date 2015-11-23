# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:11:40 2015

@author: jyoung
"""

import bisect
import itertools
import numpy as np
import os.path
import scipy.special
import scipy.stats as stats
import geneintactmatrix
import genesets


def setup_filepaths():
    """Establish full paths for input gene set file"""
    if clustType == 'protein complexes':
        if organism == 'cerevisiae':
            filepath = os.path.join('..', '..', 'DataProcessed', 
                    'Sc_prot_cmplx_Hart2007.txt')
        elif organism == 'pombe':
            filepath = os.path.join('..', '..', 'DataProcessed', 
                    'Sp_prot_cmplx_Ryan2013.2col.txt')
        else:
            filepath = ''
    else:  # clusters from functional net
        if organism == 'cerevisiae':
            filepath = ''
        else:
            filepath = ''
    
    return filepath


def btw_interact_binom():
    """Calculate cluster interaction from binomial probability"""
    numGenes = len(geneintactmatrix.get_biogrid_genes(organism, intactType))
    p = (np.sum(adjMat)/2)/scipy.special.binom(numGenes, 2)
    
    results = list()
    for pair in itertools.combinations(clust2genes.keys(), 2):
        numClust1 = len(clust2genes[pair[0]])
        numClust2 = len(clust2genes[pair[1]])
        
        # compute interaction counts
        clustIdx1 = [gene2idx[g] for g in clust2genes[pair[0]]]
        clustIdx2 = [gene2idx[g] for g in clust2genes[pair[1]]]
        count = np.sum(adjMat[np.ix_(clustIdx1, clustIdx2)])

        n = numClust1 * numClust2
        pval = stats.binom.pmf(count, n, p) + stats.binom.sf(count, n, p)
        results.append((pair, pval))

    return results


print('\nChoose from the following organisms (enter species name):')
print('1) cerevisiae')
print('2) pombe')
print('3) melanogaster')
print('4) sapiens')
organism = input()
clustType = input('\nUse "functional net" or "protein complexes" for clusters?\n')

clustFile = setup_filepaths()
clust2genes = genesets.process_file(clustFile)

intactType = input('Enter type of genetic interaction:\n')
adjMat, gene2idx = geneintactmatrix.make_adj(organism, intactType)

results = btw_interact_binom()

