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
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
import biogrid
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
    """Calculate between-cluster interaction from binomial probability"""
    numGenes = len(biogrid.get_biogrid_genes(organism, intactType))
    p = len(intactSet)/scipy.special.binom(numGenes, 2)
    print('\nThe background probability is', p)
    
    results = list()
    for i, pair in enumerate(itertools.combinations(clust2genes.keys(), 2)):
        geneset0 = clust2genes[pair[0]]
        geneset1 = clust2genes[pair[1]]
        count = sum(1 for genePair in itertools.product(geneset0, geneset1) 
                if frozenset(genePair) in intactSet)
        n = len(geneset0) * len(geneset1)
        pval = stats.binom.pmf(count, n, p) + stats.binom.sf(count, n, p)
        results.append((pair, pval))
    print('\nExamined', i+1, 'cluster pairs.')

    return results


def within_interact_binom():
    """Compute within-cluster interaction from binomial probability"""
    numGenes = len(biogrid.get_biogrid_genes(organism, intactType))
    p = len(intactSet)/scipy.special.binom(numGenes, 2)
    print('\nThe background probability is', p)
    
    results = list()
    for c in clust2genes.keys():
        count = sum(1 for genePair in itertools.combinations(clust2genes[c], 2) 
                if frozenset(genePair) in intactSet)
        n = len(clust2genes[c])
        pval = stats.binom.pmf(count, n, p) + stats.binom.sf(count, n, p)
        results.append((c, pval))

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
print('\nRead', len(clust2genes), 'clusters.')

intactType = input('\nEnter type of genetic interaction:\n')
intactSet = biogrid.get_interacting_genes(organism, intactType)

results = sorted(btw_interact_binom(), key=lambda f: f[1])

pvals = [t[1] for t in results]
rejected, pvalsCor = fdrcorrection0(pvals, is_sorted=True)
print('\nNumber of significant p-values (5% FDR, Benjamini-Hochberg):', 
        np.sum(rejected), '\n')

