# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:11:40 2015

@author: jyoung
"""

import bisect
import itertools
import multiprocessing as mp
import numpy as np
import os.path
import pickle
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


def divide_chunks(vec, n):
    """Divide vector into evenly-sized chunks"""
    # First get lengths of each chunk
    numDivFloor = len(vec)//n
    lengths = [numDivFloor] * n
    remainder = len(vec) - numDivFloor * n
    for i in range(remainder):
        lengths[i] += 1
    divpos = np.cumsum([0] + lengths)  # indices for chunk divisions

    # now divide into chunks
    for i in range(len(divpos)-1):
        yield vec[divpos[i]:divpos[i+1]]


def between_interact_stats(pairs):
    """Calculate significance of interactions between clusters"""
    genes = tuple(gene2idx.keys())
    NUMRAND = 10000
    pairpvals = list()
    
    for p in pairs:
        allRandCounts = list()        
        # randomization
        for n in range(NUMRAND):
            shufdict = dict(zip(np.random.permutation(genes), gene2idx.values()))
            randIdx1 = [shufdict[g] for g in clust2genes[p[0]]]
            randIdx2 = [shufdict[g] for g in clust2genes[p[1]]]
            allRandCounts.append(np.sum(adjMat[np.ix_(randIdx1, randIdx2)]))
        allRandCounts.sort()
        
        # calculate actual counts
        clustIdx1 = [gene2idx[g] for g in clust2genes[p[0]]]
        clustIdx2 = [gene2idx[g] for g in clust2genes[p[1]]]
        actualCount = np.sum(adjMat[np.ix_(clustIdx1, clustIdx2)])
        
        rank = bisect.bisect_left(allRandCounts, actualCount)
        pairpvals.append(( p, (NUMRAND - rank + 1)/(NUMRAND + 1) ))

    return pairpvals


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

clustPairs = tuple(itertools.combinations(clust2genes.keys(), 2))

NUMPROC = 4
pool = mp.Pool(processes=NUMPROC)
chunks = tuple(divide_chunks(clustPairs, NUMPROC))
results = [pool.apply_async(between_interact_stats, args=(chunks[jobID],)) 
        for jobID in range(NUMPROC)]
output = [p.get() for p in results]

pickle.dump(output, open('../tmp/pombe_1st_run.p', 'wb'))

