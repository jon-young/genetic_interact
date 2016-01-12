# -*- coding: utf-8 -*-
"""
Find significant between- and within-cluster genetic interactions

Created on Sat Nov 14 19:11:40 2015

@author: jyoung
"""

import bisect
import itertools
import numpy as np
import os.path
import random
import re
import scipy.special
import scipy.stats as stats
import sys
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
import biogrid
import genesets


def setup_filepaths(organism):
    """Establish full paths for input gene set file"""
    if organism == 'cerevisiae':
        filepath = os.path.join('..', '..', 'DataProcessed', 
                'Sc_prot_cmplx_Hart2007.txt')
    elif organism == 'pombe':
        filepath = os.path.join('..', '..', 'DataProcessed', 
                'Sp_prot_cmplx_Ryan2013.2col.txt')
    elif organism == 'sapiens':
        filepath = os.path.join('..', '..', 'DataProcessed', 
                'CORUM_Human_Entrez.txt')
    else:
        print('\nORGANISM NOT FOUND. EXITING...\n')
        sys.exit()

    return filepath


def determine_col(organism, clustFile):
    """Determine which gene column in the BIOGRID file to read"""
    gene = open(clustFile).readline().split('\t')[1].rstrip()
    entrezRegEx = re.compile(r'\d+')
    if organism == 'cerevisiae':
        sysNameRegEx = re.compile(r'Y[A-Z][A-Z]\d+')
        ofcSymRegEx = re.compile(r'[A-Z]+')
    elif organism == 'pombe':
        sysNameRegEx = re.compile(r'SP[AB]C\d.')
        ofcSymRegEx = re.compile(r'[a-z]+')
    else:  # organism == 'sapiens'
        sysNameRegEx = re.compile(r'\w+')
        ofcSymRegEx = re.compile(r'[A-Za-z]+.')
    
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


def get_background_probability(organism, intactType, intactSet):
    numGenes = len(set(itertools.chain.from_iterable(intactSet)))
    p = len(intactSet)/scipy.special.binom(numGenes, 2)
    print('\nThe background probability is', p)

    return p


def sparsity_withhold(intactSet, pctWithheld):
    if pctWithheld < 0 or pctWithheld >= 100:
        print('Percent withheld must be >= 0 and < 100. Exiting...')
        sys.exit()
    setSize = len(intactSet)
    numKeep = setSize - round((pctWithheld/100) * setSize)
    
    return set(random.sample(intactSet, numKeep))


def btw_interact_binom(clust2genes, intactSet, p):
    """Calculate between-cluster interaction from binomial probability"""    
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


def within_interact_binom(clust2genes, intactSet, p):
    """Compute within-cluster interaction from binomial probability"""
    results = list()
    for c in clust2genes.keys():
        count = sum(1 for genePair in itertools.combinations(clust2genes[c], 2) 
                if frozenset(genePair) in intactSet)
        n = scipy.special.binom(len(clust2genes[c]), 2)
        pval = stats.binom.pmf(count, n, p) + stats.binom.sf(count, n, p)
        results.append((c, pval))

    return results


def main():
    print('\nChoose from the following organisms (enter species name):')
    print('1) cerevisiae')
    print('2) pombe')
    print('3) sapiens')
    organism = input()
    
    clustFile = setup_filepaths(organism)
    clust2genes = genesets.process_file(clustFile)
    print('\nRead', len(clust2genes), 'clusters.')
    
    colName = determine_col(organism, clustFile)
    intactType = input('\nEnter type of genetic interaction:\n')
    intactSet = biogrid.get_interacting_genes(organism, intactType, colName)
    
    bkgrdPr = get_background_probability(organism, intactType, intactSet)
    
    # NOTE: Background probability to be determined before sparsity withholding
    print('\nEvaluate effect of genetic interaction sparsity? [Y/n]')
    sparsAns = input()
    if sparsAns == 'Y':
        print('Enter the percent of genetically interacting pairs to withhold:')
        pctWithheld = int(input())
        intactSet = sparsity_withhold(intactSet, pctWithheld)
    else:
        pass
    
    ans = input('\nExamine between- or within-cluster interactions? [b/w] ')
    if ans == 'b':
        print('\nComputing BETWEEN-cluster interactions...')
        results = sorted(btw_interact_binom(clust2genes, intactSet, bkgrdPr), 
                key=lambda f: f[1])
    else:
        print('\nComputing WITHIN-cluster interactions...')
        results = sorted(within_interact_binom(clust2genes, intactSet, bkgrdPr), 
                key=lambda f: f[1])
    
    pvals = [t[1] for t in results]
    rejected, pvalsCor = fdrcorrection0(pvals, is_sorted=True)
    print('\nNumber of significant p-values (5% FDR, Benjamini-Hochberg):', 
            np.sum(rejected), '\n')


if __name__=="__main__":
    main()

