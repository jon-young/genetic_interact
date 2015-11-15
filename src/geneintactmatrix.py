#!/usr/bin/env python

"""
Created: 03 November 2015
"""

import numpy as np
import os.path
import sys


def setup_filepaths():
    """Establish full file paths for all input data files"""
    ScGIfile = os.path.join('..', 'data', 'BIOGRID-3.4.130-yeast-post2006.txt')
    SpGIfile = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
            'BIOGRID-ORGANISM-3.4.130.tab2', 
            'BIOGRID-ORGANISM-Schizosaccharomyces_pombe_972h-3.4.130.tab2.txt')
    DmGIfile = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
            'BIOGRID-ORGANISM-3.4.130.tab2',
            'BIOGRID-ORGANISM-Drosophila_melanogaster-3.4.130.tab2.txt')
    HsGIfile = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
            'BIOGRID-ORGANISM-3.4.130.tab2',
            'BIOGRID-ORGANISM-Homo_sapiens-3.4.130.tab2.txt')
    
    ScProtFile = os.path.join('..', '..', 'DataProcessed', 
            'Sc_prot_cmplx_Hart2007.txt')
    SpProtFile = os.path.join('..', '..', 'DataProcessed', 
            'Sp_prot_cmplx_Ryan2013.txt')
    DmProtFile = ''
    HsProtFile = ''
    
    ScFnetFile = os.path.join('..', '..', 'DataDownload', 'FunctionalNet', 
            'yeastnet2.gene.txt')
    DmFnetFile = os.path.join('..', 'data', 'FlyNetEntrez-noNull.txt')
    HsFnetFile = os.path.join('..', 'data', 'H6Net_CC.net')

    org2path = {'cerevisiae': (ScGIfile, ScProtFile, ScFnetFile),
           'pombe': (SpGIfile, SpProtFile),
           'melanogaster': (DmGIfile, DmProtFile, DmFnetFile),
           'sapiens': (HsGIfile, HsProtFile, HsFnetFile)}

    return org2path


def get_all_genes(files):
    """Assemble all genes from genetic interactions, protein complexes and 
    functional net for given organism"""
    allGenes = set()
    if len(files) == 3:  # functional net exists for organism
        for line in open(files[0]):  # process genetic interactions
            tokens = line.rstrip().split('\t')
            if tokens[12] == 'genetic':
                allGenes.update(tokens[7:9])

        for line in open(files[1]):  # process protein complexes
            allGenes.add(line.rstrip().split('\t')[1])

        for line in open(files[2]):  # process functional net
            allGenes.update(line.split('\t')[:2])
    else:  # functional net not available
        for line in open(files[0]):  # process genetic interactions
            allGenes.update(line.split('\t')[7:9])

        for line in open(files[1]):  # process protein complexes
            allGenes.add(line.rstrip().split('\t')[1])

    return allGenes


def GI_matrix(intactType, filename, genes):
    """Construct adjacency matrix for genetic interaction network of given 
    organism and type"""
    sortedGenes = sorted(genes)
    gene2idx = {g:i for i,g in enumerate(sortedGenes)}
    adjMat = np.zeros((len(genes), len(genes)), dtype=np.int)

    for line in open(filename):
        tokens = line.rstrip().split('\t')
        if tokens[11] == intactType:
            i = gene2idx[tokens[7]]
            j = gene2idx[tokens[8]]
            adjMat[i,j] = 1
            adjMat[j,i] = 1

    return adjMat, gene2idx


def main():
    """NOTES: protein complexes have been already pre-processed into 
    common-format files"""
    print('\nChoose from the following organisms (enter species name):')
    print('1. cerevisiae')
    print('2. pombe')
    print('3. melanogaster')
    print('4. sapiens')
    organism = input()
    intactType = input('\nEnter desired genetic interaction type:\n')
    
    org2path = setup_filepaths()

    allGenes = get_all_genes(org2path[organism])
    print('Number of genes in adjacency matrix:', len(allGenes))
    
    adjMat, gene2idx = GI_matrix(intactType, org2path[organism][0], allGenes)


if __name__=="__main__":
    main()

