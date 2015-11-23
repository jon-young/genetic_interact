#!/usr/bin/env python

"""
Construct genetic interaction adjacency matrix

Created: 03 November 2015
"""

import numpy as np
import os.path


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
            'Sp_prot_cmplx_Ryan2013.2col.txt')
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
            tokens = line.rstrip().split('\t')
            if tokens[12] == 'genetic':
                allGenes.update(tokens[5:7])

        for line in open(files[1]):  # process protein complexes
            allGenes.add(line.rstrip().split('\t')[1])

    return allGenes


def make_adj(organism, intactType):
    """Construct adjacency matrix for genetic interaction network of given 
    organism and type"""
    org2path = setup_filepaths()
    
    genes = get_all_genes(org2path[organism])
    
    sortedGenes = sorted(genes)
    gene2idx = {g:i for i,g in enumerate(sortedGenes)}
    adjMat = np.zeros((len(genes), len(genes)), dtype=np.int)

    geneIntactFile = org2path[organism][0]
    for line in open(geneIntactFile):
        tokens = line.rstrip().split('\t')
        if tokens[11] == intactType:
            i, j = gene2idx[tokens[5]], gene2idx[tokens[6]]
            adjMat[i,j] = 1
            adjMat[j,i] = 1

    return adjMat, gene2idx


def get_biogrid_genes(organism, intactType):
    """Return genes from BIOGRID for a given interaction type.
    WARNING: GENE COLUMNS HARD-CODED"""
    org2path = setup_filepaths()

    intactGenes = set()
    geneIntactFile = org2path[organism][0]
    for line in open(geneIntactFile):
        tokens = line.rstrip().split('\t')
        if tokens[11] == intactType:
            intactGenes.update(tokens[5:7])

    return intactGenes

