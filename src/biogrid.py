#!/usr/bin/env python

"""
Functions for processing BIOGRID interaction files

Created: 25 November 2015
"""

import os.path


def setup_filepaths():
    """Setup filepaths for BIOGRID files for various organisms."""
    ScPath = os.path.join('..', 'data', 'BIOGRID-3.4.130-yeast-post2006.txt')
    SpPath = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
            'BIOGRID-ORGANISM-3.4.130.tab2', 
            'BIOGRID-ORGANISM-Schizosaccharomyces_pombe_972h-3.4.130.tab2.txt')
    DmPath = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
            'BIOGRID-ORGANISM-3.4.130.tab2',
            'BIOGRID-ORGANISM-Drosophila_melanogaster-3.4.130.tab2.txt')
    HsPath = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
            'BIOGRID-ORGANISM-3.4.130.tab2',
            'BIOGRID-ORGANISM-Homo_sapiens-3.4.130.tab2.txt')

    org2path = {'cerevisiae': ScPath, 'pombe': SpPath, 'melanogaster': DmPath,
            'sapiens': HsPath}

    return org2path


def get_biogrid_genes(organism, intactType):
    """Return genes from BIOGRID for a given interaction type.
    WARNING: GENE COLUMNS HARD-CODED"""
    org2path = setup_filepaths()

    intactGenes = set()
    geneIntactFile = org2path[organism]
    for line in open(geneIntactFile):
        tokens = line.rstrip().split('\t')
        if tokens[11] == intactType:
            intactGenes.update(tokens[5:7])

    return intactGenes


def get_interacting_genes(organism, intactType):
    """Return set of gene pairs that are of given interaction type.
    WARNING: GENE COLUMNS HARD-CODED"""
    org2path = setup_filepaths()

    intactSet = set()
    geneIntactFile = org2path[organism]
    for line in open(geneIntactFile):
        tokens = line.rstrip().split('\t')
        if tokens[11] == intactType:
            intactSet.add(frozenset(tokens[5:7]))

    return intactSet

