#!/usr/bin/env python
"""
Functions for processing BIOGRID interaction files

Created: 25 November 2015

@author: jyoung
"""

import os.path


def setup_filepaths(organism):
    """Setup filepaths for BIOGRID files for various organisms."""
    if organism == 'cerevisiae':
        biogridPath = os.path.join('..', 'data', 
                'BIOGRID-3.4.130-yeast-post2006.txt')
    elif organism == 'pombe':
        biogridPath = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
                'BIOGRID-ORGANISM-3.4.130.tab2', 
                'BIOGRID-ORGANISM-Schizosaccharomyces_pombe_972h-3.4.130.tab2'\
                        '.txt')
    elif organism == 'melanogaster':
        biogridPath = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
                'BIOGRID-ORGANISM-3.4.130.tab2', 
                'BIOGRID-ORGANISM-Drosophila_melanogaster-3.4.130.tab2.txt')
    else:  # organism == 'sapiens'
        biogridPath = os.path.join('..', '..', 'DataDownload', 'BIOGRID', 
                'BIOGRID-ORGANISM-3.4.130.tab2', 
                'BIOGRID-ORGANISM-Homo_sapiens-3.4.130.tab2.txt')

    return biogridPath


def get_interacting_genes(organism, intactType, colName):
    """Return set of gene pairs that are of given interaction type"""
    intactSet = set()
    
    biogridFile = open(setup_filepaths(organism))
    header = biogridFile.readline().rstrip().split('\t')
    colNum = header.index(colName)
    intactTypeCol = header.index('Experimental System')
    
    for line in biogridFile:
        tokens = line.rstrip().split('\t')
        if tokens[intactTypeCol] == intactType:
            intactSet.add(frozenset(tokens[colNum:colNum + 2]))
    biogridFile.close()

    return intactSet

