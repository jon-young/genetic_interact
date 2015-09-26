#!/usr/bin/env python

"""
Compare using different versions of functional gene networks for prediction of 
genetic interactions. 

Created: 2015 September 22
"""

import func_net_pred
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def get_seed_set_aucs(filenames):
    netwkFile = filenames[0]
    matrixFile = filenames[1]
    node2edgewt = func_net_pred.process_func_net(netwkFile)
    gene2idx = func_net_pred.assign_gene_indices(node2edgewt)
    matrixPath = os.path.join('..', 'data', matrixFile)
    try:
        adjMat = np.load(matrixPath)
    except:
        print('Network file not found. Creating network matrix...\n')
        adjMat = func_net_pred.build_netwk_adj_matrix(node2edgewt, gene2idx)
        np.save(matrixPath, adjMat)
    seedAUC, seed2intacts = func_net_pred.seed_set_predictability(gene2idx, 
                                                                  adjMat, 
                                                                  seedSets)
    return {p[1]: p[0] for p in seedAUC}


def plot_auc_comparison(version2aucs):
    x_coords = list()
    y_coords = list()
    versionNums = list(version2aucs.keys()).sort()
    for gene in version2aucs[versionNums[0]].keys():
        if gene in version2aucs[versionNums[1]]:
            x_coords.append(version2aucs[versionNums[0]][gene])
            y_coords.append(version2aucs[versionNums[1]][gene])
    fig = plt.figure()
    plt.plot(x_coords, y_coords)
    plt.xlabel(versionNums[0] + ' AUCs')
    plt.ylabel(versionNums[1] + ' AUCs')
    plt.grid(b=True)
    plt.show()


def main():
    experimentSys = sys.argv[1]
    vers_A_num = 'version_' + sys.argv[2]
    vers_B_num = 'version_' + sys.argv[3]
    version2files = {vers_A_num: ('humannet1.entrez.txt', 
                                  'HumanNet1_adj_matrix.npy'),
                     vers_B_num: ('H6Net_CC.net', 'HumanNet2_adj_matrix.npy')}
    biogridFile = 'BIOGRID-3.4.127-human.txt'
    seedSets = func_net_pred.read_biogrid(experimentSys, biogridFile)
    version2aucs = dict()
    for version in (vers_A_num, vers_B_num):
        inputFiles = version2files[version]
        version2aucs[version] = get_seed_set_aucs(inputFiles)


if __name__=="__main__":
    main()

