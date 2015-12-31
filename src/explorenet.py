#!/usr/bin/env python

"""
Draws functional network clusters predictive of genetic interactions
Allows user to interactively navigate through each cluster

Created: 26 December 2015
"""

import bisect
import collections
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os.path
import pandas as pd
import sys
from sklearn import metrics


def setup_filepaths():
    """Setup full file paths for functional net and BIOGRID"""
    if organism == 'cerevisiae':
        biogridpath = os.path.join('..', 'data', 
                'BIOGRID-3.4.130-yeast-post2006.txt')
        fnetpath = os.path.join('..', '..', 'DataDownload', 'FunctionalNet', 
                'yeastnet2.gene.txt')
    elif organism == 'sapiens':
        biogridpath = os.path.join('..', 'data', 'BIOGRID-3.4.130-human.txt')
        fnetpath = os.path.join('..', '..', 'DataDownload', 'FunctionalNet', 
                'H6Net_CC.net')
    elif organism == 'melanogaster':
        biogridpath = os.path.join('..', 'data', 'BIOGRID-3.4.127-fly.txt')
        fnetpath = os.path.join('..', 'data', 'FlyNetEntrez-noNull.txt')
    else:
        print('ORGANISM NOT FOUND! Exiting...')
        sys.exit()

    return biogridpath, fnetpath


def get_fnet():
    """Construct adjacency matrix of functional network"""
    G = nx.read_weighted_edgelist(fnetpath, delimiter='\t')
    
    return nx.to_pandas_dataframe(G)


class Biogrid:
    def __init__(self, df):
        self.func = df
        self.seedAUC = list()
        self.seed2interactors = dict()

    def read_biogrid(self):
        experimentSys = input('\nEnter the experimental system:\n')
        seedSets = collections.defaultdict(set)
        biogridfile = open(biogridpath)
        header = biogridfile.readline().split('\t')
        expSysColNum = header.index('Experimental System')
        for line in biogridfile:
            tokens = line.split('\t')
            if tokens[expSysColNum] == experimentSys:
                seedSets[tokens[1]].add(tokens[2])
                seedSets[tokens[2]].add(tokens[1])
        return seedSets

    def seed_set_predictability(self):
        seedSets = self.read_biogrid()
        for seedGene in seedSets.keys():
            interactors = [gene for gene in seedSets[seedGene] 
                    if gene in self.func.index]
            if len(interactors) > 1:
                llsSum = self.func.loc[interactors,:].sum(axis=0)
                trueLabels = pd.Series([0]*llsSum.size, index=llsSum.index)
                trueLabels.loc[interactors] = 1
                auc = metrics.roc_auc_score(trueLabels, llsSum)
                self.seedAUC.append((auc, seedGene))
                self.seed2interactors[seedGene] = interactors
        self.seedAUC.sort()


class NavPlot:
    def __init__(self, df, biogrid):
        self.func = df
        self.seedAUC = biogrid.seedAUC
        self.seed2interactors = biogrid.seed2interactors
        print('\nTo choose which predictive functional net clusters to draw,')
        lowLim = float(input('enter the AUC lower limit: '))
        upLim = float(input('Enter the AUC upper limit: '))
        AUCs = [self.seedAUC[i][0] for i in range(len(self.seedAUC))]
        start = bisect.bisect_left(AUCs, lowLim) 
        end = bisect.bisect(AUCs, upLim)
        minIntact = int(input('Enter the minimum # of gene interactors: '))
        self.idxs = [i for i in range(start, end+1) 
                if len(self.seed2interactors[self.seedAUC[i][1]]) >= minIntact]
        if len(self.idxs) < 1:
            print('NO SEEDS WITH MINIMUM # OF GENETIC INTERACTORS! EXITING...')
            sys.exit()
        self.pos = 0
        self.llsCut = float(input('Enter a cutoff for the LLS: '))
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        self.draw_net()

    def draw_net(self):
        seedGene = self.seedAUC[self.idxs[self.pos]][1]
        interactors = self.seed2interactors[seedGene]
        subnet = self.func[interactors].copy()
        edges = list(subnet[subnet > self.llsCut].stack().index)
        G = nx.Graph()
        G.add_edges_from(edges)
        pos_ = nx.spring_layout(G)
        plt.cla()
        colored = set(itertools.chain.from_iterable(edges)) & set(interactors)
        nx.draw_networkx(G, pos=pos_, nodelist=list(colored), node_color='c')
        otherNodes = list(set(G.nodes()) - colored)
        nx.draw_networkx(G, pos=pos_, nodelist=otherNodes, node_color='w')
        plt.title('AUC = %f' %self.seedAUC[self.idxs[self.pos]][0])
        plt.axis('off')
        self.fig.canvas.draw()
        plt.show()

    def onpress(self, event):
        end = len(self.idxs) - 1
        if event.key == 'up':
            self.pos = np.clip(self.pos + 1, 0, end)
        elif event.key == 'down':
            self.pos = np.clip(self.pos - 1, 0, end)
        else:
            pass
        self.draw_net()


print('\nChoose from the following organisms (enter species name):')
print('1) cerevisiae')
print('2) sapiens')
print('3) melanogaster')
organism = input()
biogridpath, fnetpath = setup_filepaths()
print('\nReading in functional gene network...')
funcNetDf = get_fnet()
biogrid = Biogrid(funcNetDf)
biogrid.seed_set_predictability()
NavPlot(funcNetDf, biogrid)

