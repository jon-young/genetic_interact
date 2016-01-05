#!/usr/bin/env python

"""
Draws functional network clusters predictive of genetic interactions
Allows user to interactively navigate through each cluster

Created: 26 December 2015
"""

import bisect
import collections
import itertools
import math
import matplotlib.pyplot as plt
import mygene
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
        fnetpath = os.path.join('..', 'data', 'FlyNetDataFrame.pkl')
    else:
        print('ORGANISM NOT FOUND! Exiting...')
        sys.exit()

    return biogridpath, fnetpath


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
    def __init__(self, fig, df, biogrid):
        self.func = df
        self.seedAUC = biogrid.seedAUC
        self.seed2interactors = biogrid.seed2interactors
        self.fig = fig
        self.ax = fig.add_subplot(111)
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
        self.mg = mygene.MyGeneInfo()
        self.draw_net()

    def draw_net(self):
        plt.cla()
        seedGene = self.seedAUC[self.idxs[self.pos]][1]
        interactors = self.seed2interactors[seedGene]
        subnet = self.func[interactors].copy()
        edges = list(subnet[subnet > self.llsCut].stack().index)
        G = nx.Graph()
        G.add_edges_from(edges)
        nodedist = 1.5/math.sqrt(nx.number_of_nodes(G))
        pos_ = nx.spring_layout(G, k=nodedist, scale=12.0)
        colored = set(itertools.chain.from_iterable(edges)) & set(interactors)
        nx.draw_networkx_nodes(G, pos=pos_, nodelist=list(colored), node_size=0, 
                node_color='c')
        otherNodes = list(set(G.nodes()) - colored)
        nx.draw_networkx_nodes(G, pos=pos_, nodelist=otherNodes, node_size=30, 
                node_color='w', alpha=0.25)
        nx.draw_networkx_edges(G, pos=pos_, alpha=0.25)
        try:
            labels_ = {g:self.mg.getgene(g)['name'] for g in colored}
        except:
            labels_ = {g:g for g in colored}
        nx.draw_networkx_labels(G, pos=pos_, labels=labels_, font_size=8, 
                font_color='b')
        AUC = self.seedAUC[self.idxs[self.pos]][0]
        try:
            seedName = self.mg.getgene(seedGene)['name']
        except:
            seedName = seedGene
        self.ax.set_title('Seed gene: {}; AUC = {:.3f}'.format(seedName, AUC))
        plt.axis('off')
        self.fig.canvas.draw()

    def onpress(self, event):
        end = len(self.idxs) - 1
        if event.key == 'up':
            self.pos = np.clip(self.pos + 1, 0, end)
            self.draw_net()
        elif event.key == 'down':
            self.pos = np.clip(self.pos - 1, 0, end)
            self.draw_net()
        else:
            pass


print('\nChoose from the following organisms (enter species name):')
print('1) cerevisiae')
print('2) sapiens')
print('3) melanogaster')
organism = input()
biogridpath, fnetpath = setup_filepaths()

print('\nReading in functional gene network...')
funcNetDf = pd.read_pickle(fnetpath)
biogrid = Biogrid(funcNetDf)
biogrid.seed_set_predictability()

fig = plt.figure(figsize=(12,8))
interactive = NavPlot(fig, funcNetDf, biogrid)
fig.canvas.mpl_connect('key_press_event', interactive.onpress)
plt.tight_layout()
plt.show()

