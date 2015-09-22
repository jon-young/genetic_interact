#!/usr/bin/env python

"""
Test of combining lists, which have different degrees of overlap among one 
another.

The original test set was:
{frozenset({'a', 'b', 'c', 'd'}),
 frozenset({'a', 'b', 'd'}),
 frozenset({'x', 'y', 'z', 't'}),
 frozenset({'a', 'e', 'f', 'g'}),
 frozenset({'h', 'i', 'j', 'k'}),
 frozenset({'h', 'k', 'l', 'm', 'n'}),
 frozenset({'a', 'i', 'x'}),
 frozenset({'a', 'b', 'x'})}

Created: 21 September 2015
"""

import itertools
import pdb
import random
import string
import sys


def jaccard(s1, s2):
    if isinstance(s1, set) and isinstance(s2, set):
        return len(s1 & s2)/len(s1 | s2)
    elif isinstance(s1, frozenset) and isinstance(s2, frozenset):
        return len(s1 & s2)/len(s1 | s2)
    else:
        print('Both arguments to Jaccard must be sets. Exiting...\n')
        sys.exit()


def combine_sets(setOfSets):
    minJaccard = 0.5
    print('The minimum Jaccard index is', minJaccard, '\n')
    newSetOfSets = set()
    deleteLater = set()
    for pair in itertools.combinations(setOfSets, 2):
        if jaccard(pair[0], pair[1]) >= minJaccard:
            pairUnion = pair[0] | pair[1]
            if pairUnion == pair[0]:
                newSetOfSets.add(pairUnion)
                deleteLater.add(pair[1])
            elif pairUnion == pair[1]:
                newSetOfSets.add(pairUnion)
                deleteLater.add(pair[0])
            else:
                newSetOfSets.add(pairUnion)
                deleteLater.update((pair[0], pair[1]))
        else:
            newSetOfSets.update((pair[0], pair[1]))
    newSetOfSets = {x for x in newSetOfSets if x not in deleteLater}
    while len(newSetOfSets) != len(setOfSets):
        setOfSets = newSetOfSets
        newSetOfSets = set()
        deleteLater = set()
        for pair in itertools.combinations(setOfSets, 2):
            if jaccard(pair[0], pair[1]) >= minJaccard:
                pairUnion = pair[0] | pair[1]
                if pairUnion == pair[0]:
                    newSetOfSets.add(pairUnion)
                    deleteLater.add(pair[1])
                elif pairUnion == pair[1]:
                    newSetOfSets.add(pairUnion)
                    deleteLater.add(pair[0])
                else:
                    newSetOfSets.add(pairUnion)
                    deleteLater.update((pair[0], pair[1]))
            else:
                newSetOfSets.update((pair[0], pair[1]))
        newSetOfSets = {x for x in newSetOfSets if x not in deleteLater}
    return setOfSets
    
    
def get_sets(startLetter, endLetter, numSets, minNumPerSet, maxNumPerSet):
    setOfSets = set()
    startIdx = string.ascii_lowercase.index(startLetter)
    endIdx = string.ascii_lowercase.index(endLetter)
    setToChooseFrom = list(string.ascii_lowercase)[startIdx:endIdx+1]
    for i in range(numSets):
        setSize = random.randint(minNumPerSet, maxNumPerSet)
        setOfSets.add(frozenset(random.sample(setToChooseFrom, setSize)))
    return setOfSets


def main():
    startLetter = sys.argv[1]
    endLetter = sys.argv[2]
    numSets = int(sys.argv[3])
    minNumPerSet = int(sys.argv[4])
    maxNumPerSet = int(sys.argv[5])
    s = get_sets(startLetter, endLetter, numSets, minNumPerSet, maxNumPerSet)
    print('The sets are:')
    for x in s:
        print(', '.join(x))
    print()
    combinedSets = combine_sets(s)
    print('The best combination of sets is:')
    for x in combinedSets:
        print(', '.join(x))
    print()


if __name__=="__main__":
    main()

