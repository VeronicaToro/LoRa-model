# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:11:20 2020
Recursive algorithm to enumerate set covers based on https://ieeexplore.ieee.org/document/9217736
"""
import numpy as np
import collections

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

def RemoveRedundant(C):
    _C1 = C.copy()
    for s in _C1:
        covered = set()
        for node in set(C)-{s}: 
            covered = covered | C[node]
        if all(element in covered for element in _C1[s]):
            del C[s]
    return C

def algorithm(network, E, N, node, Result, R, nodes, RR2, flag):
    if (len(R) >= 700):
        flag = True
    if flag:
        return R
    Result[node] = network[node]
    nodes.remove(node)
    for k in N:
        if k in E:
            E.remove(k)
    if not nodes:   # We ran out of N nodes
        flag = True
        return R
    if not E:
        Result = RemoveRedundant(Result)
        if (not any(compare(Result, x) for x in RR2)):
            R.append(Result)
            RR2.append(Result)
            
        Result = dict()
        return R
    else:
        Ext = []
        for i in nodes:
            if network[i].intersection(E):
                Ext.extend([i])

        for node2 in Ext:
            if not flag:
                algorithm(network, E.copy(), network[node2], int(node2), Result.copy(), R, nodes.copy(), RR2, flag)
        
    Result = dict()
    return R

def calculate_setCovers(Universe, sets):
    RR = set()
    RR2 = []
    RR3 = []
    nodes = []
    for i in sets:
        nodes.extend([i])
    flag = False
    
    for node in range(len(sets)):
        Result = dict()
        R = []
        setCovers = algorithm(sets, Universe.copy(), sets[node], node, Result, R, nodes.copy(), RR2, flag)   
        RR = np.array(setCovers)
        RR3.extend(RR.tolist())
    
    idx = 0
    RR4 = []
    for i in RR3:
        RR4.append([])
        for j in i:
            RR4[idx].append(tuple(i[j]))
        idx += 1
    return RR4