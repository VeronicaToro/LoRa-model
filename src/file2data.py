# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 08:39:45 2020

@author: torobv1
"""
import re
import math

def distance(P1, P2):
    x1 = P1[0]
    y1 = P1[1]
    x2 = P2[0]
    y2 = P2[1]
    d = math.sqrt((x1-x2)**2+(y1-y2)**2)
    return d

def file_reader(filename, read_SF_and_TP=False):

    with open(filename) as f:
        read_data = f.read()
    
    search_string = "numberOfNodes = (\d+)"
    m = re.search(search_string, read_data)
    num_nodes = int(m.group(1))
    search_string = "numberOfGateways = (\d+)"
    m = re.search(search_string, read_data)
    numGateways = int(m.group(1))
        
    GWloc = []
    for j in range(numGateways):
        search_string = "loRaGW\[{}\].\*\*.initialX = (\d+\.\d+\d+)".format(j)
        m = re.search(search_string, read_data)
        x = m.group(1)
        x = float(x.replace("m",""))
        
        search_string = "loRaGW\[{}\].\*\*.initialY = (\d+\.\d+\d+)".format(j)
        m = re.search(search_string, read_data)
        y = m.group(1)
        y = float(y.replace("m",""))
        
        GWloc.append((x,y))
#    print("GWloc:",GWloc)
    
    nodesLoc = []
    SF = dict()
    TP = dict()
    for i in range(num_nodes):
        search_string = "loRaNodes\[{}\].\*\*.initialX = (\d+\.\d+\d+)".format(i)
        m = re.search(search_string, read_data)
        x = m.group(1)
        x = float(x.replace("m",""))
            
        search_string = "loRaNodes\[{}\].\*\*.initialY = (\d+\.\d+\d+)".format(i)
        m = re.search(search_string, read_data)
        y = m.group(1)
        y = float(y.replace("m",""))
        
        nodesLoc.append((x,y))
        
        if read_SF_and_TP:
            search_string = "loRaNodes\[{}\].\*\*initialLoRaSF = (\d+)".format(i)
            m = re.search(search_string, read_data)
            sf = int(m.group(1))
                
            search_string = "loRaNodes\[{}\].\*\*initialLoRaTP = (\d+)".format(i)
            m = re.search(search_string, read_data)
            tp = m.group(1)
            tp = int(tp.replace("dBm",""))
    
            
            SF[i] = sf
            TP[i] = tp
    
    distances = []
    for j in range(numGateways):
        distances.append([])
    for node in range(len(nodesLoc)):
        d = []
        for j in range(numGateways):
            d.append(distance(nodesLoc[node],GWloc[j]))
        idx = d.index(min(d))
        distances[idx].append((node,d[idx]))
        

    return distances, SF, TP, nodesLoc, GWloc