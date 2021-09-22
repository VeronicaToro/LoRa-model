#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

class Node(object):
    def __init__(self, id, numGateways):
        self._id = id
        self._xpos = 0
        self._ypos = 0
        self._SF = 0
        self._txPower = 0
        self._possibleSFsPerTP = {}
        self._currentAvailableSFs = []
        self._currentAvailableTPs = dict()
        self._distToGW = {}
        self._pathloss = {}
        self._setInterferers = []
        self._numInterferers = 0
        self._numInterferersPerGW = dict()
#        self._numInterferers = {7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
        self._probInterferers = {7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
        self._setInterferers_dict = dict()
        self._setInterferersPerGW = dict()
        self._probOutage = dict()
        self._limits = dict()
        self._interferersSIR = dict()
        self._numGateways = numGateways
        for gw in range(numGateways):
            self._numInterferersPerGW[gw] = 0
            self._setInterferersPerGW[gw] = dict()
            self._limits[gw] = dict()
            self._interferersSIR[gw] = dict()
    
    def get_TP(self):
        return self._txPower

    def set_TP(self, tp):
        self._txPower = tp

    def set_SF(self, sf):
        self._SF = sf

    def get_SF(self):
        return self._SF
    
    def get_prob_interferers_list_PerGW(self):
        return self._setInterferersPerGW
    
    def set_prob_outage(self, prob, gw):
        self._probOutage[gw] = prob
    
    def get_prob_outage(self):
        return self._probOutage
    
    def set_distance_to_gateway(self, distToGW, gatewayNum):
        self._distToGW[gatewayNum] = distToGW

    def get_distance_to_all_gateways(self):
        return self._distToGW

    def get_distance_to_gateway(self, gatewayNum):
        return self._distToGW[gatewayNum]
    
    def init_setInterferersPerGW(self):
        for gw in range(self._numGateways):
            self._setInterferersPerGW[gw] = dict()
            
    def set_pathloss(self, pathloss, gwnum):
        self._pathloss[gwnum] = pathloss

    def get_pathloss(self, gwnum):
        return self._pathloss[gwnum]
    
    def set_location(self, x, y):
        self._xpos = x
        self._ypos = y

    def get_location(self):
        return self._xpos, self._ypos
    
    def append_interferer(self, nodeIndex, sf, gw):
        self._setInterferers.append(nodeIndex)
        self._numInterferers += 1
        self._numInterferersPerGW[gw] += 1
        
    def sum_prob_interferer(self, prob, node_index, sf, gw):
        self._probInterferers[sf] += prob
        self._setInterferers_dict[node_index] = prob
        self._setInterferersPerGW[gw][node_index] = prob
        
    def set_available_sfs_pertp(self, possibleSFsPerTP, gatewayNum):
        self._possibleSFsPerTP[gatewayNum] = possibleSFsPerTP

    def get_available_sfs_pertp(self):
        return self._possibleSFsPerTP