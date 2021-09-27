#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

class Configuration:
    """Class that manages configuration of the network"""
    def __init__(self,parent):
        self.parent = parent
    
    def set_possibleSFs_possibleTPs_allnodes(self):
        """
        Set available SFs for each TP once distance to the gateway is known
        :return:
        """
        self.parent.set_sf_ranges_alldist_alltp()
        for i in range(self.parent._numNodes):
            xLocation, yLocation = self.parent._nodes[i].get_location()
            possibleSFsPerTPperGW = dict()
            for j in range(self.parent._numGateways):
                gatewayCoords = self.parent._gateways[j].get_location()
                distToGW = np.sqrt(math.pow((xLocation-gatewayCoords[0]),2)+ math.pow((yLocation-gatewayCoords[1]),2))
                possibleSFsPerTP = self.calculate_available_sfs_for_node(distToGW)
                possibleSFsPerTPperGW[j] = possibleSFsPerTP
                self.parent._nodes[i].set_available_sfs_pertp(possibleSFsPerTP, j)
    
    def calculate_available_sfs_for_node(self, distToGW):
        """
        Based on distance to GW, return possible spreading factors for each transmit power level
        availableSFPerTxPower is a dictionary with keys of transmit power levels and value = list of spreading factors that can be used at 'key' transmit power level
        For example, availableSFPerTxPower[2] = [11, 12],
        :param distToGW:
        :return:availableSFPerTxPower
        """
        availableSFPerTxPower = OrderedDict()
        for txPower in self.parent._availableTPs:
            sfRanges = self.parent._sfRangesPerTP[txPower]
            distKeysInSFRanges = list(sfRanges.keys())
            spreadingFactors = []
            if distToGW > self.parent._maxCommDistancePerTP[txPower]:
                availableSFPerTxPower[txPower] = []
            else:
                for keyindex in range(1,len(distKeysInSFRanges)+1):
                    if distToGW < distKeysInSFRanges[keyindex]:
                        spreadingFactors = sfRanges[distKeysInSFRanges[keyindex-1]]
                        break
                availableSFPerTxPower[txPower] = spreadingFactors
        return availableSFPerTxPower
    
    def configure_TP_allnodes(self, tpStrategy="maximum"):
        logger.debug("Configuring transmission powers for all nodes with strategy {}".format(tpStrategy))
        nodesPerTP = {2:0, 5:0, 8:0, 11:0, 14:0}
        for i in range(self.parent._numNodes):
            distToGWs = self.parent._nodes[i].get_distance_to_all_gateways()
            closestGWIndex = min(distToGWs, key=distToGWs.get)
            availableSFsPerTP = self.parent._nodes[i].get_available_sfs_pertp()[closestGWIndex]
            availableTPsForNode = []
            nodeSF = self.parent._nodes[i].get_SF()
            if nodeSF:
                for tp in availableSFsPerTP.keys():
                    if availableSFsPerTP[tp] != []:
                        if nodeSF in availableSFsPerTP[tp]:
                            availableTPsForNode.append(tp)
            else:
                for tp in availableSFsPerTP.keys():
                    if availableSFsPerTP[tp] != []:
                        availableTPsForNode.append(tp)
            for tp in availableSFsPerTP.keys():
                if availableSFsPerTP[tp] != []:
                    availableTPsForNode.append(tp)
            if tpStrategy == "random":
                txPower = np.random.choice(availableTPsForNode)
            elif tpStrategy == "maximum":
                txPower = 14
            elif tpStrategy == "minimum":
                txPower = min(availableTPsForNode)
            nodesPerTP[txPower] += 1
            self.parent._nodes[i].set_TP(txPower)
        logger.debug("Number of nodes per TP = TP2 = {}, TP5 = {}, TP8 = {}, TP11 = {}, TP14 = {}".format(nodesPerTP[2], nodesPerTP[5],
                                                                                                         nodesPerTP[8], nodesPerTP[11], nodesPerTP[14]))
    
    def configure_SF_allnodes(self, sfStrategy="minimum"):
        logger.debug("Configuring spreading factors for all nodes with strategy {}".format(sfStrategy))

        outOfRangeNodes = []
        nodesPerSF = {7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
        for i in range(self.parent._numNodes):
            distToGWs = self.parent._nodes[i].get_distance_to_all_gateways()
            closestGWIndex = min(distToGWs, key=distToGWs.get)
            availableSFsPerTP = self.parent._nodes[i].get_available_sfs_pertp()[closestGWIndex]
            tp = self.parent._nodes[i].get_TP()
            currentAvailableSFs = availableSFsPerTP[tp]
            if currentAvailableSFs == []:
                currentAvailableSFs = [12]
                outOfRangeNodes.append(i)
            if sfStrategy == "random":
                sf = np.random.choice(currentAvailableSFs)
            elif sfStrategy == "minimum":
                sf = min(currentAvailableSFs)
            elif sfStrategy == "maximum":
                sf = 12
            nodesPerSF[sf] += 1
            self.parent._nodes[i].set_SF(sf)
        logger.debug("Number of nodes per SF = SF7 = {}, SF8 = {}, SF9 = {}, SF10 = {}, SF11 = {}, SF12 = {}".format(nodesPerSF[7], nodesPerSF[8], nodesPerSF[9], 
                                                                                                             nodesPerSF[10], nodesPerSF[11], nodesPerSF[12]))
        logger.debug("Number of nodes out of range = {}".format(len(outOfRangeNodes)))