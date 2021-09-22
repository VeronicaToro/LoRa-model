#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
import logging
import copy
from node import Node
from gateway import Gateway

logger = logging.getLogger(__name__)

class Deployment:
    """Class that creates the network deployment"""
    def __init__(self,parent):
        self.parent = parent
        self._distancesMatrix = []
        self._numNodesPerGW = []
        self._numOverlappingNodes = 0

    def add_nodes(self, numberOfNodes, passNodeCoords = False, nodeCoordinates=None, numOldNodes=0):
        k = 0
        for i in range(numOldNodes, numOldNodes+numberOfNodes):
            self.parent._nodes.append(Node(i, self.parent._numGateways))
            if passNodeCoords == True:
                self.set_location_for_node(i, nodeCoordinates[k][0],nodeCoordinates[k][1])
            k += 1
            
    def add_gateways(self, numGateways, gateway_coords=[], radius=0):
        self._gateway_coords = gateway_coords
        for index in range(numGateways):
            xLocation = gateway_coords[index][0]
            yLocation = gateway_coords[index][1]
            logger.debug("Deploying gateway {gw} at {x},{y}".format(gw=index, x=xLocation, y=yLocation))
            if self.isOutside(xLocation,yLocation):
                logger.error("Gateway {} coordinates ({},{}) outside network area (0-{},0-{})".
                             format(index, xLocation, yLocation, self.parent._networkSizeX, self.parent._networkSizeY))
                exit(1)
            self.parent._gateways.append(Gateway(index, xLocation, yLocation, radius))

    def isOutside(self, x, y):
        return x < 0 or x > self.parent._networkSizeX or \
               y < 0 or y > self.parent._networkSizeY
               
    def init_dist_vector_per_node(self, numGateways, gatewayCoords, totalNodes):
        for i in range(totalNodes):
            for gw_index in range(numGateways):
                xLocation,yLocation = self.parent._nodes[i].get_location()
                currentGWCoords = gatewayCoords[gw_index]
                distToGW = np.sqrt(math.pow((xLocation-currentGWCoords[0]),2) + math.pow((yLocation-currentGWCoords[1]),2))
                self.parent._nodes[i].set_distance_to_gateway(distToGW,gw_index)
                
    def set_locations_allnodes(self, numGateways=0, numNodesPerGW=[], gatewayCoords=[], deploymentArea="square", locationType="random",
                               circleRadius=None, nodesInRings = [], radii = []):
        if nodesInRings == []:
            count = 0
            for gw_index in range(numGateways):
                numNodes = numNodesPerGW[gw_index]
                radius = circleRadius[gw_index]
                currentGWCoords = gatewayCoords[gw_index]
                logger.debug("Deploying {} nodes in range {} from gateway {} at {},{}".
                             format(numNodes, radius,gw_index,currentGWCoords[0],currentGWCoords[1]))
                for i in range(numNodes):
                    node_index = i + count
                    isOutsideNetwork = True
                    while isOutsideNetwork:
                        xLocation, yLocation = self.get_coordinates_for_node(deploymentArea, locationType, radius, currentGWCoords)
                        isOutsideNetwork = self.isOutside(xLocation, yLocation)
                    distToGW = np.sqrt(math.pow((xLocation-currentGWCoords[0]),2)+ math.pow((yLocation-currentGWCoords[1]),2))
                    assert distToGW > 0
                    self.set_location_for_node(node_index, xLocation, yLocation)
                count = count + numNodes
        else:
            if len(radii) != len(nodesInRings)+1:
                logger.error("Error in function \"set_locations_allnodes\"")
                logger.error("Length of radii vector must be equal to (length of \"number of nodes per rings\") + 1")
                exit(0)
            nodeNumber = 0
            for ring in range(len(nodesInRings)):
                for i in range(nodesInRings[ring]):
                    gatewayCoords = self.parent._gateways[0].get_location()
                    randomRadius=(np.random.uniform(0,1)*(radii[ring+1]**2-radii[ring]**2)) + radii[ring]**2
                    #randomRadius = np.random.uniform(radii[ring]*radii[ring],radii[ring+1]*radii[ring+1])
                    randomAngle = np.random.uniform(0,2*np.pi)
                    xLocation = np.sqrt(randomRadius) * np.cos(randomAngle) + gatewayCoords[0]
                    yLocation = np.sqrt(randomRadius) * np.sin(randomAngle) + gatewayCoords[1]
                    distToGW = np.sqrt(math.pow((xLocation-gatewayCoords[0]),2) + math.pow((yLocation-gatewayCoords[1]),2))
                    self.set_location_for_node(nodeNumber, xLocation, yLocation)
                    nodeNumber += 1
                    
    def set_location_for_node(self, nodeNum, xLocation, yLocation):
        self.parent._nodes[nodeNum].set_location(xLocation, yLocation)

    def get_coordinates_for_node(self, deploymentType="square", locationInitialization="random",
                                 radius=0.0, gatewayCoords=(0,0)):
        if deploymentType == "square":
            if locationInitialization=="random":
                if deploymentType=="square":
                    xLocation = np.random.uniform(self.parent._networkSizeX)
                    yLocation = np.random.uniform(self.parent._networkSizeY)
                    return xLocation, yLocation
        elif deploymentType == "circle":
            if radius == 0:
                logger.error("Error generating coordinates for node in circular deployment as radius cannot be zero")
                exit(0)
            else:
                return self.get_location_uniform_circle(gatewayCoords, radius)
            
    def get_location_uniform_circle(self, gatewayCoords, radius):
        randomRadius = np.random.uniform(0, radius * radius)
        randomAngle = np.random.uniform(0, 2 * np.pi)
        xLocation = np.sqrt(randomRadius) * np.cos(randomAngle)
        yLocation = np.sqrt(randomRadius) * np.sin(randomAngle)
        return xLocation + gatewayCoords[0], yLocation + gatewayCoords[1]
    
    
    def calculate_network_distance_parameters(self):
        distances = []
        num_gw_in_range = [0 for x in range(self.parent._numNodes)]
        for j in range(self.parent._numGateways):
            distances.append([])
        for i in range(self.parent._numNodes):
            closestGW = 0
            shortestDistance = 1000
            for j in range(self.parent._numGateways):
                distance_to_gw_j = self.parent._nodes[i].get_distance_to_gateway(j)
                if distance_to_gw_j < shortestDistance:
                    shortestDistance = distance_to_gw_j
                    closestGW = j
            if shortestDistance < self.parent._gateways[closestGW].get_radius():
                distances[closestGW].append(tuple((i, shortestDistance)))
                num_gw_in_range[i] += 1
        for j in range(self.parent._numGateways):
            distances[j] = sorted(distances[j], key= lambda x: x[1])
            self._numNodesPerGW.append(len(distances[j]))
        self._distancesMatrix = copy.deepcopy(distances)
        self._numOverlappingNodes = sum(self._numNodesPerGW) - self.parent._numNodes