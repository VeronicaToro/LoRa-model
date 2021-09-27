#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This function calls classes to deploy and evaluate the network"""

import argparse
from network import Network
import logging.config
import configparser

logging.config.fileConfig('logging_config.ini')

parser = argparse.ArgumentParser(description='Evaluate delivery ratio per node in LoRa networks')
parser.add_argument('--sfStrategy', '-s', dest='sfStrategy', help='SF Strategy: random, minimum',
                    required=True)
parser.add_argument('--tpStrategy', '-t', dest='tpStrategy', help='TP Strategy: random, minimum, maximum',
                    required=True)
parser.add_argument('--seedValue', '-v', dest='seedValue', help='Seed value for the script',
                    required=False, type=int, default=42)
parser.add_argument('--directory', '-d', dest='directory', help='Directory where the config file is located', 
                    required=True, type=str)
parser.add_argument('--sfNotAssigned', '-sno', dest='sfNotAssigned', help='flag for whether the SFs and TPs are in the network pickle file (1=Yes, 0=No)',
                    required=False, type=int, default=0)
parser.add_argument('--destinationFile', '-df', dest='destinationFile', help='File name where the delivery ratio per node is going to be saved', 
                    required=False, type=str, default=None)
args = parser.parse_args()
args = parser.parse_args()

nwConfigParser = configparser.ConfigParser()
nwConfigParser.read(args.directory)
gatewayCoords = []
gatewayDeployRadius = []
numNodesPerGW = []
numNodes = 0
numGateways = 0
networkSizeX = 0
networkSizeY = 0
deployFromPickleFile = False
try:
    networkSizeX = float(nwConfigParser["network"]["sizeX"])
    networkSizeY = float(nwConfigParser["network"]["sizeY"])
    numNodes = int(nwConfigParser["network"]["numNodes"])
    numGateways = int(nwConfigParser["network"]["numGateways"])

    if nwConfigParser.has_option("network", "pickleFile"):
        # call a different deploy function
        deployFromPickleFile = True
        nwPickleFile = nwConfigParser["network"]["pickleFile"]
        logging.debug("Pickle file = {}".format(nwPickleFile))
    else:
        for gw_index in range(numGateways):
            section_name = "gateway_" + str(gw_index)
            xLocation = int(nwConfigParser[section_name]["xLocation"])
            yLocation = int(nwConfigParser[section_name]["yLocation"])
            gatewayCoords.append(tuple((xLocation, yLocation)))
            gatewayDeployRadius.append(float(nwConfigParser[section_name]["radius"]))
            numNodesPerGW.append(int(nwConfigParser[section_name]["numNodes"]))
        if numNodes != sum(numNodesPerGW):
            logging.error("Sum of nodes deployed around each gateway does not match total number of nodes")
            exit(1)
except configparser.ParsingError as err:
    logging.error("Error reading configuration parser = {}".format(err))
    
arrival_rate = 1 / 1000.0 # average sending rate of all nodes
network = Network(numNodes, numGateways, networkSizeX, networkSizeY, arrival_rate, args.seedValue)

if deployFromPickleFile == False:
    network.deploy_network(numNodes=numNodes, numGateways=numGateways, gatewayCoords=gatewayCoords,
                           deploymentArea="circle", nodeLocationType="random", circleRadius=gatewayDeployRadius,
                           numNodesPerGW=numNodesPerGW)
elif deployFromPickleFile == True:
    if args.sfNotAssigned == 1:
        # init network, and assign SFs and TPs from pickle
        network.deploy_network_from_pickle(numNodes=numNodes, numGateways=numGateways, pickleFileName=nwPickleFile)
    else:
        # Only initialize node and GW positions
        network.init_network_from_pickle(numNodes=numNodes, numGateways=numGateways, pickleFileName=nwPickleFile)

network.deployment.calculate_network_distance_parameters()

if not args.sfNotAssigned:
    network.configuration.set_possibleSFs_possibleTPs_allnodes()
    network.configuration.configure_TP_allnodes()
    network.configuration.configure_SF_allnodes()

network.calculate_prob_outage()
network.update_interferers_per_node()
network.calculate_delivery_ratio_per_node(model="quasi-orthogonal-sigma", outputFile=args.destinationFile)