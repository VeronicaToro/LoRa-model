This repository is the implementation of the model presented in ["Modeling Communication Reliability in LoRa Networks with Device-level Accuracy"](https://ieeexplore.ieee.org/document/9488783).

Setting the environment
=======================

Conda
-----

You can easily set the environment using [Conda](https://docs.conda.io/en/latest/). Simply run
```
conda config --append channels conda-forge
conda env create --file environment.yml
```

pip
---

If you prefer to set the environment using pip, you can do so as follows. This was last tested with Python version 3.8.11.
```
pip install -r requirements.txt
```
Please note that we tested this with Python version 3.8.11

Usage
=====
Run
```
python main.py [options]
```
You can also make *main.py* executable and avoid calling Python every time. For a complete list of options run

```
python main.py --help
```
There are two options that must always be specified:
* *-s*: Spreading Factor (SF) assignment strategy. It can be either *random* or *minimum*
* *-t*: Transmission Power (TP) assignment strategy. It can be either *random*, *minimum* or *maximum*
* *-d*: Directory where the config file is located. See below for more details on the config file
```
python main.py -s random -t minimum -d path_to_config_file
```

Config file
===========

You can run the model in both, uniform scenarios or user-specified networks, as follows:

Uniform scenarios
-----------------

In this setting, the nodes are uniformly distributed around the gateways. You need to specify the number of gateways and nodes, the radius of deployment for each gateway and their location. This parameter are set in the file *network_config.ini*. For instance, for a network with 2 gateways located at coordinates (500 m, 500 m), (1000 m, 500 m), 50 nodes deployed around each gateway in a radius of 500 m for both gateways, the file should look as follows:

```
[network]
sizeX = 1500
sizeY = 1000
numNodes = 100
numGateways = 2

[gateway_0]
numNodes = 50
xLocation = 500
yLocation = 500
radius = 500

[gateway_1]
numNodes = 50
xLocation = 1000
yLocation = 500
radius = 500
```

Note that the parameters *sizeX* and *sizeY* define the whole deployed area, such that all nodes are contained in a rectangle of *sizeX* x *sizeY*. Moreover, under [network], *numNodes* must be equal to the sum of *numNodes* for each gateway.

To run the algorithm with the network specifications in the file called *network_config_ini*:
```
python main.py -s minimum -t maximum -d network_config.ini
```


User-specified networks
-----------------------

For a custom network, you must specify the location of all the nodes in the same format used in the example file in *network_ini_files*. Moreover, if you would like to specify the SF and TP of all nodes, you can do so in the same format of the example file. Additionally, you must enable the *-sno* flag when running the algorithm, as follows:

```
python main.py -s minimum -t maximum -d custom_network.ini -sno 1
```

Even though the *-s* and *-t* flags have no effect in this case, they must always be entered.


Saving your results
===================

After running the algorithm, you will only get the *Overall delivery ratio per node* printed in the terminal. However, if you would like to store the delivery ratio per node in a numpy file, then you must specify the path and file name you want to create with the results, using the option *-df*. Remember to include the file extension *.npy*. If such a file already exists, the results will be overwritten.
```
python main.py -s minimum -t maximum -d network_config.ini -df results.npy 
```
