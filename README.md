This repository is the implementation of the model presented in ["Modeling Communication Reliability in LoRa Networks with Device-level Accuracy"](https://ieeexplore.ieee.org/document/9488783).

Setting the environment
=======================

Conda
-----

You can easily set the environment using [Conda](https://docs.conda.io/en/latest/). Simply run
```
conda config --append channels conda-forge
conda create --name lora-env --file requirements.txt
```
to create a Conda environment called *lora-env*.

pip
---

If you prefer to set the environment using [pip](https://pypi.org/project/pip/), you can do so as follows. This was last tested with Python version 3.7.4 and 3.8.11.

First, create the pip virtual environment (here, the environment is called *.lora-env* and will be created in a folder with the same name):
```
python -m venv .lora-env
```
Activate it
```
source .lora-env/bin/activate
```
And install the dependencies
```
pip install -r requirements.txt
```
To deactivate the environment, simply run:
```
deactivate
```

Usage
=====
Clone the repository:
```
git clone https://github.com/VeronicaToro/LoRa-model.git
```
From the *src/* folder you can run the model. That is, first run
```
cd directory_where_LoRa-model_was_cloned/LoRa-model/src/
```
Then, run the model:
```
python main.py [options]
```
You can also make *main.py* executable and avoid calling Python every time. For a complete list of options run

```
python main.py --help
```
There are three options that must always be specified:
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

In uniform scenarios, nodes are uniformly distributed around gateways. Users can specify the number of nodes, the number of gateways, locations of gateway(s) and distance (*radius*) within which nodes are deployed. Specify the parameters in the file *network_config_files/network_config.ini*. For instance, for a network with 2 gateways located at coordinates (500 m, 500 m) and (1000 m, 500 m), 50 nodes deployed around each gateway within a radius of 500 m from both gateways, a sample is provided below. Note that the parameters *sizeX* and *sizeY* define the whole deployment area, such that all nodes are contained in a rectangle of sizes *sizeX*  and *sizeY*. Moreover, under [network], the total *numNodes* must be equal to the sum of *numNodes* deployed around each gateway.

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

To run the algorithm with the network specifications in the file called *network_config.ini*, inside the folder *network_config_files/*:
```
python main.py -s minimum -t maximum -d ../network_config_files/network_config.ini
```


User-specified networks
-----------------------

For a custom network, you must specify the location of all the nodes in the same format used in the example file in the *network_ini_files/* folder. Moreover, if you would like to specify the SF and TP of all nodes, you can do so in the same format as in the example file provided in this repository. Additionally, you must enable the *-sno* flag when running the code. This flag indicates whether the SFs and TPs are in the network pickle file.

```
python main.py -s minimum -t maximum -d ../network_config_files/custom_network.ini -sno 1
```

Even though the *-s* and *-t* flags have no effect in this case, they must always be entered.


Saving your results
===================

After running the algorithm, you will only get the *Overall delivery ratio per node* printed in the terminal. However, if you would like to store the delivery ratio per node in a [numpy](https://numpy.org/) file, then you must specify the path and file name you want to create with the results, using the option *-df*. Remember to include the file extension *.npy*. If such a file already exists, the results will be overwritten.
```
python main.py -s minimum -t maximum -d ../network_config_files/network_config.ini -df results.npy 
```
