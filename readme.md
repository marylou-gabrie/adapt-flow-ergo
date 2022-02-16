Repository associated with the publication:

 Adaptation of the Independent Metropolis-Hastings Sampler with Normalizing Flow Proposals
 James Brofos, Marylou Gabrie, Marcus A. Brubaker and Roy Lederman 
 Yale University, New York University, Flatiron Institute, York University, and  Vector Institute 


The back-bone of the code is contained in the ``adaflonaco`` python package. 

## Installation

From this root folder, install the package in editable mode by running
```
pip install . -e
```

Then you should be able to run the test
```
python tests/test_phifour_adapt.py
```

## For benchmark

We also provide an implementation of algorithm proposed in 
A framework for adaptive mcmc targeting multimodal distributions. Annals of Statistics, 48(5), 2930–2952. https://doi.org/10.1214/19-AOS1916
Pompe E., Holmes C., & Łatuszyński K. (2020). 

The back-bone of the code is contained in the ``jams`` python package. 

