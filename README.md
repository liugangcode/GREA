Graph Rationalization with Environment-based Augmentations
====

This is the source code for the paper:

Graph Rationalization with Environment-based Augmentations

by Anonymous Author(s)

## Requirements

This code package was developed and tested with Python 3.9.9 and PyTorch 1.10.1. All dependencies specified in the ```requirements.txt``` file. The packages can be installed by
```
pip install -r requirements.txt
```

## Usage

Following are the commands to reproduce the experiment results on HIV and oxygen permeability.

```
# HIV
python main_pyg.py --dataset ogbg-molhiv

# oxygen permeability
python main_pyg.py --dataset plym-o2_prop
```

#### Implementation on other datasets

TODO