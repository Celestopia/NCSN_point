import numpy as np
import torch.nn as nn
import os
import sys
from main import dict2namespace, main
from utils.config import hyperparameter_dict_default



args=dict2namespace(hyperparameter_dict_default)


for i in [100,200,300,500,1000]:
    args.sampling.n_steps_each = i
    main(args)


