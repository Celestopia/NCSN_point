import numpy as np
import os
import sys
from main import hyperparameter_dict, dict2namespace, main
import torch.nn as nn


args=dict2namespace(hyperparameter_dict)




#for i in [0.5,1.0,2.0,4.0,8.0,16.0,32.0]:
#    args.model.sigma_begin = i
#    main(args)
#
#args.model.sigma_begin = 1.00

#for i in [0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.1,0.2]:
#    args.model.sigma_end = i
#    main(args)
#
#args.model.sigma_end = 0.01

#for i in [4,6,8,10,16,24,36,48,72]:
#    args.model.num_classes = i
#    main(args)
#
#args.model.num_classes = 8

for i in [100,200,300,500,1000]:
    args.sampling.n_steps_each = i
    main(args)

args.sampling.n_steps_each = 500

for i in [64, 256,512]:
    args.model.hidden_dim = i
    main(args)

args.model.hidden_dim = 128

#for i in [nn.SELU, nn.ELU, nn.ReLU, nn.Tanh, nn.CELU]:
#    args.model.activation = i
#    main(args)

