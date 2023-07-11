import os
import numpy as np
import scipy
import torch

if __name__ == '__main__':

    #%%
    print(os.getcwd())
    os.system("hostname")

    #%%
    print(np.random.randn(4, 2))
    print(torch.cuda.is_available())