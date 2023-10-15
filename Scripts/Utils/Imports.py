# basic python
import sys, os, glob, time, pdb, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from PIL import Image
from tqdm import tqdm
import scipy.io
import importlib
from importlib import reload

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

# utils
from Scripts.Utils.GetLowestGPU import *
from Scripts.Utils.TimeRemaining import *

# publication quality plots
from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png', 'pdf') # plt.savefig('name.pdf', format='pdf')
