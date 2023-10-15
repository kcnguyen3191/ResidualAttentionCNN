import torch,random
import numpy as np
import random


# RANDOM_SEED = 42
RANDOM_SEED = np.random.randint(10000)
# RANDOM_SEED = random_seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
    
    
# class RandomSeed:
#     '''This is singleton class.'''
#     random_seed = 0
#     __instance = None

#     @staticmethod 
#     def getSeed(self):
#         """ Static access method. """
# #         if RandomSeed.__instance == None:
# #             RandomSeed()
#         return self.random_seed


#     def setSeed(self, random_seed=42):
#         self.random_seed = random_seed

#     def __init__(self):
#         """ Virtually private constructor. """
#         if RandomSeed.__instance != None:
#              raise Exception("This class is a singleton!")
#         else:
#              RandomSeed.__instance = self

            
# s = RandomSeed()
    
