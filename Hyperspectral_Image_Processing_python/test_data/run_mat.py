import scipy
from scipy.io import loadmat
import pandas as pd
import numpy as np

annots = loadmat('indian_pines.mat')
#print(annots)

ip = annots['indian_pines']
list_ip = ip.tolist()
print (len(list_ip))
print(ip.shape)

newData = list(zip(list_ip[0], list_ip[1], list_ip[2]))
columns = ['pines_x', 'pines_y', 'pines_z']
df = pd.DataFrame(newData, columns=columns)
df