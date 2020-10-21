import shutil

import os

source = 'C:/Users/ASUS/Documents/test/benign/'

dest1 = 'D:/untitled/Face-Recognition-master/diseases/'

files = os.listdir(source)

for f in files:

    shutil.copy(source+f, dest1)