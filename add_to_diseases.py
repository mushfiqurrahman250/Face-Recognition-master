import shutil

import os

source = 'C:/Users/ASUS/Documents/train/benign/'
source1 = 'C:/Users/ASUS/Documents/train/malignant/'

dest1 = 'D:/untitled/Face-Recognition-master/diseases/'
dest2 = 'D:/untitled/Face-Recognition-master/diseases_m/'

files = os.listdir(source)

for f in files:

    shutil.copy(source+f, dest1)

files = os.listdir(source1)

for f in files:

    shutil.copy(source1+f, dest2)