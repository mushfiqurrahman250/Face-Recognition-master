import math
import os
from os import walk

from face_cutter import face_cut

mypath = "D:/untitled/Face-Recognition-master/images/"
f = []
for (_, _, filenames) in walk(mypath):
    break

for i in range(0, math.ceil(len(filenames)-1)):
    name = filenames[i][0:len(filenames[i]) - 4]
    path = os.path.join('test', name)
    face_cut(mypath, name)
