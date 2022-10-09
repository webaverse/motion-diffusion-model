import sys
import numpy
import json

filename = sys.argv[1]
npy = numpy.load(filename, allow_pickle=True)
npy0 = npy.item(0);
s = json.dumps({
    'motion': npy0['motion'].tolist(),
})
print(s)