import sys
import numpy
import json

filename = sys.argv[1]
npy = numpy.load(filename, allow_pickle=True)
npy0 = npy.item(0);
s = json.dumps({
    'motion': npy0['motion'].tolist(),
    'text': npy0['text'],
    'lengths': npy0['lengths'].tolist(),
    'num_samples': npy0['num_samples'],
    'num_repetitions': npy0['num_repetitions'],
})
print(s)