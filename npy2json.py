import sys
import numpy

filename = sys.argv[1]
npy = numpy.load(filename, allow_pickle=True)
npy0 = npy.item(0);