import gzip
import struct
import numpy as np

def read_images(filename):
    with gzip.open(filename, 'rb') as f:  # otwieramy .gz
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def read_labels(filename):
    with gzip.open(filename, 'rb') as f:  # otwieramy .gz
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)
