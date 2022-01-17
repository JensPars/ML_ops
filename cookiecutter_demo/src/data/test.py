import gzip
f = gzip.open('/Users/jensparslov/Documents/DTU/ML_ops/cookiecutter_demo/data/raw/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 1000

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)