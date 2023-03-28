import numpy as np
import os

path = './target-data/test'
files = os.listdir(path)
for f in files:
	if(f.endswith('.npy')):
		keypoints = np.load(os.path.join(path, f))
		if len(keypoints) == 0:
			print(f)

