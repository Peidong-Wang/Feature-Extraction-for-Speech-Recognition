import numpy as np


def deltas(feat, N):
	if N < 1:
		raise ValueError('N must be an integer >= 1')

	denominator = 2 * sum([i**2 for i in range(1, N+1)])

	NUMFRAMES = len(feat)

	delta_feat = np.empty_like(feat)

	padded = np.concatenate((np.tile(feat[0,:],(N,1)), feat, np.tile(feat[-1,:],(N,1))), axis=0)
	for t in range(NUMFRAMES):
		delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1])
	delta_feat /= denominator

	return delta_feat
