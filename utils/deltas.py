import tensorflow as tf
import numpy as np


def deltas(feat, N, feature_dimension):
	if N < 1:
		raise ValueError('N must be an integer >= 1')

	denominator = 2 * sum([i**2 for i in range(1, N+1)])

	delta_feat = []
	padded = tf.concat([tf.reshape(tf.tile(feat[0,:], [N]), [N,-1]), feat, tf.reshape(tf.tile(feat[-1,:], [N]), [N,-1])], 0)
	
	extended = padded[N:-N]
	for i in range(N):
		if i == N - 1:
			extended = tf.concat([padded[N-(i+1):-N-(i+1)], extended, padded[N+(i+1):]], 1)
		else:
			extended = tf.concat([padded[N-(i+1):-N-(i+1)], extended, padded[N+(i+1):-N+(i+1)]], 1)
	extended = tf.reshape(tf.transpose(extended), [2*N+1, feature_dimension, -1])

	delta_feat = tf.transpose(tf.tensordot(tf.cast(tf.range(-N,N+1), tf.float32), extended, [[0],[0]])) / denominator

	return delta_feat
