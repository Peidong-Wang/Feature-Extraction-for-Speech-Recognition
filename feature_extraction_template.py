import tensorflow as tf
import wavefile
import numpy as np

import sys
sys.path.append("./utils")

import fft2melmx
import deltas
import extract_window


# BLOCK: input and output files/folders


# feature extraction arguments
epsilon = 1e-40

num_bins = 80

samp_freq = 16000
low_freq = 20
high_freq = None

window_opts_dict = {}

window_opts_dict['frame_shift_ms'] = 10.0
window_opts_dict['frame_length_ms'] = 25.0
window_opts_dict['dither'] = 1.0
window_opts_dict['preemph_coeff'] = 0.97
window_opts_dict['remove_dc_offset'] = True
window_opts_dict['window_type'] = "povey"
window_opts_dict['round_to_power_of_two'] = True
window_opts_dict['blackman_coeff'] = 0.42
window_opts_dict['snip_edges'] = True

window_opts_dict['window_shift'] = int(samp_freq * 0.001 * window_opts_dict['frame_shift_ms'])
window_opts_dict['window_size'] = int(samp_freq * 0.001 * window_opts_dict['frame_length_ms'])
window_opts_dict['padded_window_size'] = extract_window.shift_bit_length(window_opts_dict['window_size'])

# generate mel weights
mel_wts = np.array(fft2melmx.fft2melmx(window_opts_dict['padded_window_size'], samp_freq, num_bins, 1.0, low_freq, high_freq, True, 'const')[0], np.float32)
mel_wts = mel_wts.T


# define TensorFlow graph
x = tf.placeholder(tf.float32, shape=[None, 80])

log_mel = tf.log(x + epsilon) # log(mel) features

feature_delta = deltas.deltas(log_mel, 2, 80)
feature_delta_delta = deltas.deltas(feature_delta, 2, 80)

feature_deltas = tf.concat([log_mel, feature_delta, feature_delta_delta], axis=1)

feature_out = feature_deltas - tf.reduce_mean(feature_deltas, axis=0, keep_dims=True) # a mean normalization on the deltas features is typically used to normalize the magnitudes of input features


# extract features
with tf.Session() as sess:
	
	# BLOCK: get wav_file, the name of the wave file to be processed
	
	with wavefile.WaveReader(wav_file) as wav_reader:

		# sanity checks
		channels = wav_reader.channels
		assert channels == 1
		assert wav_reader.samplerate == 16000

		samples = wav_reader.frames
		wav_data = np.empty((channels, samples), dtype=np.float32, order='F')
		wav_reader.read(wav_data)
		wav_data = np.squeeze(wav_data)		

	wav_data = wav_data * np.iinfo(np.int16).max

	windowed_utterance = extract_window.extract_window(0, wav_data, window_opts_dict)

	ffted_utterance = np.fft.fft(windowed_utterance) 

	power_spectrum = np.square(np.absolute(ffted_utterance).astype(np.float32)) # note here we use the power spectrum
	power_spectrum_nodup = power_spectrum[:, 0:window_opts_dict['padded_window_size']//2+1] # remove redundant dimensions

	mel_spectrum = np.matmul(power_spectrum_nodup, mel_wts)

	feature_to_store = sess.run(feature_out, feed_dict={x: mel_spectrum}).astype(np.float32) # the output is the mean normalized deltas features

	# BLOCK: save feature_to_store
