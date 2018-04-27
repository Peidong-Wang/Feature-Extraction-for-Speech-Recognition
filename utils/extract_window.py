import numpy as np


def first_sample_of_frame(frame, opts_dict):
	frame_shift = opts_dict['window_shift']
	if opts_dict['snip_edges']:
		return frame * frame_shift
	else:
		midpoint_of_frame = frame_shift * frame + frame_shift // 2
		beginning_of_frame = midpoint_of_frame - opts_dict['window_size'] // 2
		return beginning_of_frame


def num_frames(num_samples, opts_dict, flush=True):
	frame_shift = opts_dict['window_shift']
	frame_length = opts_dict['window_size']
	if opts_dict['snip_edges']:
		if num_samples < frame_length:
			return 0
		else:
			return (1 + ((num_samples - frame_length) // frame_shift))
	else:
		num_frames = (num_samples + (frame_shift // 2)) // frame_shift
		if flush:
			return num_frames
		end_sample_of_last_frame = first_sample_of_frame(num_frames - 1, opts_dict) + frame_length
		while num_frames > 0 and end_sample_of_last_frame > num_samples:
			num_frames -= 1
			end_sample_of_last_frame -= frame_shift
		return num_frames


def dither(waveform, dither_value):
	if dither_value == 0.0:
		return waveform
	dim = waveform.shape[0]
	for i in range(dim):
		waveform[i] += np.random.normal() * dither_value
	return waveform


def preemphasize(waveform, preemph_coeff):
	if preemph_coeff == 0.0:
		return waveform
	assert preemph_coeff >= 0.0 and preemph_coeff <= 1.0
	dim = waveform.shape[0]
	for i in range(dim-1, 0, -1):
		waveform[i] -= preemph_coeff * waveform[i-1]
	waveform[0] -= preemph_coeff * waveform[0]
	return waveform


def window_functions(opts_dict):
	frame_length = opts_dict['window_size']
	assert frame_length > 0
	window = np.empty(frame_length)
	a = 2 * np.pi / (frame_length - 1)
	for i in range(frame_length):
		if opts_dict['window_type'] == "hanning":
			window[i] = 0.5 - 0.5*np.cos(a * i)
		elif opts_dict['window_type'] == "hamming":
			window[i] = 0.54 - 0.46*np.cos(a * i)
		elif opts_dict['window_type'] == "povey":
			window[i] = np.power(0.5 - 0.5*np.cos(a * i), 0.85)
		elif opts_dict['window_type'] == "rectangular":
			window[i] = 1.0
		elif opts_dict['window_type'] == "blackman":
			window[i] = opts_dict['blackman_coeff'] - 0.5*np.cos(a * i) + (0.5 - opts_dict['blackman_coeff']) * np.cos(2 * a * i)
		else:
			print("Invalid window type: " + opts_dict['window_type'])
			exit()
	return window


# the same as the one in Kaldi, process a single frame
def process_window(opts_dict, window_function, input_frame):
	frame_length = opts_dict['window_size']
	assert input_frame.shape[0] == frame_length
	processing_frame = input_frame

	if opts_dict['dither'] != 0.0:
		processing_frame = dither(processing_frame, opts_dict['dither'])

	if opts_dict['remove_dc_offset']:
		processing_frame = processing_frame - np.mean(processing_frame)

	# ommitted log_energy_pre_window for now

	if opts_dict['preemph_coeff'] != 0.0:
		processing_frame = preemphasize(processing_frame, opts_dict['preemph_coeff'])

	processing_frame = np.multiply(processing_frame, window_function)

	return processing_frame


# extract_window() is not exactly the same as the one in Kaldi, it processes all the frames at once
def extract_window(sample_offset, waveform, opts_dict): 
	wave_dim = waveform.shape[0]
	assert sample_offset >= 0 and wave_dim != 0
	frame_length = opts_dict['window_size']
	frame_length_padded = opts_dict['padded_window_size']
	num_samples = sample_offset + wave_dim

	frame_num = num_frames(num_samples, opts_dict)

	window_function = window_functions(opts_dict)

	windowed_output = np.zeros([frame_num, frame_length_padded])
	for frame in range(frame_num):
		start_sample = first_sample_of_frame(frame, opts_dict)
		end_sample = start_sample + frame_length

		if opts_dict['snip_edges']:
			assert start_sample >= sample_offset and end_sample <= num_samples
		else:
			assert sample_offset == 0 or start_sample >= sample_offset

		wave_start = start_sample - sample_offset
		wave_end = wave_start + frame_length

		single_frame = np.zeros(frame_length)

		if wave_start >= 0 and wave_end <= wave_dim:
			single_frame[0:frame_length] = waveform[wave_start:wave_end]
		else:
			for s in range(frame_length):
				s_in_wave = s + wave_start
				while s_in_wave < 0 or s_in_wave >= wave_dim:
					if s_in_wave < 0:
						s_in_wave = -s_in_wave - 1
					else:
						s_in_wave = 2 * wave_dim - 1 - s_in_wave
				single_frame[s] = waveform[s_in_wave]

		processed_frame = process_window(opts_dict, window_function, single_frame)

		windowed_output[frame, 0:frame_length] = processed_frame

	return windowed_output	


def shift_bit_length(x):
	return 1<<(x-1).bit_length()
