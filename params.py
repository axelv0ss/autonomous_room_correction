"""
Specify necessary parameters for the project.
"""
import pyaudio
import numpy as np

BUFFER = 2048
PA_FORMAT = pyaudio.paFloat32
NP_FORMAT = np.float32
RATE = 44100

# Need to be multiples of BUFFER
# SNIPPET_LENGTH = 2048 * 216  # 10.03s of audio for 2048-buffer
# BACKGROUND_LENGTH = 2048 * 1296  # 60.19s of audio for 2048-buffer
SNIPPET_LENGTH = 2048 * 50
BACKGROUND_LENGTH = 2048 * 200

# How much time (in seconds) that the meas_in stream is ahead of ref_in stream.
# This parameter can be calibrated with help of the interface
MEAS_REF_LATENCY = 0.155
LATENCY_MEASUREMENT_LENGTH = 2048 * 100

# F_LIMITS = [20, 20480]
# OCT_FRAC = 1 / 24
F_LIMITS = [30, 18000]
OCT_FRAC = 1 / 24

EXPORT_WAV = True  # Export wav

# For plots in interface
FONTSIZE_TITLES = 12
FONTSIZE_LABELS = 10
FONTSIZE_LEGENDS = 9
FONTSIZE_TICKS = 9

assert SNIPPET_LENGTH % BUFFER == 0, "SNIPPET_LENGTH must be an integer multiple of BUFFER"
assert BACKGROUND_LENGTH % BUFFER == 0, "BACKGROUND_LENGTH must be an integer multiple of BUFFER"
assert LATENCY_MEASUREMENT_LENGTH % BUFFER == 0, "LATENCY_MEASUREMENT_LENGTH must be an integer multiple of BUFFER"
assert BACKGROUND_LENGTH >= SNIPPET_LENGTH, "BACKGROUND_LENGTH must be greater than SNIPPET_LENGTH"
