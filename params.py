"""
Specify necessary parameters for the project.
"""

import pyaudio
import numpy as np

# Audio parameters
RATE = 44100
PA_FORMAT = pyaudio.paFloat32
NP_FORMAT = np.float32
BUFFER = 16384

# Need to be multiples of BUFFER
# # DEV
# SNIPPET_LENGTH = 16384 * 10
# BACKGROUND_LENGTH = 16384 * 50
# # TEST
SNIPPET_LENGTH = 16384 * 14  # 5s of audio for 16384-buffer
BACKGROUND_LENGTH = SNIPPET_LENGTH * 10

# How much time (in seconds) that the meas_in stream is ahead of ref_in stream.
# This parameter can be calibrated with help of the interface
MEAS_REF_LATENCY = 0.590
LATENCY_MEASUREMENT_LENGTH = 212992

# Program parameters
EXPORT_WAV = True
# F_LIMITS = [40, 20000]  # TEST
F_LIMITS = [500, 15000]  # DEV
OCT_FRAC = 1 / 6

# For evolutionary algorithm
POP_SIZE = 10
NUM_FILTERS = 6
GAIN_LIMITS = [-5, 5]
Q_LIMITS = [1, 6]

PROP_PROMOTED = 0.3  # Proportion of chains promoted in each iteration

PROB_MUT = 0.3  # Probability of independently mutating a each parameter of each filter in filter pool
STDEV_FC = 0.2  # Standard deviation (proportion) with which to mutate fc
STDEV_GAIN = 1  # Standard deviation (linear) with which to mutate gain
STDEV_Q = 0.2  # Standard deviation (proportion) with which to mutate Q

PROP_RND = 0.3  # Proportion of random filters added in filter pool

# To prevent signal clipping (dB)
ATTENUATE_OUTPUT = -0

# For plots in interface
FONTSIZE_TITLES = 12
FONTSIZE_LABELS = 10
FONTSIZE_LEGENDS = 9
FONTSIZE_TICKS = 9

# Routing, audio device indices
# REF_IN_INDEX = 3  # Soundflower (2ch)
# MEAS_OUT_INDEX = 2  # soundcard
# MEAS_IN_INDEX = 2  # soundcard

REF_IN_INDEX = 2  # Soundflower (2ch)
MEAS_OUT_INDEX = 1  # Built-in Output
MEAS_IN_INDEX = 0  # Built-in Microphone

assert SNIPPET_LENGTH % BUFFER == 0, "SNIPPET_LENGTH must be an integer multiple of BUFFER"
assert BACKGROUND_LENGTH % BUFFER == 0, "BACKGROUND_LENGTH must be an integer multiple of BUFFER"
assert LATENCY_MEASUREMENT_LENGTH % BUFFER == 0, "LATENCY_MEASUREMENT_LENGTH must be an integer multiple of BUFFER"
assert BACKGROUND_LENGTH >= SNIPPET_LENGTH, "BACKGROUND_LENGTH must be greater than SNIPPET_LENGTH"
assert int(POP_SIZE * PROP_PROMOTED) >= 2, \
    "The number of promoted chains every iteration must not be less than 2 for meaningful crossover, but is {0}"\
    .format(int(POP_SIZE * PROP_PROMOTED))
