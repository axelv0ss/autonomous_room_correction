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
# Can be calibrated by enabling wav export and see how much lag there is
MEAS_REF_LATENCY = 0.569
LATENCY_MEASUREMENT_LENGTH = 212992

# Program parameters
EXPORT_WAV = True
F_LIMITS = [40, 20000]  # TEST
# F_LIMITS = [500, 15000]  # DEV
OCT_FRAC = 1 / 6

# After this number of attempts of creating a chain with non-overlapping filters from the
# filter pool is reached, wipe the current attempt and start again.
MAX_CROSSOVER_ATTEMPTS = 100
MAX_MUTATION_ATTEMPTS = 500

# For evolutionary algorithm
POP_SIZE = 100
NUM_FILTERS = 5
GAIN_LIMITS = [-10, 10]
Q_LIMITS = [1, 6]

PROP_PROMOTED = 0.33  # Proportion of chains promoted in each iteration

STDEV_FC = 0.5 * 2  # Standard deviation (proportion) with which to mutate fc
STDEV_GAIN = 2 * 2  # Standard deviation (linear) with which to mutate gain
STDEV_Q = 0.5 * 2  # Standard deviation (proportion) with which to mutate Q

PROP_RND = 0  # Proportion of random filters added in filter pool

# To prevent signal clipping (dB)
ATTENUATE_OUTPUT = -0

# For plots in interface
FONTSIZE_TITLES = 12
FONTSIZE_LABELS = 10
FONTSIZE_LEGENDS = 9
FONTSIZE_TICKS = 9

# Routing, audio device indices
REF_IN_INDEX = 3  # Soundflower (2ch)
MEAS_OUT_INDEX = 2  # soundcard
MEAS_IN_INDEX = 2  # soundcard

# REF_IN_INDEX = 2  # Soundflower (2ch)
# MEAS_OUT_INDEX = 1  # Built-in Output
# MEAS_IN_INDEX = 0  # Built-in Microphone

assert SNIPPET_LENGTH % BUFFER == 0, "SNIPPET_LENGTH must be an integer multiple of BUFFER"
assert BACKGROUND_LENGTH % BUFFER == 0, "BACKGROUND_LENGTH must be an integer multiple of BUFFER"
assert LATENCY_MEASUREMENT_LENGTH % BUFFER == 0, "LATENCY_MEASUREMENT_LENGTH must be an integer multiple of BUFFER"
assert BACKGROUND_LENGTH >= SNIPPET_LENGTH, "BACKGROUND_LENGTH must be greater than SNIPPET_LENGTH"
assert int(POP_SIZE * PROP_PROMOTED) >= 2, \
    "The number of promoted chains every iteration must not be less than 2 for meaningful crossover, but is {0}"\
    .format(int(POP_SIZE * PROP_PROMOTED))
