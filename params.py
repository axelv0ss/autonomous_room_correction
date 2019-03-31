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
# MEAS_REF_LATENCY = 0.569  # Macbook
MEAS_REF_LATENCY = 0.585  # Mac Pro
LATENCY_MEASUREMENT_LENGTH = 212992

# Program parameters
EXPORT_WAV = True
F_LIMITS = [40, 20000]  # TEST
# F_LIMITS = [500, 15000]  # DEV
OCT_FRAC = 1 / 6

# After this number of attempts of creating a chain with non-overlapping filters from the
# filter pool is reached, wipe the current attempt and start again.
MAX_CROSSOVER_ATTEMPTS = 100
# After this number of attempts of mutating the filter such that it doesn't overlap with any existing
# filters, just use the original parameters (non-mutated). In this case it is called a "failed mutation".
MAX_MUTATION_ATTEMPTS = 1000

# For evolutionary algorithm
POP_SIZE = 100
NUM_FILTERS = 10
GAIN_LIMITS = [-15, 15]
Q_LIMITS = [1, 9]

PROP_PROMOTED = 0.3  # Proportion of chains promoted in each iteration

STDEV_FC = 0.5 * 0.5  # Standard deviation (proportion) with which to mutate fc
STDEV_GAIN = 2 * 0.5  # Standard deviation (linear) with which to mutate gain
STDEV_Q = 0.5 * 0.5   # Standard deviation (proportion) with which to mutate Q

PROP_RND = 0  # Proportion of random filters added in filter pool

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

# Mac Pro
REF_IN_INDEX = 3  # Soundflower (2ch)
# MEAS_OUT_INDEX = 2  # M-Track Hub
MEAS_OUT_INDEX = 1  # UMC404HD 192k
MEAS_IN_INDEX = 1  # UMC404HD 192k

assert SNIPPET_LENGTH % BUFFER == 0, "SNIPPET_LENGTH must be an integer multiple of BUFFER"
assert BACKGROUND_LENGTH % BUFFER == 0, "BACKGROUND_LENGTH must be an integer multiple of BUFFER"
assert LATENCY_MEASUREMENT_LENGTH % BUFFER == 0, "LATENCY_MEASUREMENT_LENGTH must be an integer multiple of BUFFER"
assert BACKGROUND_LENGTH >= SNIPPET_LENGTH, "BACKGROUND_LENGTH must be greater than SNIPPET_LENGTH"
assert int(POP_SIZE * PROP_PROMOTED) >= 2, \
    "The number of promoted chains every iteration must not be less than 2 for meaningful crossover, but is {0}"\
    .format(int(POP_SIZE * PROP_PROMOTED))
