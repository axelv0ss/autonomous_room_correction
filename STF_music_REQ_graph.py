import numpy as np
from os import listdir
from matplotlib import pyplot as plt
from scipy.io.wavfile import read, write
import itertools


def bg_create_snippets(background_rec):
    """
    Splits background_rec into smaller snippets of length SNIPPET_LENGTH.
    Splits background_rec into as many whole multiples of length SNIPPET_LENGTH as possible.
    x is time, y is amplitude.
    """
    num_snippets = int(np.floor(len(background_rec) / SNIPPET_LENGTH))
    print("Splitting into {0} background time domain snippets of duration {1}s..."
          .format(num_snippets, round(SNIPPET_LENGTH / RATE, 2)))
    
    snippets_td = list()
    
    sampling_intervals = [(i * SNIPPET_LENGTH, (i + 1) * SNIPPET_LENGTH) for i in range(num_snippets)]
    for si, sf in sampling_intervals:
        x = np.arange(0, (sf - si) / RATE, 1 / RATE, dtype=NP_FORMAT)
        y = background_rec[si:sf]
        assert len(x) == len(y)
        snippets_td.append((x, y))
    
    return snippets_td


def bg_hanning_snippets_td(snippets_td):
    """
    Applies a Hanning envelope to all snippets
    Used to fulfill FFT requirement that time-domain signal needs to be periodic (zeros at beginning and end).
    Note: The Hanning function reduces the power contaied in the signal by 1/2.
          Compensate for this by multiplying the signal by 2:
          http://www.azimadli.com/vibman/thehanningwindow.htm
    """
    print("Applying Hanning window to {0} time domain snippets...".format(len(snippets_td)))
    snippets_td = [(x_td, 2 * y_td * np.hanning(len(y_td))) for x_td, y_td in snippets_td]
    return snippets_td


def bg_ft_snippets(snippets_td):
    """
    Function to calculate and append absolute value of Fourier Transform for each snippet to the list
    snippets_fd, where snippets_fd = [(x1, y1), (x2, y2), ... , (xn, yn)]

    The (complex) FFT is converted to magnitude [dBFS] using convert_to_dbfs().

    The resulting arrays are: x [Hz] and y [dBFS].
    """
    print("Fourier transforming {0} snippets...".format(len(snippets_td)))
    
    snippets_fd = list()
    
    # Transform all snippets
    for x_td, y_td in snippets_td:
        x_fd, y_fd = fourier_transform(y_td)
        snippets_fd.append((x_fd, y_fd))
    
    # Take magnitude (abs) and convert to dB
    snippets_fd = [(x_fd, convert_to_dbfs(y_fd)) for x_fd, y_fd in snippets_fd]
    return snippets_fd


def bg_mask_snippets_fd(snippets_fd):
    """
    Masks the data to return only within the desired frequency range (in Hz)
    F_LIMITS = (lo_lim, hi_lim)
    """
    print("Masking {0} frequency domain snippets between {1}Hz and {2}Hz..."
          .format(len(snippets_fd), F_LIMITS[0], F_LIMITS[1]))
    
    temp = list()
    for x, y in snippets_fd:
        mask = np.logical_and(F_LIMITS[0] < x, x < F_LIMITS[1])
        temp.append((x[mask], y[mask]))
    snippets_fd = temp
    
    return snippets_fd


def bg_generate_bg_model_fd(snippets_fd):
    """
    Generates a non-smoothened model of the background by taking the average y-values of all snippets
    """
    print("Generating background model from {0} frequency domain snippets...".format(len(snippets_fd)))
    # Ensure all frequency steps (x-elements) of the FT'd snippets are the same
    for i, j in itertools.permutations(range(len(snippets_fd)), 2):
        assert (snippets_fd[i][0] == snippets_fd[j][0]).all(), \
            "Frequency steps (x-elements) in self.snippets_fd are not the same, cannot generate model"
    
    # Do the averaging
    x_model = snippets_fd[0][0]
    y_model = np.array(snippets_fd)[:, 1].mean(axis=0)
    
    bg_model_fd = (x_model, y_model)
    
    return bg_model_fd


def bg_smoothen_bg_model_fd(bg_model_fd):
    """
    Smoothens the background model
    """
    print("Smoothening the background model with {0} octave bins...".format(round(OCT_FRAC, 4)))
    x_model, y_model = smoothen_data_fd(*bg_model_fd)
    bg_model_fd = (x_model, y_model)
    
    return bg_model_fd


def bg_smoothen_snippets_fd(snippets_fd):
    """
    Smoothens the snippets.
    The smoothened snippets are never used in calculations,
    only for plotting on the same axes as the model for reference.
    """
    snippets_fd_smooth = list()
    print("Smoothening {0} snippets with {1} octave bins..."
          .format(len(snippets_fd), round(OCT_FRAC, 4)))
    for x, y in snippets_fd:
        snippets_fd_smooth.append(smoothen_data_fd(x, y))
    
    return snippets_fd_smooth


def fourier_transform(y_td):
    """
    This function calculates the magnitude and performs normalisation operations to preserve magnitude between the
    time and frequency domains. Returns the positive frequency data only.

    y_td is the time domain data.
    Returns the (complex) transformed data in linear scale (not dBFS, use convert_to_dbfs() for that).
    """
    # N is number of samples, dt is time-domain spacing between samples
    N = len(y_td)

    # Transform the data and obtain the frequency array
    y_fd = np.fft.fft(y_td)
    x_fd = np.fft.fftfreq(N, 1 / RATE)

    # Mask the negative frequencies
    mask = x_fd > 0
    x_fd = x_fd[mask]
    y_fd = y_fd[mask]

    # Multiply by 2 to compensate for the loss of negative frequencies (half the spectrum)
    # Divide by number of samples to compensate for the duration of the signal
    # These operations together ensure that magnitude is preserved and normalised
    y_fd = 2 * y_fd / N

    return x_fd, y_fd


def convert_to_dbfs(y):
    """
    Converts a complex-valued input array y into dBFS units using
    [dB] = 20 * log10(mag/ref), where ref = 1
    """
    return 20 * np.log10(np.abs(y))


def smoothen_data_fd(x_in, y_in):
    """
    Takes the linear average (since y-data is [dB]) of the elements in each bin to obtain a smoother function.
    Takes in arrays of FT spectrum [dB] and log-bins them using a spacing of OCT_FRAC.
    """
    # Provides evenly spaced bins on log axes. x_out provides the center of each bin.
    # The last bin edge is unlikely to be equal to F_LIMITS[1], so take the last bin to encapsulate F_LIMITS[1].
    last_step = np.log2(F_LIMITS[1] / F_LIMITS[0])
    bin_edges = F_LIMITS[0] * 2 ** np.arange(0, last_step + OCT_FRAC, step=OCT_FRAC)
    x_out = np.sqrt(bin_edges[:-1] * (bin_edges[1:]))

    # print(len(x_in))

    # Vectorised for efficiency
    indices = np.digitize(x_in, bin_edges)
    # print(indices)
    # print(len(x_in))
    unique_vals, i_start, count = np.unique(indices, return_counts=True, return_index=True)
    y_out = np.zeros(len(unique_vals))

    for i in range(len(unique_vals)):
        # Take the elements of the y_raw data corresponding to the x_raw values grouped by bin
        # Linear average since we are working in dB (log) scale
        # print(x_in[i_start[i]:i_start[i] + count[i]])
        y_out[i] = np.average(y_in[i_start[i]:i_start[i] + count[i]])

    assert len(x_out) == len(y_out), "Empty bins were encountered in smoothing! Check resolution in frequency " \
                                     "domain vs parameter OCT_FRAC.\nTry increasing the length of the recording or " \
                                     "increase OCT_FRAC.\nlen(x)={0}, len(y)={1}\nbin_edges={2}\nx_in={3}" \
                                     .format(len(x_out), len(y_out), bin_edges, x_in)
    return x_out, y_out


def hanning_snippets_td(ref_in_snippet_td, meas_in_snippet_td, verbose=True):
    """
    Applies a Hanning envelope to the reference and measurement snippets.
    Used to fulfill FFT requirement that time-domain signal needs to be periodic (zeros at beginning and end).
    Note: The Hanning function reduces the power contaied in the signal by 1/2.
          Compensate for this by multiplying the signal by 2:
          http://www.azimadli.com/vibman/thehanningwindow.htm
    """
    if verbose: print("Applying Hanning window to ref_in and meas_in time domain snippets...")
    window = np.hanning(len(ref_in_snippet_td[1]))
    ref_in_snippet_td[1] = 2 * window * ref_in_snippet_td[1]
    meas_in_snippet_td[1] = 2 * window * meas_in_snippet_td[1]
    return ref_in_snippet_td, meas_in_snippet_td


def ft_snippets(ref_in_snippet_td, meas_in_snippet_td, verbose=True):
    """
    Function to calculate and append absolute value of Fourier Transform for the reference and measurement snippets.

    The (complex) FFT is converted to magnitude [dBFS] using convert_to_dbfs().

    The resulting arrays are: x [Hz] and y [dBFS].
    """
    if verbose: print("Fourier transforming ref_in and meas_in snippets...")

    # Transform both snippets
    x_fd_ref, y_fd_ref = fourier_transform(ref_in_snippet_td[1])
    x_fd_meas, y_fd_meas = fourier_transform(meas_in_snippet_td[1])

    # Take magnitude (abs) and convert to dB
    ref_in_snippet_fd = [x_fd_ref, convert_to_dbfs(y_fd_ref)]
    meas_in_snippet_fd = [x_fd_meas, convert_to_dbfs(y_fd_meas)]

    return ref_in_snippet_fd, meas_in_snippet_fd


def mask_snippets_fd(ref_in_snippet_fd, meas_in_snippet_fd, verbose=True):
    """
    Masks the data to return only within the desired frequency range (in Hz)
    F_LIMITS = (lo_lim, hi_lim)
    """
    if verbose:
        print("Masking ref_in and meas_in frequency domain snippets between {0}Hz and {1}Hz..."
              .format(F_LIMITS[0], F_LIMITS[1]))

    assert (ref_in_snippet_fd[0] == meas_in_snippet_fd[0]).all(), \
        "Frequency steps (x-elements) in ref_in_snippet_fd and meas_in_snippet_fd are not the same, " \
        "cannot continue."

    x_fd, y_fd_ref = ref_in_snippet_fd
    y_fd_meas = meas_in_snippet_fd[1]

    # Create and apply the mast
    mask = np.logical_and(F_LIMITS[0] < x_fd, x_fd < F_LIMITS[1])
    ref_in_snippet_fd = (x_fd[mask], y_fd_ref[mask])
    meas_in_snippet_fd = (x_fd[mask], y_fd_meas[mask])

    return ref_in_snippet_fd, meas_in_snippet_fd


def smoothen_snippets_fd(ref_in_snippet_fd, meas_in_snippet_fd, verbose=True):
    """
    Smoothens the ref_in and meas_in snippets.
    """
    if verbose:
        print("Smoothening ref_in and meas_in frequency domain snippets with {0} octave bins..."
              .format(round(OCT_FRAC, 4)))

    x_fd, y_fd_ref = ref_in_snippet_fd
    y_fd_meas = meas_in_snippet_fd[1]

    # Save as lists to preserve mutability (required in subtract_bg_from_meas())
    ref_in_snippet_fd_smooth = list(smoothen_data_fd(x_fd, y_fd_ref))
    meas_in_snippet_fd_smooth = list(smoothen_data_fd(x_fd, y_fd_meas))

    return ref_in_snippet_fd_smooth, meas_in_snippet_fd_smooth


def subtract_bg_from_meas(bg_model, meas_in_snippet_fd_smooth, verbose=True):
    """
    Subtracts the pre-generated background model from the meas_in snippet.
    Uses the fancy formula we derived (see lab book page 36)
    """
    if bg_model is None:
        raise Exception

    if verbose: print("Subtracting the background model from the meas_in frequency domain snippet...")
    assert (meas_in_snippet_fd_smooth[0] == bg_model[0]).all(), \
        "Cannot subtract background from meas_in: Their frequency steps (x-values) are not identical!"

    y_fd_bg = bg_model[1]
    x_fd_meas, y_fd_meas = meas_in_snippet_fd_smooth

    # If background is louder than measurement, raise exception
    if (y_fd_meas - y_fd_bg <= 0).any():
        raise Exception

    y_fd_meas = 20 * np.log10(10 ** (y_fd_meas/20) - 10 ** (y_fd_bg/20))

    meas_in_snippet_fd_smooth = [x_fd_meas, y_fd_meas]

    return meas_in_snippet_fd_smooth


def normalise_snippets_fd(ref_in_snippet_fd_smooth, meas_in_snippet_fd_smooth, verbose=True):
    """
    Normalises the magnitude of the snippets such that they are both centered at 0 dBFS.
    Calculates the average magnitude of the meas_in and ref_in snippets and compensated with how far it is from
    0 dBFS.
    """
    if verbose: print("Normalising ref_in and meas_in frequency domain snippets to 0 dBFS...")
    avg_ref = np.average(ref_in_snippet_fd_smooth[1])
    avg_meas = np.average(meas_in_snippet_fd_smooth[1])

    ref_in_snippet_fd_smooth[1] -= avg_ref
    meas_in_snippet_fd_smooth[1] -= avg_meas

    return ref_in_snippet_fd_smooth, meas_in_snippet_fd_smooth


def calculate_stf(ref_in_snippet_fd_smooth, meas_in_snippet_fd_smooth, verbose=True):
    """
    Calculates the STF by subtracting ref_in from meas_in.
    """
    if verbose: print("Calculating STF...")
    x_stf = ref_in_snippet_fd_smooth[0]
    y_stf = meas_in_snippet_fd_smooth[1] - ref_in_snippet_fd_smooth[1]
    stf = [x_stf, y_stf]

    return stf


def calculate_ms(stf, verbose=True):
    """
    Calculates the value of the objective function: Mean-Squared.
    A measure of the STF curve's deviation from being flat.
    MS gives extra weight to outliers as these are extra sensitive to perceived sound.
    """
    ms = np.average(np.sum(np.square(stf[1])))
    if verbose: print("Calculated MS: {0}".format(ms))
    return ms


# wd = "H://MSci//Initial_investigation"
wd = "//Users//axel//Documents//_Coursework//Y4//MSci_Project//_MSci//stf_vs_req_graph////"

song_names = ("Shufokan by Snarky Puppy",
              "Where You Come From by Disclosure",
              "Africa by Toto",
              "Levels by Avicii",
              "The Pretender by Foo Fighters",
              "Toxic by Britney Spears",
              "Could You Be Loved by Bob Marley",
              "Homecoming by Kanye West",
              "I'll Fly For You by Spandau Ballet",
              "Under The Bridge by Red Hot Chili Peppers"
              )

song_intervals = [(45, 365),
                  (461, 768),
                  (864, 1143),
                  (1210, 1540),
                  (1632, 1865),
                  (1935, 2131),
                  (2183, 2390),
                  (2440, 2638),
                  (2700, 2984),
                  (3090, 3269)
                  ]

SAMPLE_DELAY_SEC = 0.104
SNIPPET_LENGTH_SEC = 5

# Audio parameters
RATE = 44100
NP_FORMAT = np.float32

# Program parameters
F_LIMITS = [40, 19000]
OCT_FRAC = 1 / 6

# For plots in interface
FONTSIZE_TITLES = 12
FONTSIZE_LABELS = 15
FONTSIZE_LEGENDS = 13
FONTSIZE_TICKS = 15


SNIPPET_LENGTH = RATE * SNIPPET_LENGTH_SEC
SAMPLE_DELAY = int(RATE * SAMPLE_DELAY_SEC)


def produce_background():
    file_path = wd + "music_10composite_32bFlt44p1K_background.wav"
    
    composite_in = read(file_path)
    background_rec = composite_in[1]
    
    snippets_td = bg_create_snippets(background_rec)
    snippets_td = bg_hanning_snippets_td(snippets_td)
    snippets_fd = bg_ft_snippets(snippets_td)
    snippets_fd = bg_mask_snippets_fd(snippets_fd)
    bg_model_fd = bg_generate_bg_model_fd(snippets_fd)
    bg_model_fd = bg_smoothen_bg_model_fd(bg_model_fd)
    snippets_fd_smooth = bg_smoothen_snippets_fd(snippets_fd)
    
    # # Plot raw and model
    # f, ax = plt.subplots(1, 1, figsize=(12,7))
    #
    # for i in range(len(snippets_fd_smooth)):
    #     if i == 0:
    #         ax.semilogx(snippets_fd_smooth[i][0], snippets_fd_smooth[i][1], color="grey", linewidth=1, label="Raw")
    #     else:
    #         ax.semilogx(snippets_fd_smooth[i][0], snippets_fd_smooth[i][1], color="grey", linewidth=1)
    #
    # ax.semilogx(*bg_model_fd, label="Model", linewidth=1, color="black")
    #
    # ax.grid(which="major", linestyle="-", alpha=0.4)
    # ax.grid(which="minor", linestyle="--", alpha=0.2)
    # ax.minorticks_on()
    # ax.legend()
    # plt.suptitle("Background Model: Smootened RMS Average of n={0} Backgrounds".format(len(snippets_fd_smooth)))
    # plt.show()
    
    return bg_model_fd


def produce_req_curve():
    file_path = wd + "REQ_TDS_19022019_HS8_centre.txt"
    freq = []
    db = []
    with open(file_path, "r") as infile:
        for line in infile:
            if line[0] != "*":
                freq.append(float(line.split(" ")[0]))
                db.append(float(line.split(" ")[1]))
    req_freq_raw = np.array(freq)
    req_db_raw = np.array(db)
    
    # Mask
    m = np.logical_and(req_freq_raw >= F_LIMITS[0], req_freq_raw <= F_LIMITS[1])
    req_freq_raw = req_freq_raw[m]
    req_db_raw = req_db_raw[m]
    
    # plt.figure(figsize=(15, 8))
    # plt.semilogx(req_freq_raw, req_db_raw, label="REQ")
    # plt.title("RTF Measurement using REQ TDS")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Level (dB)")
    # plt.legend()
    # plt.minorticks_on()
    # plt.grid(which="major", linestyle="-", alpha=0.4)
    # plt.grid(which="minor", linestyle="--", alpha=0.2)
    #
    # plt.show()

    req_freq, req_db = smoothen_data_fd(req_freq_raw, req_db_raw)

    # plt.figure(figsize=(15, 8))
    # plt.semilogx(req_freq_raw, req_db_raw, label="REQ")
    # plt.semilogx(req_freq, req_db, label="REQ 1/6 Oct Smoothing")
    # plt.title("RTF Measurement using REQ TDS")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Level (dB)")
    # plt.legend()
    # plt.minorticks_on()
    # plt.grid(which="major", linestyle="-", alpha=0.4)
    # plt.grid(which="minor", linestyle="--", alpha=0.2)
    #
    # plt.show()
    
    return req_freq, req_db
    
    
def plot_all_stfs(bg_model_fd, req_freq, req_db):
    file_path_ref = wd + "music_10composite_32bFlt44p1K_reference.wav"
    file_path_mes = wd + "music_10composite_32bFlt44p1K_measurement.wav"
    
    composite_in_ref = read(file_path_ref)
    composite_ref = composite_in_ref[1]
    
    composite_in_mes = read(file_path_mes)
    composite_mes = composite_in_mes[1]
    
    stfs = dict()
    
    x_td = np.arange(0, SNIPPET_LENGTH_SEC, 1 / RATE, dtype=NP_FORMAT)
    
    for name, (start, stop) in zip(song_names, song_intervals):
        start += int((stop - start) / 3)
        
        start_index = start * RATE
        stop_index = (start + SNIPPET_LENGTH_SEC) * RATE
        
        ref_in_snippet_td = [x_td, composite_ref[start_index:stop_index]]
        meas_in_snippet_td = [x_td, composite_mes[start_index + SAMPLE_DELAY:stop_index + SAMPLE_DELAY]]
        
        write(wd + "{0}_ref_in.wav".format(name), RATE, ref_in_snippet_td[1])
        write(wd + "{0}_meas_in.wav".format(name), RATE, meas_in_snippet_td[1])
        
        v = True  # Verbose
        ref_in_snippet_td, meas_in_snippet_td = hanning_snippets_td(ref_in_snippet_td, meas_in_snippet_td, v)
        ref_in_snippet_fd, meas_in_snippet_fd = ft_snippets(ref_in_snippet_td, meas_in_snippet_td, v)
        ref_in_snippet_fd, meas_in_snippet_fd = mask_snippets_fd(ref_in_snippet_fd, meas_in_snippet_fd, v)
        ref_in_snippet_fd, meas_in_snippet_fd = smoothen_snippets_fd(ref_in_snippet_fd, meas_in_snippet_fd, v)
        try:
            meas_in_snippet_fd = subtract_bg_from_meas(bg_model_fd, meas_in_snippet_fd, v)
        except:
            print("!! QuietMeasurementError: Skipped {0}".format(name))
            continue
        ref_in_snippet_fd, meas_in_snippet_fd = normalise_snippets_fd(ref_in_snippet_fd, meas_in_snippet_fd, v)
        stf = calculate_stf(ref_in_snippet_fd, meas_in_snippet_fd, v)
        
        stfs[name] = stf

        # ms = calculate_ms(stf, v)
        # ms_vals.append(ms)

    plt.figure(1, figsize=(9, 7))

    req_db_normalised = req_db - np.average(req_db)
    plt.semilogx(req_freq, req_db_normalised, label="Time Delay Spectrometry", color="black")

    for name, stf in stfs.items():
        plt.semilogx(*stf, label=name, zorder=-1)

    # plt.title("Transfer Function Measurements Using TDS Versus Music")
    plt.xlabel("Frequency (Hz)", fontsize=FONTSIZE_LABELS)
    plt.ylabel("Magnitude (dB)", fontsize=FONTSIZE_LABELS)
    plt.legend(fontsize=FONTSIZE_LEGENDS)
    plt.minorticks_on()
    plt.grid(which="major", linestyle="-", alpha=0.4)
    plt.grid(which="minor", linestyle="--", alpha=0.2)
    plt.tick_params(labelsize=FONTSIZE_TICKS)
    plt.show()


def plot_all_stfs_poster(bg_model_fd, req_freq, req_db):
    FONTSIZE_TITLES = 12
    FONTSIZE_LABELS = 25
    FONTSIZE_LEGENDS = 20
    FONTSIZE_TICKS = 20
    
    file_path_ref = wd + "music_10composite_32bFlt44p1K_reference.wav"
    file_path_mes = wd + "music_10composite_32bFlt44p1K_measurement.wav"
    
    composite_in_ref = read(file_path_ref)
    composite_ref = composite_in_ref[1]
    
    composite_in_mes = read(file_path_mes)
    composite_mes = composite_in_mes[1]
    
    stfs = dict()
    
    x_td = np.arange(0, SNIPPET_LENGTH_SEC, 1 / RATE, dtype=NP_FORMAT)
    
    for name, (start, stop) in zip(song_names, song_intervals):
        start += int((stop - start) / 3)
        
        start_index = start * RATE
        stop_index = (start + SNIPPET_LENGTH_SEC) * RATE
        
        ref_in_snippet_td = [x_td, composite_ref[start_index:stop_index]]
        meas_in_snippet_td = [x_td, composite_mes[start_index + SAMPLE_DELAY:stop_index + SAMPLE_DELAY]]
        
        write(wd + "{0}_ref_in.wav".format(name), RATE, ref_in_snippet_td[1])
        write(wd + "{0}_meas_in.wav".format(name), RATE, meas_in_snippet_td[1])
        
        v = True  # Verbose
        ref_in_snippet_td, meas_in_snippet_td = hanning_snippets_td(ref_in_snippet_td, meas_in_snippet_td, v)
        ref_in_snippet_fd, meas_in_snippet_fd = ft_snippets(ref_in_snippet_td, meas_in_snippet_td, v)
        ref_in_snippet_fd, meas_in_snippet_fd = mask_snippets_fd(ref_in_snippet_fd, meas_in_snippet_fd, v)
        ref_in_snippet_fd, meas_in_snippet_fd = smoothen_snippets_fd(ref_in_snippet_fd, meas_in_snippet_fd, v)
        try:
            meas_in_snippet_fd = subtract_bg_from_meas(bg_model_fd, meas_in_snippet_fd, v)
        except:
            print("!! QuietMeasurementError: Skipped {0}".format(name))
            continue
        ref_in_snippet_fd, meas_in_snippet_fd = normalise_snippets_fd(ref_in_snippet_fd, meas_in_snippet_fd, v)
        stf = calculate_stf(ref_in_snippet_fd, meas_in_snippet_fd, v)
        
        stfs[name] = stf
        
        # ms = calculate_ms(stf, v)
        # ms_vals.append(ms)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    req_db_normalised = req_db - np.average(req_db)
    plt.semilogx(req_freq, req_db_normalised, label="Time Delay Spectrometry", color="black", linewidth=1.8)
    
    first = True
    for name, stf in stfs.items():
        if first:
            first = False
            plt.semilogx(*stf, label="10 Songs of Varying Genre", zorder=-1, color="gray", linewidth=1.8)
        else:
            plt.semilogx(*stf, zorder=-1, color="gray", linewidth=1.8)
    
    # plt.title("Transfer Function Measurements Using TDS Versus Music")
    plt.xlabel("Frequency (Hz)", fontsize=FONTSIZE_LABELS)
    plt.ylabel("Magnitude (dB)", fontsize=FONTSIZE_LABELS)
    plt.legend(fontsize=FONTSIZE_LEGENDS)
    plt.minorticks_on()
    plt.grid(which="major", linestyle="-", alpha=0.4)
    plt.grid(which="minor", linestyle="--", alpha=0.2)
    plt.tick_params(labelsize=FONTSIZE_TICKS)

    plt.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig("music_stf.png", transparent=False, dpi=800)


bg_model_fd = produce_background()
req_freq, req_db = produce_req_curve()

plot_all_stfs_poster(bg_model_fd, req_freq, req_db)
