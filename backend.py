"""
Has a class called Program with all necessary methods
- Shared arrays,
- Process functions.
- Program
- FilterChain
- PeakFilter
Have mp process objects defined and spawned within the Program class in init()
Have e.g. model_background, run_algorithm methods... (see program outline doc)
"""

import time
from params import *
from scipy import signal
import threading
from PyQt5 import QtCore
import queue
import itertools


class PeakFilter(object):
    def __init__(self, fc, gain, q):
        """
        Filter object to be used in FilterChain

        :param fc: The center frequency (in Hz) of the filter
        :param gain: The gain (in dB) of the filter
        :param q: The Q-factor of the filter
        """
        self.set_params(fc, gain, q)

    def set_params(self, fc, gain, q):
        """
        Used by FilterChain to set parameters of each filter
        """
        # Save new instance variables
        self.fc = fc
        self.gain = gain
        self.q = q

        # Frequencies declared as fractions of sampling rate
        f_frac = self.fc / RATE
        w_frac = 2 * np.pi * f_frac

        # Declare Transfer Function in S-domain
        num = [1, 10 ** (self.gain / 20) * w_frac / self.q, w_frac ** 2]
        den = [1, w_frac / self.q, w_frac ** 2]

        # Calculate coefficients in Z-domain using Bi-linear transform
        self.b, self.a = signal.bilinear(num, den)

    def get_b_a(self):
        """
        Used by FilterChain to get digital filter coefficients
        """
        return self.b[:], self.a[:]

    def get_settings(self):
        """
        Used by FilterChain to print the state of the filter chain.
        Also used as legend in plot.
        """
        return "fc={0}, g={1}, q={2}".format(self.fc, self.gain, self.q)

    def get_tf(self):
        """
        Get the (complex) Transfer Function of the filter.
        Needs to be converted to dB using 20*log(abs(H))
        """
        w_f, h = signal.freqz(*self.get_b_a())
        f = w_f * RATE / (2 * np.pi)
        return f, h


class FilterChain(object):
    def __init__(self, *filters):
        """
        *filters: Arbitrary number of filter objects, e.g. PeakFilter
        """
        self.filters = filters

        # Calculate initial conditions
        self.zi = [signal.lfilter_zi(*filt.get_b_a()) for filt in self.filters]

    def filter_signal(self, data_in):
        """
        Filters the current buffer and updates initial conditions for all filters in the chain accordingly
        """
        data_out = data_in
        # Filter data and update all initial conditions for the buffer of data
        for i, filt in enumerate(self.filters):
            data_out, self.zi[i] = signal.lfilter(*filt.get_b_a(), data_out, zi=self.zi[i])
        return data_out

    def set_filter_params(self, i, fc, gain, q):
        """
        Updates the filter of index i with new settings fc, gain and q.
        """
        # Construct an iterator to go through the elements as required, ignoring the FLAG
        self.filters[i].set_params(fc, gain, q)

    def get_num_filters(self):
        return len(self.filters)

    def get_chain_tf(self):
        """
        Get the (complex) Transfer Function of the chain.
        Needs to be converted to dB using 20*log(abs(H))
        """
        H = 1
        for filt in self.filters:
            f, h = filt.get_tf()
            H *= h
        return f, H

    def get_chain_settings(self):
        """
        Returns the current configuration of the filter chain in string form.
        """
        out = "Current settings of filter chain:"
        for i, filt in enumerate(self.filters):
            out += "\n{0} {1}".format(i, filt.get_settings())
        return out

    def get_all_filters_settings_tf(self):
        """
        Get the current settings and transfer function of all filters in the chain.
        Returns a list of dicts in the form:
        [{"settings": str0, "tf": [f0, h0]}, {"settings": str1, "tf": [f1, h1]}, ...]
        """
        out = list()
        for filt in self.filters:
            out.append({"settings": filt.get_settings(), "tf": filt.get_tf()})
        return out


class MainStream(QtCore.QThread):
    def __init__(self, ref_in_buffer_queue, meas_out_buffer_queue, chain, bypass_chain):
        super().__init__()
        self.paused = False
        self.shutting_down = False
        self.ref_in_buffer_queue = ref_in_buffer_queue
        self.meas_out_buffer_queue = meas_out_buffer_queue

        self.chain = chain
        self.bypass_chain = bypass_chain

    def run(self):
        p = pyaudio.PyAudio()
        # Initialise stream
        stream = p.open(format=PA_FORMAT,
                        frames_per_buffer=BUFFER,
                        rate=RATE,
                        channels=2,
                        input=True,
                        output=True,
                        input_device_index=2,
                        output_device_index=1,
                        stream_callback=self.callback)

        print("\nMain stream started!")
        print("Available audio devices by index:")
        for i in range(p.get_device_count()):
            print(i, p.get_device_info_by_index(i)['name'])

        while not self.shutting_down:
            if stream.is_active() and not self.paused:
                pass
            elif stream.is_active() and self.paused:
                print("\nMain stream paused!")
                stream.stop_stream()
            elif not stream.is_active() and not self.paused:
                print("\nMain stream started!")
                stream.start_stream()
            elif not stream.is_active() and self.paused:
                pass
            time.sleep(0.1)

        print("\nMain stream shut down!")
        stream.close()
        p.terminate()

    def callback(self, in_bytes, frame_count, time_info, flag):
        """
        Callback function for the ref_in/meas_out stream
        """
        in_data = np.fromstring(in_bytes, dtype=NP_FORMAT)   # Convert audio data in bytes to an array.
        in_data = np.reshape(in_data, (BUFFER, 2))        # Reshape array to two channels.
        in_data = np.average(in_data, axis=1)             # Convert audio data to mono by averaging channels.

        # Write ref_in data to shared queue.
        try:
            self.ref_in_buffer_queue.put_nowait(in_data)
        except queue.Full:
            self.ref_in_buffer_queue.get_nowait()
            self.ref_in_buffer_queue.put_nowait(in_data)

        # If the EQ settings have changed, apply them. Otherwise ignore.
        # try:
        #     self.chain = self.chain_queue.get_nowait()
        #     print("Chain settings updated!\n{0}".format(self.chain.get_chain_settings()))
        # except queue.Empty:
        #     pass

        # Check if bypass state has changed.
        # try:
        #     self.bypass = self.bypass_queue.get_nowait()
        # except queue.Empty:
        #     pass

        # Filter/process data
        if not self.bypass_chain.get_state():
            out_data = self.chain.filter_signal(in_data)
        else:
            out_data = in_data

        # Write meas_out data to shared queue.
        # TODO Is it even necessary to write this data to a shared queue??
        try:
            self.meas_out_buffer_queue.put_nowait(out_data)
        except queue.Full:
            self.meas_out_buffer_queue.get_nowait()
            self.meas_out_buffer_queue.put_nowait(out_data)

        out_data = np.repeat(out_data, 2)                   # Convert to 2-channel audio for compatib. with stream
        out_bytes = out_data.astype(NP_FORMAT).tostring()     # Convert audio data back to bytes and return
        return out_bytes, pyaudio.paContinue

    def toggle_pause(self):
        self.paused = not self.paused

    def shutdown(self):
        self.shutting_down = True


class MeasStream(QtCore.QThread):
    def __init__(self, meas_in_buffer_queue):
        super().__init__()
        # self.program_shutdown = program_shutdown
        self.shutting_down = False
        self.meas_in_buffer_queue = meas_in_buffer_queue

    def run(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=PA_FORMAT,
                        frames_per_buffer=BUFFER,
                        rate=RATE,
                        channels=1,
                        input=True,
                        input_device_index=0,
                        stream_callback=self.callback)

        print("\nMeas stream started!")

        while not self.shutting_down:
            # print(self.program_shutdown.is_set())
            time.sleep(0.1)

        print("\nMeas stream shut down!")
        stream.close()
        p.terminate()

    def callback(self, in_bytes, frame_count, time_info, flag):
        """
        Callback function for the meas_in stream
        """
        audio_data = np.fromstring(in_bytes, dtype=NP_FORMAT)  # Convert audio data in bytes to an array.

        # Write meas_in data to shared queue.
        try:
            self.meas_in_buffer_queue.put_nowait(audio_data)
        except queue.Full:
            self.meas_in_buffer_queue.get_nowait()
            self.meas_in_buffer_queue.put_nowait(audio_data)

        return in_bytes, pyaudio.paContinue

    def shutdown(self):
        self.shutting_down = True


class BackgroundModel(QtCore.QThread):
    def __init__(self, sendback_queue, meas_in_buffer_queue):
        super().__init__()
        self.sendback_queue = sendback_queue
        self.meas_in_buffer_queue = meas_in_buffer_queue
        self.background_rec = None
        self.snippets_td = list()
        self.snippets_fd = list()
        self.bg_model_fd = None
        self.snippets_fd_smooth = list()

    def run(self):
        """
        Do all the necessary processing etc to generate the model.
        Sends the data back to the GUI using
        """
        print("\nGenerating background model:")
        # Record
        self.record_background()

        # Process
        self.create_snippets()
        self.hanning_snippets_td()
        self.ft_snippets()
        self.mask_snippets_fd()
        self.generate_bg_model_fd()
        self.smoothen_bg_model_fd()
        self.smoothen_snippets_fd()

        # Send back the data
        print("Background model generated!")
        self.sendback_queue.put(self.bg_model_fd)
        self.sendback_queue.put(self.snippets_fd_smooth)

        # Finishing this run() function triggers any GUI listeners for the self.finished() flag

    def record_background(self):
        """
        Uses _record to record the background.
        Exports to wav for reference.
        Stops the main stream for quietness during background recording.
        """
        time.sleep(0.1)
        file_name = "BACKGROUND_REC.wav"
        print("Recording {0}s from ref_in_buffer_queue ({1} export={2})..."
              .format(round(BACKGROUND_LENGTH / RATE, 2), file_name, EXPORT_WAV))

        background_rec_queue = queue.Queue(maxsize=1)
        start_flag = threading.Event()
        start_flag.set()
        _record(self.meas_in_buffer_queue, background_rec_queue, BACKGROUND_LENGTH, start_flag=start_flag)
        self.background_rec = background_rec_queue.get()

        if EXPORT_WAV:
            from scipy.io.wavfile import write
            write(file_name, RATE, np.array(self.background_rec, dtype=NP_FORMAT))

        print("Recorded {0}s from ref_in_buffer_queue ({1} export={2})"
              .format(round(BACKGROUND_LENGTH / RATE, 2), file_name, EXPORT_WAV))

    def create_snippets(self):
        """
        Splits self.background_rec into smaller snippets of length SNIPPET_LENGTH.
        Splits self.background_rec into as many whole multiples of length SNIPPET_LENGTH as possible.
        x is time, y is amplitude.
        """
        num_snippets = int(np.floor(len(self.background_rec)/SNIPPET_LENGTH))
        print("Splitting into {0} background time domain snippets of duration {1}s..."
              .format(num_snippets, round(SNIPPET_LENGTH / RATE, 2)))

        sampling_intervals = [(i * SNIPPET_LENGTH, (i + 1) * SNIPPET_LENGTH) for i in range(num_snippets)]
        for si, sf in sampling_intervals:
            x = np.arange(0, (sf - si)/RATE, 1 / RATE, dtype=NP_FORMAT)
            y = self.background_rec[si:sf]
            assert len(x) == len(y)
            self.snippets_td.append((x, y))

    def hanning_snippets_td(self):
        """
        Applies a Hanning envelope to all snippets in self.snippets_td
        Used to fulfill FFT requirement that time-domain signal needs to be periodic (zeros at beginning and end).
        Note: The Hanning function reduces the power contaied in the signal by 1/2.
              Compensate for this by multiplying the signal by 2:
              http://www.azimadli.com/vibman/thehanningwindow.htm
        """
        print("Applying Hanning window to {0} time domain snippets...".format(len(self.snippets_td)))
        self.snippets_td = [(x_td, 2 * y_td * np.hanning(len(y_td))) for x_td, y_td in self.snippets_td]

    def ft_snippets(self):
        """
        Function to calculate and append absolute value of Fourier Transform for each snippet to the list
        self.snippets_fd, where self.snippets_fd = [(x1, y1), (x2, y2), ... , (xn, yn)]

        This function calculates the magnitude and performs normalisation operations to preserve magnitude between the
        time and frequency domains.

        The Fourier magnitudes are converted to decibels [dB] = 20 * log10(mag/ref), where ref = 1
        """
        print("Fourier transforming {0} snippets...".format(len(self.snippets_td)))

        # Transform all snippets
        for x_td, y_td in self.snippets_td:
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
            # Take absolute value for magnitude
            # These operations together ensure that magnitude is preserved and normalised
            y_fd = 2 * np.abs(y_fd / N)

            self.snippets_fd.append((x_fd, y_fd))

        # Convert to dB
        self.snippets_fd = [(x_fd, convert_to_dbfs(y_fd)) for x_fd, y_fd in self.snippets_fd]

    def mask_snippets_fd(self):
        """
        Masks the data to return only within the desired frequency range (in Hz)
        F_LIMITS = (lo_lim, hi_lim)
        """
        print("Masking {0} frequency domain snippets between {1}Hz and {2}Hz..."
              .format(len(self.snippets_fd), F_LIMITS[0], F_LIMITS[1]))

        temp = list()
        for x, y in self.snippets_fd:
            mask = np.logical_and(F_LIMITS[0] < x, x < F_LIMITS[1])
            temp.append((x[mask], y[mask]))
        self.snippets_fd = temp

    def generate_bg_model_fd(self):
        """
        Generates a non-smoothened model of the background by taking the average y-values of all snippets
        """
        print("Generating background model from {0} frequency domain snippets...".format(len(self.snippets_fd)))
        # Ensure all frequency steps (x-elements) of the FT'd snippets are the same
        for i, j in itertools.permutations(range(len(self.snippets_fd)), 2):
            assert (self.snippets_fd[i][0] == self.snippets_fd[j][0]).all(), \
                "Frequency steps (x-elements) in self.snippets_fd are not the same, cannot generate model"

        # Do the averaging
        x_model = self.snippets_fd[0][0]
        y_model = np.array(self.snippets_fd)[:, 1].mean(axis=0)  # TODO ensure this works as intended

        self.bg_model_fd = (x_model, y_model)

    def smoothen_bg_model_fd(self):
        """
        Smoothens the background model
        """
        print("Smoothening the background model with {0} octave bins...".format(round(OCT_FRAC, 4)))
        x_model, y_model = self.smoothen_data_fd(*self.bg_model_fd)
        self.bg_model_fd = (x_model, y_model)

    def smoothen_snippets_fd(self):
        """
        Smoothens the snippets.
        The smoothened snippets are never used in calculations,
        only for plotting on the same exes as the model for reference.
        """
        print("Smoothening {0} snippets with {1} octave bins..."
              .format(len(self.snippets_fd), round(OCT_FRAC, 4)))
        for x, y in self.snippets_fd:
            self.snippets_fd_smooth.append(self.smoothen_data_fd(x, y))

    @staticmethod
    def smoothen_data_fd(x_in, y_in):
        """
        Takes the linear average of the elements in each bin to obtain a smoother function.
        Takes in arrays of FT spectrum and log-bins them using a spacing of OCT_FRAC.
        """
        # Provides evenly spaced bins on log axes
        bin_edges = F_LIMITS[0] * 2 ** np.arange(0, 10 + OCT_FRAC, step=OCT_FRAC)
        x_out = np.sqrt(bin_edges[:-1] * (bin_edges[1:] - 1))

        # Vectorised for efficiency
        indices = np.digitize(x_in, bin_edges)
        unique_vals, i_start, count = np.unique(indices, return_counts=True, return_index=True)
        y_out = np.zeros(len(unique_vals))

        for i in range(len(unique_vals)):
            # Take the elements of the y_raw data corresponding to the x_raw values grouped by bin
            # Linear average since we are working in dB (log) scale
            y_out[i] = np.average(y_in[i_start[i]:i_start[i] + count[i]])

        assert len(x_out) == len(y_out), "Empty bins were encountered in smoothing! Check resolution in frequency " \
                                         "domain vs parameter OCT_FRAC.\nlen(x)={0}, len(y)={1}\nx={2}\ny={3}" \
                                         .format(len(x_out), len(y_out), x_out, y_out)
        return x_out, y_out


class LatencyCalibration(QtCore.QThread):
    """
    Used to determine the MEAS_REF_LATENCY value to synchronise the ref_in and meas_in streams.
    """
    def __init__(self, sendback_queue, ref_in_buffer_queue, meas_in_buffer_queue):
        super().__init__()
        self.sendback_queue = sendback_queue
        self.ref_in_buffer_queue = ref_in_buffer_queue
        self.meas_in_buffer_queue = meas_in_buffer_queue

    def run(self):
        """
        Records the two streams without any latency compensation and send the result back to the GUI for plotting.
        """
        print("\nRunning calibration recordings:")
        print("Calibration: Recording {0}s from ref_in_buffer_queue..."
              .format(round(LATENCY_MEASUREMENT_LENGTH / RATE, 2)))
        print("Calibration: Recording {0}s from meas_in_buffer_queue..."
              .format(round(LATENCY_MEASUREMENT_LENGTH / RATE, 2)))

        # Use queues and threading here to ensure we capture identical parts of the song
        ref_in_rec_queue = queue.Queue(maxsize=1)
        meas_in_rec_queue = queue.Queue(maxsize=1)

        # To ensure synchronised recording
        ref_in_start_flag = threading.Event()
        meas_in_start_flag = threading.Event()

        p_ref = threading.Thread(target=_record, args=(self.ref_in_buffer_queue, ref_in_rec_queue,
                                                       LATENCY_MEASUREMENT_LENGTH, ref_in_start_flag))
        p_meas = threading.Thread(target=_record, args=(self.meas_in_buffer_queue, meas_in_rec_queue,
                                                        LATENCY_MEASUREMENT_LENGTH, meas_in_start_flag))

        p_ref.start()
        p_meas.start()

        # To ensure synchronised recording
        time.sleep(0.2)
        ref_in_start_flag.set()
        time.sleep(MEAS_REF_LATENCY)
        meas_in_start_flag.set()

        p_ref.join()
        p_meas.join()

        ref_in_rec = ref_in_rec_queue.get()
        meas_in_rec = meas_in_rec_queue.get()

        # Return
        print("Calibration recordings finished!")
        x = np.arange(0, LATENCY_MEASUREMENT_LENGTH / RATE, 1 / RATE, dtype=NP_FORMAT)
        self.sendback_queue.put([x, ref_in_rec])
        self.sendback_queue.put([x, meas_in_rec])

        # Finishing this run() function triggers any GUI listeners for the self.finished() flag


class AlgorithmIteration(QtCore.QThread):
    def __init__(self, sendback_queue, ref_in_buffer_queue, meas_in_buffer_queue):
        super().__init__()
        self.sendback_queue = sendback_queue
        self.ref_in_buffer_queue = ref_in_buffer_queue
        self.meas_in_buffer_queue = meas_in_buffer_queue
        self.ref_in_snippet_td = None
        self.meas_in_snippet_td = None
        self.ref_in_snippet_fd = None
        self.meas_in_snippet_fd = None

    def run(self):
        print("\nRunning iteration:")
        # Record
        self.record_snippets()
        self.hanning_snippets_td()
        # Process

        # Return
        print("Iteration finished!")
        self.sendback_queue.put(self.ref_in_snippet_td)
        self.sendback_queue.put(self.meas_in_snippet_td)

        # Finishing this run() function triggers any GUI listeners for the self.finished() flag

    def record_snippets(self):
        """
        Uses _record to record the two streams in parallel.
        Exports to wav for reference.
        """
        file_names = ["REF_IN_REC.wav", "MEAS_IN_REC.wav"]

        print("Recording {0}s from ref_in_buffer_queue ({1} export={2})..."
              .format(round(SNIPPET_LENGTH / RATE, 2), file_names[0], EXPORT_WAV))
        print("Recording {0}s from meas_in_buffer_queue ({1} export={2})..."
              .format(round(SNIPPET_LENGTH / RATE, 2), file_names[1], EXPORT_WAV))

        # Use queues and threading here to ensure we capture identical parts of the song
        ref_in_rec_queue = queue.Queue(maxsize=1)
        meas_in_rec_queue = queue.Queue(maxsize=1)

        # To ensure synchronised recording
        ref_in_start_flag = threading.Event()
        meas_in_start_flag = threading.Event()

        p_ref = threading.Thread(target=_record, args=(self.ref_in_buffer_queue, ref_in_rec_queue,
                                                       SNIPPET_LENGTH, ref_in_start_flag))
        p_meas = threading.Thread(target=_record, args=(self.meas_in_buffer_queue, meas_in_rec_queue,
                                                        SNIPPET_LENGTH, meas_in_start_flag))

        p_ref.start()
        p_meas.start()

        # To ensure synchronised recording
        time.sleep(0.2)
        ref_in_start_flag.set()
        time.sleep(0.174)  # This value was empirically determined
        meas_in_start_flag.set()

        p_ref.join()
        p_meas.join()

        ref_in_rec = ref_in_rec_queue.get()
        meas_in_rec = meas_in_rec_queue.get()

        if EXPORT_WAV:
            from scipy.io.wavfile import write
            write(file_names[0], RATE, ref_in_rec)
            write(file_names[1], RATE, meas_in_rec)

        print("Recorded {0}s from ref_in_buffer_queue to REF_IN_REC ({1} export={2})"
              .format(round(SNIPPET_LENGTH / RATE, 2), file_names[0], EXPORT_WAV))
        print("Recorded {0}s from meas_in_buffer_queue to MEAS_IN_BFR ({1} export={2})"
              .format(round(SNIPPET_LENGTH / RATE, 2), file_names[1], EXPORT_WAV))

        x = np.arange(0, SNIPPET_LENGTH/RATE, 1 / RATE, dtype=NP_FORMAT)
        self.ref_in_snippet_td = [x, ref_in_rec]
        self.meas_in_snippet_td = [x, meas_in_rec]

    def hanning_snippets_td(self):
        """
        Applies a Hanning envelope to the reference and measurement snippets.
        Used to fulfill FFT requirement that time-domain signal needs to be periodic (zeros at beginning and end).
        Note: The Hanning function reduces the power contaied in the signal by 1/2.
              Compensate for this by multiplying the signal by 2:
              http://www.azimadli.com/vibman/thehanningwindow.htm
        """
        print("Applying Hanning window to ref_in and meas_in time domain snippets...")
        window = np.hanning(len(self.ref_in_snippet_td[1]))
        self.ref_in_snippet_td[1] = 2 * window * self.ref_in_snippet_td[1]
        self.meas_in_snippet_td[1] = 2 * window * self.meas_in_snippet_td[1]

    def ft_snippets(self):
        """
        Function to calculate and append absolute value of Fourier Transform for each snippet to the list
        self.snippets_fd, where self.snippets_fd = [(x1, y1), (x2, y2), ... , (xn, yn)]

        This function calculates the magnitude and performs normalisation operations to preserve magnitude between the
        time and frequency domains.

        The Fourier magnitudes are converted to decibels [dB] = 20 * log10(mag/ref), where ref = 1
        """
        print("Fourier transforming ref_in and meas_in snippets...")

        # Transform both snippets
        def transform(x_td, y_td):
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

        self.ref_in_snippet_fd = transform(*self.ref_in_snippet_td)
        self.meas_in_snippet_fd = transform(*self.meas_in_snippet_td)

        # Convert to dB and take magnitude
        self.snippets_fd = [(x_fd, convert_to_dbfs(y_fd)) for x_fd, y_fd in self.snippets_fd]

    def mask_snippets_fd(self):
        """
        Masks the data to return only within the desired frequency range (in Hz)
        F_LIMITS = (lo_lim, hi_lim)
        """
        print("Masking {0} frequency domain snippets between {1}Hz and {2}Hz..."
              .format(len(self.snippets_fd), F_LIMITS[0], F_LIMITS[1]))

        temp = list()
        for x, y in self.snippets_fd:
            mask = np.logical_and(F_LIMITS[0] < x, x < F_LIMITS[1])
            temp.append((x[mask], y[mask]))
        self.snippets_fd = temp


class Flag(object):
    """
    A simple boolean flag object to pass between threads
    """
    def __init__(self, init_state):
        self.state = init_state

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state


def _record(bfr_queue, rec_queue, rec_length, start_flag):
    """
    Records into a numpy array of rec_length from the queue object bfr_queue.
    Takes values from the buffer and puts them sequentially in the numpy array.
    Puts the resulting recording in rec_queue.
    """
    time.sleep(0.2)  # Wait to ensure buffer is full before beginning recording
    assert rec_length % BUFFER == 0, \
        "The recording array length needs to be an integer multiple of the buffer length"

    num_iters = int(rec_length / BUFFER)
    rec = np.zeros(rec_length, dtype=NP_FORMAT)

    start_flag.wait()

    for i in range(num_iters):
        bfr = bfr_queue.get()  # This waits until the buffer is full, then fetches it
        rec[i * BUFFER:(i + 1) * BUFFER] = bfr[:]  # Record the audio to the array

    rec_queue.put(rec)


def convert_to_dbfs(y):
    """
    Converts the input array y into dBFS units using
    [dB] = 20 * log10(mag/ref), where ref = 1
    """
    return 20 * np.log10(np.abs(y))


# def main_stream(REF_IN_BFR, MEAS_OUT_BFR, CHAIN_SETTINGS, MAIN_STREAM_ENABLED):
#     """
#     Spawned as a separate process.
#     Handles the reference in and measurement out audio streams.
#     Records to shared array REF_IN_BFR and MEAS_OUT_BFR.
#     """
#     # Initialise objects
#     p = pyaudio.PyAudio()
#     # MAIN_STREAM_ENABLED.value = 1
#     # chain = FilterChain(CHAIN_SETTINGS)
#
#     def callback(in_bytes, frame_count, time_info, flag):
#         """
#         Callback function for the ref_in/meas_out stream
#         """
#         audio_data = np.fromstring(in_bytes, dtype=NP_FORMAT)   # Convert audio data in bytes to an array.
#         audio_data = np.reshape(audio_data, (BUFFER, 2))        # Reshape array to two channels.
#         audio_data = np.average(audio_data, axis=1)             # Convert audio data to mono by averaging channels.
#         REF_IN_BFR[:] = audio_data[:]                           # Write ref_in data to shared array.
#
#         if CHAIN_SETTINGS[0] != 0:                              # If the EQ settings have changed,
#             chain.set_filter_params(CHAIN_SETTINGS)             # apply the new settings
#             CHAIN_SETTINGS[0] = 0                               # Reset the flag
#             print("\nFilter settings updated!\n{0}"
#                   .format(chain.get_settings()))
#
#         audio_data = chain.filter_signal(audio_data)            # Filter/process data
#
#         MEAS_OUT_BFR[:] = audio_data[:]                         # Write meas_out data to shared array.
#         audio_data = np.repeat(audio_data, 2)                   # Convert to 2-channel audio for compatib. with stream
#         out_bytes = audio_data.astype(NP_FORMAT).tostring()     # Convert audio data back to bytes and return
#         return out_bytes, pyaudio.paContinue
#
#     # Initialise stream
#     stream = p.open(format=PA_FORMAT,
#                     frames_per_buffer=BUFFER,
#                     rate=RATE,
#                     channels=2,
#                     input=True,
#                     output=True,
#                     input_device_index=2,
#                     output_device_index=1,
#                     stream_callback=callback)
#
#     print("\nMain stream started!")
#     print("Available audio devices by index:")
#     for i in range(p.get_device_count()):
#         print(i, p.get_device_info_by_index(i)['name'])
#     print(chain.get_settings())
#
#     while True:
#         time.sleep(0.1)
#         if PROGRAM_ALIVE.value == 0:
#             print("\nMain stream shut down!")
#             break
#         elif stream.is_active() and MAIN_STREAM_ENABLED.value == 1:
#             pass
#         elif stream.is_active() and MAIN_STREAM_ENABLED.value == 0:
#             print("\nMain stream paused!")
#             stream.stop_stream()
#         elif not stream.is_active() and MAIN_STREAM_ENABLED.value == 1:
#             print("\nMain stream started!")
#             stream.start_stream()
#         elif not stream.is_active() and MAIN_STREAM_ENABLED.value == 0:
#             pass
#
#     stream.close()
#     p.terminate()


# def meas_stream(MEAS_IN_BFR):
#     """
#     Spawned as a separate process.
#     Handles the measurement in stream.
#     Records to shared array MEAS_IN_BFR.
#     """
#     p = pyaudio.PyAudio()
#     MEAS_STREAM_ENABLED.value = 1
#
#     def callback(in_bytes, frame_count, time_info, flag):
#         """
#         Callback function for the meas_in stream
#         """
#         audio_data = np.fromstring(in_bytes, dtype=NP_FORMAT)   # Convert audio data in bytes to an array.
#         MEAS_IN_BFR[:] = audio_data[:]                          # Write meas_in data to shared array.
#         return in_bytes, pyaudio.paContinue
#
#     stream = p.open(format=PA_FORMAT,
#                     frames_per_buffer=BUFFER,
#                     rate=RATE,
#                     channels=1,
#                     input=True,
#                     input_device_index=0,
#                     stream_callback=callback)
#
#     print("\nMeas stream started!")
#
#     while True:
#         time.sleep(0.1)
#         if PROGRAM_ALIVE.value == 0:
#             print("\nMeas stream shut down!")
#             break
#         elif stream.is_active() and MEAS_STREAM_ENABLED.value == 1:
#             pass
#         elif stream.is_active() and MEAS_STREAM_ENABLED.value == 0:
#             print("\nMeas stream paused!")
#             stream.stop_stream()
#         elif not stream.is_active() and MEAS_STREAM_ENABLED.value == 1:
#             print("\nMeas stream started!")
#             stream.start_stream()
#         elif not stream.is_active() and MEAS_STREAM_ENABLED.value == 0:
#             pass
#
#     stream.close()
#     p.terminate()

#
# def toggle_main_stream():
#     if MAIN_STREAM_ENABLED.value == 1:
#         MAIN_STREAM_ENABLED.value = 0
#     elif MAIN_STREAM_ENABLED.value == 0:
#         MAIN_STREAM_ENABLED.value = 1
#
#
# def toggle_meas_stream():
#     if MEAS_STREAM_ENABLED.value == 1:
#         MEAS_STREAM_ENABLED.value = 0
#     elif MEAS_STREAM_ENABLED.value == 0:
#         MEAS_STREAM_ENABLED.value = 1

