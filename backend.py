import time
from params import *
from scipy import signal
import threading
from PyQt5 import QtCore
import queue
import itertools
import random


class QuietMeasurementException(Exception):
    """
    Raised when the measurement is quieter than the background in any part of the frequency spectrum.
    Technically this shows as NaNs present in the array of measurement values.
    Used to postpone the algorithm until a sufficiently loud measurement is obtained.
    """
    pass


class NoBackgroundException(Exception):
    """
    Raised when algorithm is run without first taking a background measurement.
    """
    pass


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


class PeakFilter(object):
    count = 0

    def __init__(self, fc, gain, q):
        """
        Filter object to be used in FilterChain

        :param fc: The center frequency (in Hz) of the filter
        :param gain: The gain (in dB) of the filter
        :param q: The Q-factor of the filter
        """
        self.set_params(fc, gain, q)

        self.id = PeakFilter.count
        PeakFilter.count += 1

    def set_params(self, fc, gain, q):
        """
        Used by __init__() to set parameters of each filter
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
        return "id={0} fc={1}, g={2}, q={3}".format(self.id, self.fc, self.gain, self.q)

    def get_tf(self):
        """
        Get the (complex) Transfer Function of the filter.
        Needs to be converted to dB using 20*log(abs(H))
        """
        w_f, h = signal.freqz(*self.get_b_a())
        f = w_f * RATE / (2 * np.pi)
        return f, h


class FilterChain(object):
    count = 0
    
    def __init__(self, *filters):
        """
        *filters: Arbitrary number of filter objects, e.g. PeakFilter
        """
        self.filters = filters

        self.stf = None
        # TODO this should be none, only for debugging
        # self.ms = int(np.random.random()*1000)
        self.ms = None

        # Calculate initial conditions
        self.zi = [signal.lfilter_zi(*filt.get_b_a()) for filt in self.filters]

        self.id = FilterChain.count
        FilterChain.count += 1

    def filter_signal(self, data_in):
        """
        Filters the current buffer and updates initial conditions for all filters in the chain accordingly
        """
        data_out = data_in

        # Reduce the amplitude of the signal an equivalent amount to the high gain limit for the filters.
        # This prevents clipping.
        data_out *= 10 ** (ATTENUATE_OUTPUT / 20)

        # Filter data and update all initial conditions for the buffer of data
        for i, filt in enumerate(self.filters):
            data_out, self.zi[i] = signal.lfilter(*filt.get_b_a(), data_out, zi=self.zi[i])
        return data_out

    def set_filter_params(self, i, fc, gain, q):
        """
        Updates the filter of index i with new settings fc, gain and q.
        """
        self.filters[i].set_params(fc, gain, q)

    def set_stf_ms(self, stf, ms):
        self.stf = stf
        self.ms = ms

    def get_filters(self):
        return self.filters[:]

    def apply_chain_state(self, chain):
        """
        Takes in another chain objects and copies its settings
        """
        self.filters = chain.get_filters()

        # Calculate initial conditions
        # self.zi = [signal.lfilter_zi(*filt.get_b_a()) for filt in self.filters]

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
        for filt in self.filters:
            out += "\n{0}".format(filt.get_settings())
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

    def get_stf_tf(self):
        return self.stf[:], self.ms


class Population(object):
    """
    Population for evolutionary algorithm.
    Initialisation -> Evaluation -> Terminate? -> Selection -> Variation
    """

    def __init__(self, initial_ms):
        """
        initial_ms: Initial ms value without any filters applied to use as a
                     termination criterion in the first iteration.

        Uses global parameters:
        POP_SIZE      The size of the population.
        NUM_FILTERS   The number of PeakFilter objects in each chain (population member).
        F_LIMITS      Tuples of the form (lo, hi). Indicates the limits of the
        GAIN_LIMITS   corresponding parameter to apply.
        Q_LIMITS
        """
        self.population = list()  # Will contain FilterChain objects
        # self.avg_ms_list = [initial_ms]
        self.best_ms_list = [initial_ms]
        self.generate_initial_population()

    def generate_initial_population(self):
        """
        Populate self.population with randomly generated members.
        """
        for i in range(POP_SIZE):
            filters = []
            for j in range(NUM_FILTERS):
                filters.append(PeakFilter(*self.random_filter_params()))
            self.population.append(FilterChain(*filters))

    @staticmethod
    def random_filter_params():
        """
        Generates a random set of filter parameters: fc, gain, q
        Respects the limits set in params.py
        Biases fc towards F_MODE, being the mode of a triangular distribution
        """
        assert F_LIMITS[0] > 0, "The lower frequency limit must be positive"
        assert GAIN_LIMITS[1] >= 0, "The high gain limit must be non-negative"

        # For random frequency (log distribution between f_lo, f_hi)
        # Generates a triangular distribution in log scale with a peak at mode_f
        rand_mode = np.log2(F_MODE / F_LIMITS[0]) / np.log2(F_LIMITS[1] / F_LIMITS[0])
        rand = np.random.triangular(0, rand_mode, 1)

        alpha = rand * np.log2(F_LIMITS[1] / F_LIMITS[0])  # Generate the exponent
        fc = F_LIMITS[0] * 2 ** alpha

        # Linear distribution
        gain = np.random.random() * (GAIN_LIMITS[1] - GAIN_LIMITS[0]) + GAIN_LIMITS[0]

        # Linear distribution
        q = np.random.random() * (Q_LIMITS[1] - Q_LIMITS[0]) + Q_LIMITS[0]

        return fc, gain, q

    def get_population(self):
        return self.population[:]

    def calculate_new_population(self):
        """
        Termination if performing worse than the last element in self.avg_ms_list
        Generate the same number of new solutions as the number of solutions terminated
        Pick random filters from the top performers
        """
        print("Calculating new population...")
        
        # Sort population by the chain's MS value
        self.population.sort(key=lambda chain: chain.ms)
        # Determine which chains get promoted
        num_promoted = int(POP_SIZE * PROP_PROMOTED)
        promoted = self.population[:num_promoted]

        print("Num promoted: {0} ({1}% of {2})".format(num_promoted, 100 * PROP_PROMOTED, POP_SIZE))

        # Generate a filter pool of all filters from the promoted chains
        filter_pool = list()
        for chain in promoted:
            filter_pool.extend(chain.filters)
        
        # Append random filters into filter pool for randomness/mutation
        # TODO code mutation, not purely random
        num_rnd = int(len(filter_pool) * PROP_RND)
        for _ in range(num_rnd):
            filter_pool.append(PeakFilter(*self.random_filter_params()))
        
        print("Filter pool: {0} filters, {1} from random ({2}% of {3})"
              .format(len(filter_pool), num_rnd, 100 * PROP_RND, len(filter_pool) - num_rnd))
        
        # Create the new filter chains
        num_new_chains = POP_SIZE - num_promoted
        new_chains = list()
        for _ in range(num_new_chains):
            chain = FilterChain(*random.sample(filter_pool, NUM_FILTERS))
            new_chains.append(chain)
            
        print("Num new chains: {0} ({1} - {2})".format(num_new_chains, POP_SIZE, num_promoted))

        # Save the new population!
        # Do a random shuffle to remove bias of promoted and new_chains being
        # applied in different parts of the song. Shouldn't matter but just in case.
        # Lastly, reset the self.population list. We will create the new one.
        self.population = promoted + new_chains
        
        print()
        for chain in self.population:
            for filt in chain.filters:
                print(filt.id, end="\t")
            print()
            
        random.shuffle(self.population)
        
        # # Generate new chains by cross-breeding
        # num_new_chains = POP_SIZE - len(promoted)
        # new_chains = list()
        # for _ in range(num_new_chains):
        #     chain = FilterChain(*random.sample(filter_pool, NUM_FILTERS))
        #     new_chains.append(chain)
        #
        #
        # # Mutate all chains over all parameters (?)
        # # TODO
        # for chain in new_population:
        #     for filt in chain.filters:
        #         for param in [filt.fc, filt.gain, filt.q]:
        #             if np.random.random() < PROB_MUTATION:
        #                 pass
    
    def save_best_ms(self):
        best_chain = sorted(self.population, key=lambda x: x.ms)[0]
        self.best_ms_list.append(best_chain.ms)
    
    # def calculate_avg_ms(self):
    #     """
    #     Append to self.avg_ms_list. Gives a measure of how the correction is performing over time.
    #     """
    #     avg = sum([chain.ms for chain in self.population]) / len(self.population)
    #     self.avg_ms_list.append(avg)
    #
    #     for chain in self.population:
    #         print(chain.ms)
    #         for filt in chain.filters:
    #             print(filt.id, end=" ")
    #         print()
    #         print()
    #
    #     print("Current average MS: {0}".format(avg), end="\n")


class MainStream(QtCore.QThread):
    """
    This thread always runs continuously in the background (daemon).
    Handles two streams of data:
    * Reference in: The raw system audio (music) that is being played. Converts it to mono.
    * Measurement out: The reference in data with the applied filter chain. This is what is being played out in the
                       speakers
    """
    def __init__(self, ref_in_buffer, meas_out_buffer, chain, bypass_chain, paused, shutting_down, main_sync_event):
        super().__init__()
        self.ref_in_buffer = ref_in_buffer
        self.meas_out_buffer = meas_out_buffer

        self.chain = chain
        self.bypass_chain = bypass_chain
        self.paused = paused
        self.shutting_down = shutting_down
        self.main_sync_event = main_sync_event

    def run(self):
        p = pyaudio.PyAudio()
        # Initialise stream
        stream = p.open(format=PA_FORMAT,
                        frames_per_buffer=BUFFER,
                        rate=RATE,
                        channels=2,
                        input=True,
                        output=True,
                        input_device_index=REF_IN_INDEX,
                        output_device_index=MEAS_OUT_INDEX,
                        stream_callback=self.callback)

        print("Available audio devices by index:")
        for i in range(p.get_device_count()):
            print(i, p.get_device_info_by_index(i)['name'])
        print("\nMain stream started (in: {0}, out: {1})!".format(REF_IN_INDEX, MEAS_OUT_INDEX))

        while not self.shutting_down.get_state():
            if stream.is_active() and not self.paused.get_state():
                pass
            elif stream.is_active() and self.paused.get_state():
                print("\nMain stream paused!")
                stream.stop_stream()
            elif not stream.is_active() and not self.paused.get_state():
                print("\nMain stream started!")
                stream.start_stream()
            elif not stream.is_active() and self.paused.get_state():
                pass
            time.sleep(0.1)

        print("\nMain stream shut down!")
        stream.close()
        p.terminate()

    def callback(self, in_bytes, frame_count, time_info, flag):
        """
        Callback function for the ref_in/meas_out stream
        """
        # print("main: {0}".format(time_info))
        in_data = np.fromstring(in_bytes, dtype=NP_FORMAT)   # Convert audio data in bytes to an array.
        in_data = np.reshape(in_data, (BUFFER, 2))        # Reshape array to two channels.
        in_data = np.average(in_data, axis=1)             # Convert audio data to mono by averaging channels.

        # Write ref_in data to ref_in_buffer.
        self.ref_in_buffer[:] = in_data[:]

        # Filter/process data
        if not self.bypass_chain.get_state():
            out_data = self.chain.filter_signal(in_data)
        else:
            out_data = in_data

        # Write meas_out data to shared queue.
        # TODO Is it even necessary to write this data to a shared array??
        self.meas_out_buffer[:] = out_data[:]
        # assert (out_data <= 1).all(), "Output signal clipped! Max: {0}" \
        #                               .format(np.max(out_data))
        if (out_data > 1).any():
            print("WARNING: Output signal clipped! Max: {0}".format(np.max(out_data)))
            print(out_data[out_data.argmax() - 5:out_data.argmax() + 6])
            print()

        out_data = np.repeat(out_data, 2)                   # Convert to 2-channel audio for compatib. with stream
        out_bytes = out_data.astype(NP_FORMAT).tostring()     # Convert audio data back to bytes and return

        self.main_sync_event.set()  # Sets the flag for synchronised recording: buffer ready for collection

        return out_bytes, pyaudio.paContinue


class MeasStream(QtCore.QThread):
    """
    This thread always runs continuously in the background (daemon).
    Handles the measurement in data stream, i.e. what is being captured by the microphone.
    """
    def __init__(self, meas_in_buffer, shutting_down, meas_sync_event):
        super().__init__()
        # self.program_shutdown = program_shutdown
        self.meas_in_buffer = meas_in_buffer
        self.shutting_down = shutting_down
        self.meas_sync_event = meas_sync_event

    def run(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=PA_FORMAT,
                        frames_per_buffer=BUFFER,
                        rate=RATE,
                        channels=1,
                        input=True,
                        input_device_index=MEAS_IN_INDEX,
                        stream_callback=self.callback)

        print("\nMeas stream started (in: {0})!".format(MEAS_IN_INDEX))

        while not self.shutting_down.get_state():
            time.sleep(0.1)

        print("\nMeas stream shut down!")
        stream.close()
        p.terminate()

    def callback(self, in_bytes, frame_count, time_info, flag):
        """
        Callback function for the meas_in stream
        """
        # print("meas: {0}".format(time_info))
        audio_data = np.fromstring(in_bytes, dtype=NP_FORMAT)  # Convert audio data in bytes to an array.

        # Write meas_in data to shared queue.
        self.meas_in_buffer[:] = audio_data

        self.meas_sync_event.set()  # Sets the flag for synchronised recording: buffer ready for collection
        return in_bytes, pyaudio.paContinue


class BackgroundModel(QtCore.QThread):
    def __init__(self, sendback_queue, meas_in_buffer, meas_sync_event):
        super().__init__()
        self.sendback_queue = sendback_queue
        self.meas_in_buffer = meas_in_buffer
        self.meas_sync_event = meas_sync_event

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
        file_name = "BACKGROUND_REC.wav"
        print("Recording {0}s from ref_in_buffer ({1} export={2})..."
              .format(round(BACKGROUND_LENGTH / RATE, 2), file_name, EXPORT_WAV))

        self.background_rec = np.zeros(BACKGROUND_LENGTH, dtype=NP_FORMAT)
        start_event = threading.Event()
        start_event.set()
        _record(self.meas_in_buffer, self.background_rec, self.meas_sync_event, start_event)

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

        The (complex) FFT is converted to magnitude [dBFS] using convert_to_dbfs().

        The resulting arrays are: x [Hz] and y [dBFS].
        """
        print("Fourier transforming {0} snippets...".format(len(self.snippets_td)))

        # Transform all snippets
        for x_td, y_td in self.snippets_td:
            x_fd, y_fd = fourier_transform(y_td)
            self.snippets_fd.append((x_fd, y_fd))

        # Take magnitude (abs) and convert to dB
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
        y_model = np.array(self.snippets_fd)[:, 1].mean(axis=0)

        self.bg_model_fd = (x_model, y_model)

    def smoothen_bg_model_fd(self):
        """
        Smoothens the background model
        """
        print("Smoothening the background model with {0} octave bins...".format(round(OCT_FRAC, 4)))
        x_model, y_model = smoothen_data_fd(*self.bg_model_fd)
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
            self.snippets_fd_smooth.append(smoothen_data_fd(x, y))


class LatencyCalibration(QtCore.QThread):
    """
    Used to determine the MEAS_REF_LATENCY value to synchronise the ref_in and meas_in streams.
    Currently not in use.
    """
    def __init__(self, sendback_queue, ref_in_buffer, meas_in_buffer, main_sync_event, meas_sync_event):
        super().__init__()
        self.sendback_queue = sendback_queue
        self.ref_in_buffer = ref_in_buffer
        self.meas_in_buffer = meas_in_buffer
        self.main_sync_event = main_sync_event
        self.meas_sync_event = meas_sync_event

    def run(self):
        """
        Records the two streams and send the result back to the GUI for plotting.
        """
        print("\nTaking latency measurements:")
        print("Latency measurement: Recording {0}s from ref_in_buffer..."
              .format(round(LATENCY_MEASUREMENT_LENGTH / RATE, 2)))
        print("Latency measurement: Recording {0}s from meas_in_buffer..."
              .format(round(LATENCY_MEASUREMENT_LENGTH / RATE, 2)))

        # Arrays in which to record
        ref_in_rec = np.zeros(LATENCY_MEASUREMENT_LENGTH, dtype=NP_FORMAT)
        meas_in_rec = np.zeros(LATENCY_MEASUREMENT_LENGTH, dtype=NP_FORMAT)

        # Use threading here to ensure we capture identical parts of the song
        # Use events to control exactly when each recording starts
        ref_in_start_event = threading.Event()
        meas_in_start_event = threading.Event()

        ref_thread = threading.Thread(target=_record, args=(self.ref_in_buffer, ref_in_rec, self.main_sync_event,
                                                            ref_in_start_event))
        meas_thread = threading.Thread(target=_record, args=(self.meas_in_buffer, meas_in_rec, self.meas_sync_event,
                                                             meas_in_start_event))

        ref_thread.start()
        meas_thread.start()

        # To ensure synchronised recording
        time.sleep(0.5)
        ref_in_start_event.set()
        time.sleep(MEAS_REF_LATENCY)
        meas_in_start_event.set()

        ref_thread.join()
        meas_thread.join()

        # Return
        print("Latency measurement finished!")
        x = np.arange(0, LATENCY_MEASUREMENT_LENGTH / RATE, 1 / RATE, dtype=NP_FORMAT)
        self.sendback_queue.put([x, ref_in_rec])
        self.sendback_queue.put([x, meas_in_rec])

        # Finishing this run() function triggers any GUI listeners for the self.finished() flag


class Algorithm(QtCore.QThread):
    """
    The evolutionary algorithm that runs in the background and continuously sends back data for plotting.
    """
    def __init__(self, sendback_queue, ref_in_buffer, meas_in_buffer, main_sync_event, meas_sync_event, bg_model,
                 live_chain, update_filter_ax_signal, update_stf_ax_signal, algorithm_running):
        super().__init__()
        self.sendback_queue = sendback_queue
        self.ref_in_buffer = ref_in_buffer
        self.meas_in_buffer = meas_in_buffer
        self.main_sync_event = main_sync_event
        self.meas_sync_event = meas_sync_event
        self.bg_model = bg_model
        self.live_chain = live_chain

        self.population = None
        self.update_filter_ax_signal = update_filter_ax_signal
        self.update_stf_ax_signal = update_stf_ax_signal
        self.algorithm_running = algorithm_running
        # self.stf = None
        # self.ms = None

    def run(self):
        # Measure initial STF
        print("Measuring initial STF, recording {0}s...".format(round(SNIPPET_LENGTH / RATE, 2)))
        initial_stf, initial_ms = self.measure_stf_ms(verbose=False)
        print("Calculated initial MS: {0}".format(initial_ms))

        # Initialise the population
        self.population = Population(initial_ms)
        iteration = 1
        # Iterate through every chain in the population
        while self.algorithm_running.get_state():
            print("\n-------- ITERATION {0} ------------------------".format(iteration))
            for i, chain in enumerate(self.population.get_population()):
                print("\nApplying chain {0}/{1}, id={2}".format(i + 1, len(self.population.get_population()), chain.id))
                # print(chain.get_chain_settings())
                self.update_filter_ax_signal.emit()  # Trigger updating the filter chain plot

                # Apply the current chain
                self.live_chain.apply_chain_state(chain)

                # Measure STF and MS, and save as instance attribute of the chain
                print("Measuring STF, recording {0}s...".format(round(SNIPPET_LENGTH / RATE, 2)))
                stf, ms = self.measure_stf_ms(verbose=False)
                print("Calculated MS: {0}".format(ms))
                chain.set_stf_ms(stf, ms)

            best_chain = sorted(self.population.get_population(), key=lambda x: x.ms)[0]
            print("\nBest MS: {0} (id={1})".format(best_chain.ms, best_chain.id))
            
            # self.population.calculate_avg_ms()
            self.population.save_best_ms()
            self.population.calculate_new_population()
            
            # Send back stuff
            self.sendback_queue.put(initial_stf)
            self.sendback_queue.put(initial_ms)
            self.sendback_queue.put(best_chain.stf)
            self.sendback_queue.put(best_chain.ms)
            self.sendback_queue.put(self.population.best_ms_list)
            self.update_stf_ax_signal.emit()

            iteration += 1

        self.live_chain.apply_chain_state(best_chain)

        # Finishing this run() function triggers any GUI listeners for the self.finished() flag

    def measure_stf_ms(self, verbose=True):
        """
        Measures the STF (System Transfer Function) and its associated Mean-Squared value.
        Returns: stf (x, y arrays), ms (float)
        """
        if verbose: print("\nMeasuring STF:")

        # Keep iterating until a sufficiently loud measurement is obtained
        while True:
            # Record
            ref_in_snippet_td, meas_in_snippet_td = self.record_snippets(verbose)

            # Calculate the STF
            ref_in_snippet_td, meas_in_snippet_td = self.hanning_snippets_td(ref_in_snippet_td, meas_in_snippet_td, verbose)
            ref_in_snippet_fd, meas_in_snippet_fd = self.ft_snippets(ref_in_snippet_td, meas_in_snippet_td, verbose)
            ref_in_snippet_fd, meas_in_snippet_fd = self.mask_snippets_fd(ref_in_snippet_fd, meas_in_snippet_fd, verbose)
            ref_in_snippet_fd_smooth, meas_in_snippet_fd_smooth = self.smoothen_snippets_fd(ref_in_snippet_fd,
                                                                                            meas_in_snippet_fd, verbose)
            try:
                meas_in_snippet_fd_smooth = self.subtract_bg_from_meas(self.bg_model, meas_in_snippet_fd_smooth, verbose)
            except QuietMeasurementException:
                # If nans, iterate through while loop once again
                print("\nMeasurement too quiet, measuring STF again...")
                pass
            except NoBackgroundException:
                print("No background model generated, continuing...")
                break
            else:
                # If no nans, exit the while loop
                break

        ref_in_snippet_fd_smooth, meas_in_snippet_fd_smooth = self.normalise_snippets_fd(ref_in_snippet_fd_smooth,
                                                                                         meas_in_snippet_fd_smooth, verbose)
        stf = self.calculate_stf(ref_in_snippet_fd_smooth, meas_in_snippet_fd_smooth, verbose)
        ms = self.calculate_ms(stf, verbose)

        # Return the calculated STF and MS value
        if verbose: print("STF and MS calculated!")

        return stf, ms

    def record_snippets(self, verbose=True):
        """
        Uses _record to record the two streams in parallel.
        Exports to wav for reference.
        """
        file_names = ["REF_IN_REC.wav", "MEAS_IN_REC.wav"]

        if verbose:
            print("Recording {0}s from ref_in_buffer ({1} export={2})..."
                  .format(round(SNIPPET_LENGTH / RATE, 2), file_names[0], EXPORT_WAV))
            print("Recording {0}s from meas_in_buffer ({1} export={2})..."
                  .format(round(SNIPPET_LENGTH / RATE, 2), file_names[1], EXPORT_WAV))

        # Arrays in which to record
        ref_in_rec = np.zeros(SNIPPET_LENGTH, dtype=NP_FORMAT)
        meas_in_rec = np.zeros(SNIPPET_LENGTH, dtype=NP_FORMAT)

        # Use threading here to ensure reference and measurement capture identical parts of the song
        # Use events to control exactly when each recording starts (there's a slight delay when spawning threads)
        ref_in_start_event = threading.Event()
        meas_in_start_event = threading.Event()

        ref_thread = threading.Thread(target=_record, args=(self.ref_in_buffer, ref_in_rec, self.main_sync_event,
                                                            ref_in_start_event))
        meas_thread = threading.Thread(target=_record, args=(self.meas_in_buffer, meas_in_rec, self.meas_sync_event,
                                                             meas_in_start_event))

        ref_thread.start()
        meas_thread.start()

        # To ensure synchronised recording
        time.sleep(1)
        ref_in_start_event.set()
        time.sleep(MEAS_REF_LATENCY)
        meas_in_start_event.set()

        ref_thread.join()
        meas_thread.join()

        if EXPORT_WAV:
            from scipy.io.wavfile import write
            write(file_names[0], RATE, ref_in_rec)
            write(file_names[1], RATE, meas_in_rec)

        if verbose:
            print("Recorded {0}s from ref_in_buffer to REF_IN_REC ({1} export={2})"
                  .format(round(SNIPPET_LENGTH / RATE, 2), file_names[0], EXPORT_WAV))
            print("Recorded {0}s from meas_in_buffer to MEAS_IN_BFR ({1} export={2})"
                  .format(round(SNIPPET_LENGTH / RATE, 2), file_names[1], EXPORT_WAV))

        x = np.arange(0, SNIPPET_LENGTH/RATE, 1 / RATE, dtype=NP_FORMAT)
        return [x, ref_in_rec], [x, meas_in_rec]

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def subtract_bg_from_meas(bg_model, meas_in_snippet_fd_smooth, verbose=True):
        """
        Subtracts the pre-generated background model from the meas_in snippet.
        Uses the fancy formula we derived (see lab book page 36)
        """
        if bg_model is None:
            raise NoBackgroundException

        if verbose: print("Subtracting the background model from the meas_in frequency domain snippet...")
        assert (meas_in_snippet_fd_smooth[0] == bg_model[0]).all(), \
            "Cannot subtract background from meas_in: Their frequency steps (x-values) are not identical!"

        y_fd_bg = bg_model[1]
        x_fd_meas, y_fd_meas = meas_in_snippet_fd_smooth

        # If background is louder than measurement, raise exception
        if (y_fd_meas - y_fd_bg <= 0).any():
            raise QuietMeasurementException

        y_fd_meas = 20 * np.log10(10 ** (y_fd_meas/20) - 10 ** (y_fd_bg/20))

        meas_in_snippet_fd_smooth = [x_fd_meas, y_fd_meas]

        return meas_in_snippet_fd_smooth

    @staticmethod
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

    @staticmethod
    def calculate_stf(ref_in_snippet_fd_smooth, meas_in_snippet_fd_smooth, verbose=True):
        """
        Calculates the STF by subtracting ref_in from meas_in.
        """
        if verbose: print("Calculating STF...")
        x_stf = ref_in_snippet_fd_smooth[0]
        y_stf = meas_in_snippet_fd_smooth[1] - ref_in_snippet_fd_smooth[1]
        stf = [x_stf, y_stf]

        return stf

    @staticmethod
    def calculate_ms(stf, verbose=True):
        """
        Calculates the value of the objective function: Mean-Squared.
        A measure of the STF curve's deviation from being flat.
        MS gives extra weight to outliers as these are extra sensitive to perceived sound.
        """
        ms = np.average(np.sum(np.square(stf[1])))
        if verbose: print("Calculated MS: {0}".format(ms))
        return ms


def _record(bfr_array, rec_array, stream_sync_event, start_event):
    """
    Records into rec_array from bfr_array.
    Takes values from the buffer and puts them sequentially in the rec array.
    stream_sync_event: acts as a clock to ensure the buffer does not get written more than once,
    but synchronises with the stream frequency.
    start_event: to ensure synchronised recording for multiple threads
    """
    time.sleep(0.5)  # Wait to ensure buffer is full before beginning recording
    assert len(rec_array) % len(bfr_array) == 0, \
        "The recording array length needs to be an integer multiple of the buffer size"

    num_iters = int(len(rec_array) / len(bfr_array))

    start_event.wait()

    for i in range(num_iters):
        stream_sync_event.wait()  # This waits until the buffer is full, then fetches it
        rec_array[i * BUFFER:(i + 1) * BUFFER] = bfr_array[:]  # Record the audio to the array
        stream_sync_event.clear()  # Clears the flag and waits for the stream to set it again


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
