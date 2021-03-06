import time
from params import *
from scipy import signal
import threading
from PyQt5 import QtCore
# import queue
import itertools
import random
from copy import deepcopy


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


class TimedEvent(threading.Event):
    """
    yeet
    
    A superclass of the treading.Event object with method to return a timestamp since the last .set()
    """
    def __init__(self):
        super().__init__()
    
    def set(self):
        super().set()
        self.last_set = time.time()
    
    def get_dt(self):
        return time.time() - self.last_set


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
        Used by __init__() to set parameters of each filter.
        """
        if (fc is None) and (gain is None) and (q is None):
            return
        
        # Save new instance variables
        if fc is not None:
            self.fc = fc
        if gain is not None:
            self.gain = gain
        if q is not None:
            self.q = q

        # Frequencies declared as fractions of sampling rate
        f_frac = self.fc / RATE
        w_frac = 2 * np.pi * f_frac

        # Declare Transfer Function in S-domain
        num = [1, 10 ** (self.gain / 20) * w_frac / self.q, w_frac ** 2]
        den = [1, w_frac / self.q, w_frac ** 2]

        # Calculate coefficients in Z-domain using Bi-linear transform
        self.b, self.a = signal.bilinear(num, den)

        bw = self.fc / self.q
        self.f1 = (-bw + np.sqrt(bw ** 2 + 4 * self.fc ** 2)) / 2
        self.f2 = bw + self.f1

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
        return "filter_id={0}, fc={1}, g={2}, q={3}".format(self.id, self.fc, self.gain, self.q)

    def get_tf(self):
        """
        Get the (complex) Transfer Function of the filter.
        Needs to be converted to dB using 20*log(abs(H))
        """
        w_f, h = signal.freqz(*self.get_b_a(), worN=4096*8)
        f = w_f * RATE / (2 * np.pi)
        return f, h
    
    def get_f1_f2(self):
        """
        Returns a tuple of the restricted band for where other filters cannot go.
        """
        return self.f1, self.f2


class FilterChain(object):
    count = 0
    
    def __init__(self, *filters, kind="r"):
        """
        *filters: Arbitrary number of filter objects, e.g. PeakFilter
        kind: can be "r", "p", "c" for random, promoted, crossover.
        """
        self.filters = filters
        self.kind = kind
        
        self.stf = None
        # TODO this should be none, only for debugging
        self.fitness = None

        # Calculate initial conditions
        self.zi = [signal.lfilter_zi(*filt.get_b_a()) for filt in self.filters]

        self.id = FilterChain.count
        FilterChain.count += 1
    
    def __eq__(self, other):
        """
        Required for removing duplicates in filter pool using set() function
        """
        return self.id == other.id
    
    def __hash__(self):
        """
        Required for removing duplicates in filter pool using set() function
        """
        return hash(("id", self.id))

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

    def set_stf_fitness(self, stf, fitness):
        self.stf = stf
        self.fitness = fitness

    def get_filters(self):
        return self.filters[:]

    def apply_chain_state(self, chain):
        """
        Takes in another chain objects and copies its settings
        """
        self.filters = chain.get_filters()
        self.id = chain.id

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
        out = "Settings of filter chain:"
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
        return self.stf[:], self.fitness


class Population(object):
    """
    Population for evolutionary algorithm.
    Initialisation -> Evaluation -> Terminate? -> Selection -> Variation
    """

    def __init__(self, initial_fitness):
        """
        initial_fitness: Initial fitness value without any filters applied to use as a
                         termination criterion in the first iteration.

        Uses global parameters:
        POP_SIZE      The size of the population.
        NUM_FILTERS   The number of PeakFilter objects in each chain (population member).
        F_LIMITS      Tuples of the form (lo, hi). Indicates the limits of the
        GAIN_LIMITS   corresponding parameter to apply.
        Q_LIMITS
        """
        self.population = list()  # Will contain FilterChain objects
        self.best_fitness_list = [initial_fitness]
        self.generate_initial_population()

    def generate_initial_population(self):
        """
        Populate self.population with randomly generated members.
        """
        for i in range(POP_SIZE):
            filters = []
            for j in range(NUM_FILTERS):
                
                # Loop until there are no filter band overlaps
                while True:
                    filter_candidate = PeakFilter(*self.random_filter_params())
                    if is_filter_allowed(filters, filter_candidate):
                        break
                filters.append(filter_candidate)
                
            self.population.append(FilterChain(*filters))

    @staticmethod
    def random_filter_params():
        """
        Generates a random set of filter parameters: fc, gain, q
        Respects the limits set in params.py
        """
        assert F_LIMITS[0] > 0, "The lower frequency limit must be positive"
        assert GAIN_LIMITS[1] >= 0, "The high gain limit must be non-negative"

        # For random frequency (log distribution between f_lo, f_hi)
        alpha_fc = np.random.random() * np.log2(F_LIMITS[1] / F_LIMITS[0])  # Generate the exponent
        fc = F_LIMITS[0] * 2 ** alpha_fc

        # Linear distribution for gain (already in log-scale)
        gain = np.random.random() * (GAIN_LIMITS[1] - GAIN_LIMITS[0]) + GAIN_LIMITS[0]

        # Log (proportionate) distribution, by nature of Q-factor
        alpha_q = np.random.random() * np.log2(Q_LIMITS[1] / Q_LIMITS[0])  # Generate the exponent
        q = Q_LIMITS[0] * 2 ** alpha_q

        return fc, gain, q

    def get_population(self):
        # deepcopy is needed because the filter chains are mutable and change between iterations
        return self.population[:]

    def calculate_new_population(self):
        """
        1. Promote
        2. Generate filter pool
        3. Add random filters
        4. Create new chains
        5. Mutate filters
        """
        print("\nCalculating new population...")
        
        # 1. Promote the best chains
        # Sort population by the chain's fitness value
        self.population.sort(key=lambda chain: chain.fitness)
        # Determine which chains get promoted
        num_promoted = int(round(POP_SIZE * PROP_PROMOTED))
        promoted = self.population[:num_promoted]
        # Set the kind of all promoted chains to be p (for promoted)
        for i in range(len(promoted)):
            promoted[i].kind = "p"

        print("Num promoted: {0} ({1}% of {2})".format(num_promoted, 100 * PROP_PROMOTED, POP_SIZE))

        # 2. Generate a filter pool of all filters from the promoted chains
        filter_pool = list()
        for chain in promoted:
            filter_pool.extend(chain.filters)
        print("Total number of filters currently in pool: {0}".format(len(filter_pool)))
        
        # REMOVED THE BELOW: IT DOES MAKE SENSE THAT A POPULAR FILTER HAS A HIGHER PROBABILIY OF BEING SELECTED
        # Remove duplicates
        # filter_pool = list(set(filter_pool))
        # print("Removed duplicates, new number of filters in pool: {0}".format(len(filter_pool)))

        # 3. Append random filters into filter pool for randomness/mutation
        num_rnd = int(round(len(filter_pool) * PROP_RND))
        for _ in range(num_rnd):
            filter_pool.append(PeakFilter(*self.random_filter_params()))
        print("Added {0} random filters ({1}% of {2})"
              .format(num_rnd, 100 * PROP_RND, len(filter_pool) - num_rnd))
        print("{0} total number of filters currently in pool".format(len(filter_pool)))

        # 4. Create the new filter chains
        num_new_chains = POP_SIZE - num_promoted
        new_chains = list()
        for i in range(num_new_chains):
            # Create the list of non overlapping filters
            new_filters = list()
            print("Chain {0}/{1}: Finding combination of filters that is not overlapping...".format(i + 1, num_new_chains))
            # Continue looping until the required number of filters for the new chain is reached.
            while len(new_filters) < NUM_FILTERS:
                crossover_attempt = 1
                filter_found = False
                # Try to add a new filter with a maximum number of attempts
                while crossover_attempt <= MAX_CROSSOVER_ATTEMPTS:
                    filter_candidate = random.choice(filter_pool)
                    if is_filter_allowed(new_filters, filter_candidate):
                        filter_found = True
                        break
                    crossover_attempt += 1
                
                # Append the filter to list if a non-overlapping one is found
                if filter_found:
                    new_filters.append(filter_candidate)
                # If a suitable filter was not found within the maximum number of attempts,
                # wipe the current attempt and try again.
                else:
                    print("Maximum number of {0} attempts reached, trying again...".format(MAX_CROSSOVER_ATTEMPTS))
                    new_filters = list()
                
            chain = FilterChain(*new_filters, kind="c")
            new_chains.append(chain)
            # print("...done!")
        
        print("Created {0} new chains from filter pool".format(num_new_chains))
        
        # 5. Mutation
        # To keep track/print progress
        self.num_fc_mut, self.num_gain_mut, self.num_q_mut = 0, 0, 0
        num_filters_mut = 0
        failed_mutations = 0
        self.mutation_table = list()
        
        for i, chain in enumerate(new_chains):
            new_filters = list()
            print("Chain {0}/{1}: Mutating... ".format(i + 1, num_new_chains))
            # Iterate through all individual filters and mutate
            for j, filt in enumerate(chain.get_filters()):
                # Returns the original and the mutated filter
                orig_filt, new_filt = self.mutate_filter(filt)
                
                mutation_attempt = 1
                mutation_succeeded = False
                while mutation_attempt <= MAX_MUTATION_ATTEMPTS:
                    # Try fitting the mutated filter
                    # Keep mutating the original until obtaining a mutated filter that fits
                    if is_filter_allowed(new_filters, new_filt):
                        mutation_succeeded = True
                        break
                    else:
                        _, new_filt = self.mutate_filter(orig_filt)
                        mutation_attempt += 1
                
                if mutation_succeeded:
                    new_filters.append(new_filt)
                    num_filters_mut += 1
                    print("- Filter {0}: Succeeded in {1} attempts!".format(j + 1, mutation_attempt))
                else:
                    new_filters.append(orig_filt)
                    failed_mutations += 1
                    print("FAILED after max {0} attempts, using original filter!".format(mutation_attempt))
            
            # Apply the new set of mutated filters
            assert len(chain.filters) == len(new_filters)
            chain.filters = tuple(new_filters)

            # # Iterate through all individual filters and mutate
            # for filt in chain.get_filters():
            #     # Returns the original and the mutated filter
            #     orig_filt, new_filt = self.mutate_filter(filt)
            #
            #     new_filters_mutated.append(new_filt)
            #     new_filters_mutated_originals.append(orig_filt)
            #     num_filters_mut += 1
            #
            # # Try fitting the mutated filters
            # for i in range(len(new_filters_mutated)):
            #     new_filt = new_filters_mutated[i]
            #     orig_filt = new_filters_mutated_originals[i]
            #     # Keep mutating the original until obtaining a mutated filter that fits
            #     while not is_filter_allowed(new_filters, new_filt):
            #         _, new_filt = self.mutate_filter(orig_filt)
            #     new_filters.append(new_filt)
            #
            # assert len(chain.filters) == len(new_filters)
            # chain.filters = tuple(new_filters)
            # chain.kind = chain_kind
            
        print("Mutation results: failed_mutations = {0}, num_filters_mut = {1}, num_fc = {2}, num_gain = {3}, num_q = {4}"
              .format(failed_mutations, num_filters_mut, self.num_fc_mut, self.num_gain_mut, self.num_q_mut))
        
        print("New population calculated!")
        
        print("\nMUTATION TABLE:")
        print("%-12s%-20s%-20s" % ("param", "old", "new"))
        print("-" * 32)
        for param, old, new in self.mutation_table:
            print("%-8s%-12f%-12f" % (param, old, new))
            
        # Save the new population!
        self.population = promoted + new_chains
        
        # print("\nKINDS NEXT ITER:")
        # for chain in self.population:
        #     print(chain.kind)
    
    def mutate_filter(self, filt):
        """
        Mutates all filter parameters.
        """
        fc, gain, q = filt.fc, filt.gain, filt.q
        
        # Mutate fc
        fc = self.mutate_fc(filt.fc)
        self.num_fc_mut += 1
        self.mutation_table.append(("fc", filt.fc, fc))

        # Mutate gain
        gain = self.mutate_gain(filt.gain)
        self.num_gain_mut += 1
        self.mutation_table.append(("gain", filt.gain, gain))

        # Mutate Q
        q = self.mutate_q(filt.q)
        self.num_q_mut += 1
        self.mutation_table.append(("Q", filt.q, q))
        
        return filt, PeakFilter(fc, gain, q)
        
    def mutate_fc(self, fc_old):
        """
        Takes an old fc value and mutates it using a Guassian with standard deviation STDEV_FC.
        Respects F_LIMITS. If the mutated value is outside of limits, return the limit.
        """
        x_oct = np.random.normal(loc=0, scale=STDEV_FC)
        
        fc_new = fc_old * 2 ** x_oct
        
        if fc_new > F_LIMITS[1]:
            fc_new = F_LIMITS[1]
        elif fc_new < F_LIMITS[0]:
            fc_new = F_LIMITS[0]
        
        return fc_new
    
    def mutate_gain(self, gain_old):
        """
        Takes an old gain value and mutates it using a Guassian with standard deviation STDEV_GAIN.
        Respects GAIN_LIMITS. If the mutated value is outside of limits, return the limit.
        """
        factor_gain = np.random.normal(loc=0, scale=STDEV_GAIN)
        gain_new = gain_old + factor_gain
        
        if gain_new > GAIN_LIMITS[1]:
            gain_new = GAIN_LIMITS[1]
        elif gain_new < GAIN_LIMITS[0]:
            gain_new = GAIN_LIMITS[0]
        
        return gain_new

    def mutate_q(self, q_old):
        """
        Takes an old q value and mutates it using a Guassian with standard deviation STDEV_Q.
        Respects Q_LIMITS. If the mutated value is outside of limits, return the limit.
        """
        x_oct = np.random.normal(loc=0, scale=STDEV_Q)

        q_new = q_old * 2 ** x_oct
        
        if q_new > Q_LIMITS[1]:
            q_new = Q_LIMITS[1]
        elif q_new < Q_LIMITS[0]:
            q_new = Q_LIMITS[0]
    
        return q_new
    
    def save_best_fitness(self):
        best_chain = sorted(self.population, key=lambda x: x.fitness)[0]
        self.best_fitness_list.append(best_chain.fitness)
    
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
    #     print("Current average fitness: {0}".format(avg), end="\n")


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
        start_event = TimedEvent()
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
        ref_in_start_event = TimedEvent()
        meas_in_start_event = TimedEvent()

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
                 live_chain, update_filter_ax_signal, collect_algorithm_queue_data_signal,
                 update_fitness_iter_ax_signal, update_stf_ax_signal, algorithm_running):
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
        self.update_fitness_iter_ax_signal = update_fitness_iter_ax_signal
        self.collect_algorithm_queue_data_signal = collect_algorithm_queue_data_signal
        self.update_stf_ax_signal = update_stf_ax_signal
        self.algorithm_running = algorithm_running

    def run(self):
        # Measure initial STF
        print("\nMeasuring initial STF, recording {0}s...".format(round(SNIPPET_LENGTH / RATE, 2)))
        initial_stf, initial_fitness = self.measure_stf_fitness(verbose=False)
        print("Calculated initial fitness: {0}".format(initial_fitness))

        # Initialise the population
        self.population = Population(initial_fitness)
        iteration = 1
        # Iterate through every chain in the population
        while self.algorithm_running.get_state():
            print("\n-------- ITERATION {0} ------------------------".format(iteration))
            for i, chain in enumerate(self.population.get_population()):
                print("\nApplying chain {0}/{1}, id={2}, kind={3}".format(i + 1, len(self.population.get_population()),
                                                                          chain.id, chain.kind))
                # print(chain.get_chain_settings())
    
                self.live_chain.apply_chain_state(chain)  # Apply the current chain
                self.update_filter_ax_signal.emit()  # Trigger updating the filter chain plot
                
                # TODO
                time.sleep(1)  # Delay to prevent VHOOFH from affecting recordings
                # Measure STF and fitness, and save as instance attribute of the chain
                print("Measuring STF, recording {0}s...".format(round(SNIPPET_LENGTH / RATE, 2)))
                stf, fitness = self.measure_stf_fitness(verbose=False)
                print("Calculated fitness: {0}".format(fitness))
                chain.set_stf_fitness(stf, fitness)
        
            best_chain = sorted(self.population.get_population(), key=lambda x: x.fitness)[0]
            print("\nIteration's best fitness: {0} (id={1}, kind={2})"
                  .format(best_chain.fitness, best_chain.id, best_chain.kind))
            self.population.save_best_fitness()
            
            # Send back stuff
            # deepcopy is needed because the filter chains are mutable and change between iterations
            self.sendback_queue.put(initial_stf)
            self.sendback_queue.put(initial_fitness)
            self.sendback_queue.put(deepcopy(best_chain.stf))
            self.sendback_queue.put(best_chain.fitness)
            self.sendback_queue.put(self.population.best_fitness_list)
            self.sendback_queue.put(deepcopy(self.population.get_population()))
            
            self.collect_algorithm_queue_data_signal.emit()  # Trigger collection of algorithm data from sendback_queue
            self.update_fitness_iter_ax_signal.emit()  # Trigger updating the algorithm progression plot
            self.update_stf_ax_signal.emit()  # Trigger updating the STF plot
            
            # To ensure the program thread has time to collect the population before the new one is calculated.
            # This potentially fixes the issue of wrong kind-labels showing up on the graph.
            # Needed because the filter chains are mutable and change between iterations
            time.sleep(0.2)
            
            self.population.calculate_new_population()
            iteration += 1

        self.live_chain.apply_chain_state(best_chain)
        self.update_filter_ax_signal.emit()  # Trigger updating the filter chain plot

        # Finishing this run() function triggers any GUI listeners for the self.finished() flag

    def measure_stf_fitness(self, verbose=True):
        """
        Measures the STF (System Transfer Function) and its associated fitness value.
        Returns: stf (x, y arrays), fitness (float)
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
        fitness = self.calculate_fitness(stf, verbose)

        # Return the calculated STF and fitness value
        if verbose: print("STF and fitness calculated!")

        return stf, fitness

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
        ref_in_start_event = TimedEvent()
        meas_in_start_event = TimedEvent()

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
        meas_in_start_event.set()  # TODO This arrangement means the latency can only be in multiples of BUFFER! Fix!

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
    def calculate_fitness(stf, verbose=True):
        """
        Calculates the value of the objective function.
        A measure of the STF curve's deviation from being flat.
        """
        fitness = np.average(np.abs(stf[1]))
        if verbose: print("Calculated fitness: {0}".format(fitness))
        return fitness


def _record(bfr_array, rec_array, stream_sync_event, start_event):
    """
    Records into rec_array from bfr_array.
    Takes values from the buffer and puts them sequentially in the rec array.
    stream_sync_event: acts as a clock to ensure the buffer does not get written more than once,
    but instead synchronises with the stream frequency.
    start_event: to ensure synchronised recording for multiple threads
    """
    time.sleep(0.5)  # Wait to ensure buffer is full before beginning recording
    # assert len(rec_array) % len(bfr_array) == 0, \
    #     "The recording array length needs to be an integer multiple of the buffer size"
    
    start_event.wait()
    # Pick a buffer sample offset with an upper limit of BUFFER.
    dt = stream_sync_event.get_dt()
    
    # Need a maximum value for sample offset, cannot look back further than a buffer
    if int(dt * RATE) > BUFFER:
        sample_offset = BUFFER
        print("WARNING: Buffer took longer than expected to update")
    else:
        sample_offset = BUFFER - int(dt * RATE)
    # print("dt * RATE = {0}".format(dt * RATE))
    
    """
    if sample_offset < 0
    means that dt * RATE > BUFFER
    means that the buffer took longer than normal to update
    """
    i = sample_offset
    
    if i > 0:
        # print(0, i)
        rec_array[:i] = bfr_array[-i:]
        # print("rec_array[:{0}]".format(i))
        stream_sync_event.clear()  # Clears the flag and waits for the stream to set it again
    while i < sample_offset + ((len(rec_array) - sample_offset) // BUFFER) * BUFFER:
        # print(i, i + BUFFER)
        stream_sync_event.wait()  # This waits until the buffer is updated, then fetches it
        # print("rec_array[{0}:{1}], len(bfr_array[:]={2}".format(i, i + BUFFER, len(bfr_array)))
        rec_array[i:i + BUFFER] = bfr_array[:]
        i += BUFFER
        stream_sync_event.clear()  # Clears the flag and waits for the stream to set it again
    if i < len(rec_array):
        # print(i, len(rec_array))
        stream_sync_event.wait()  # This waits until the buffer is full, then fetches it
        # print(len(rec_array[i:]))
        # print(len(bfr_array[:len(rec_array) - i]))
        rec_array[i:] = bfr_array[:len(rec_array) - i]
        # print("rec_array[i:]".format(i))
    
    # print("Rec array:\n{0}".format(rec_array))
    

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


def is_filter_allowed(existing_filters, new_filter):
    """
    Returns: bool (True of False)
    True if the new filter does not overlap with any filter in existing filters.
    False otherwise.
    """
    f1, f2 = new_filter.get_f1_f2()
    for filt in existing_filters:
        f1_ex, f2_ex = filt.get_f1_f2()
        
        if not (f2 < f1_ex or f1 > f2_ex):
            # print("!! REJECTED FILTER. range_existing=[{0}, {1}], range_new=[{2}, {3}]".format(f1_ex, f2_ex, f1, f2))
            return False
    # print("!! ACCEPTED FILTER")
    return True
