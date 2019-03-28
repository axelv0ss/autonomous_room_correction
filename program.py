from backend import *
from queue import Queue

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from copy import deepcopy

# np.random.seed(1)

# TODO is meas_out_buffer really needed?


class Program(QWidget):
    # Signals are required to be class variables, just a pyQT thing
    update_filter_ax_signal = QtCore.pyqtSignal()
    collect_algorithm_queue_data_signal = QtCore.pyqtSignal()
    update_stf_ax_signal = QtCore.pyqtSignal()
    update_fitness_iter_ax_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        # Connect signals to functions
        self.update_filter_ax_signal.connect(self.update_filter_ax)
        self.collect_algorithm_queue_data_signal.connect(self.collect_algorithm_queue_data)
        self.update_stf_ax_signal.connect(self.update_stf_ax)
        self.update_fitness_iter_ax_signal.connect(self.update_fitness_iter_ax)
        
        self.title = "Autonomous Room Correction  //  Rate: {0}Hz  //  Format: {1}  //  Buffer: {2}  //  " \
                     "Range: {3}Hz - {4}Hz".format(RATE, str(NP_FORMAT).split('\'')[1], BUFFER, *F_LIMITS)
        self.setWindowTitle(self.title)
        self.setGeometry(200, 50, 1200, 1000)

        # Used to save the evolution of the algorithm
        self.population_history = list()

        self.init_gui()
        self.init_shared_objects()
        self.init_queues()
        self.init_live_chain()
        self.init_plots()
        self.show()
        self.init_streams()

    def init_shared_objects(self):
        """
        Initialises objects that are shared between the front-end and back-end processes.
        """
        self.ref_in_buffer = np.zeros(BUFFER, dtype=NP_FORMAT)
        self.meas_out_buffer = np.zeros(BUFFER, dtype=NP_FORMAT)
        self.meas_in_buffer = np.zeros(BUFFER, dtype=NP_FORMAT)
        self.bypass_live_chain = Flag(False)
        self.shutting_down = Flag(False)
        self.main_stream_paused = Flag(False)
        self.algorithm_running = Flag(False)
        self.main_sync_event = TimedEvent()  # Acts as a clock for synchronised recording
        self.meas_sync_event = TimedEvent()  # Acts as a clock for synchronised recording
        self.bg_model = None

    def init_queues(self):
        """
        Initialises the queues required to pass data from the back-end to the front-end for plotting.
        """
        self.bg_model_queue = Queue()
        self.latency_queue = Queue()
        self.filter_queue = Queue()
        self.alg_queue = Queue()

    def init_live_chain(self):
        """
        Initialises the (initially neutral) filter chain applied to the measurement output signal.
        """
        filters = list()
        for i in range(NUM_FILTERS):
            fc, gain, q = Population.random_filter_params()
            filters.append(PeakFilter(fc, 0, 1))
        self.live_chain = FilterChain(*filters)

    def init_streams(self):
        """
        Initialises the main and measurement streams. The main stream is responsible for the reference in (unfiltered
        system audio) and the processed measurement out audio, while the measurement stream is responsible for the
        measurement in audio (what is picked up by the microphone).
        """
        self.main_stream = MainStream(self.ref_in_buffer, self.meas_out_buffer, self.live_chain, self.bypass_live_chain,
                                      self.main_stream_paused, self.shutting_down, self.main_sync_event)
        self.main_stream.start()
        time.sleep(0.5)  # Required to avoid SIGSEGV error
        self.meas_stream = MeasStream(self.meas_in_buffer, self.shutting_down, self.meas_sync_event)
        self.meas_stream.start()

    def init_gui(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        (self.bg_model_ax, self.fitness_iter_ax), (self.filter_ax, self.stf_ax) = self.figure.subplots(2, 2)
        self.figure.tight_layout()
        
        self.bg_btn = QPushButton("Take Background Measurement")
        self.bg_btn.clicked.connect(self.start_bg_model_measurement)
        self.alg_btn = QPushButton("Start Algorithm")
        self.alg_btn.clicked.connect(self.start_algorithm)
        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)
        self.bypass_cbox = QCheckBox("Bypass")
        self.bypass_cbox.clicked.connect(self.toggle_bypass)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.bg_btn)
        button_layout.addWidget(self.alg_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.bypass_cbox)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def init_plots(self):
        self.plot_xlimits = F_LIMITS[:]
        
        # Background Measurement
        self.bg_model_ax.set_title("Background measurement", fontsize=FONTSIZE_TITLES)
        self.bg_model_ax.set_ylabel("Magnitude (dBFS)", fontsize=FONTSIZE_LABELS)
        self.bg_model_ax.set_xlabel("Frequency (Hz)", fontsize=FONTSIZE_LABELS)
        self.bg_model_ax.minorticks_on()
        self.bg_model_ax.tick_params(labelsize=FONTSIZE_TICKS)
        self.bg_model_ax.grid(which="major", linestyle="-", alpha=0.4)
        self.bg_model_ax.grid(which="minor", linestyle="--", alpha=0.2)
        self.bg_model_ax.set_xscale("log")

        # Algorithm Progression
        self.fitness_iter_ax.set_title("Algorithm progression", fontsize=FONTSIZE_TITLES)
        self.fitness_iter_ax.set_ylabel(r"Objective, $\phi$", fontsize=FONTSIZE_LABELS)
        self.fitness_iter_ax.set_xlabel("Generation", fontsize=FONTSIZE_LABELS)
        self.fitness_iter_ax.minorticks_on()
        self.fitness_iter_ax.tick_params(labelsize=FONTSIZE_TICKS)
        self.fitness_iter_ax.grid(which="major", linestyle="-", alpha=0.4)
        self.fitness_iter_ax.grid(which="minor", linestyle="--", alpha=0.2)
        
        # Current Filter Chain
        self.filter_ax.set_title("Current CTF", fontsize=FONTSIZE_TITLES)
        self.filter_ax.set_ylabel("Magnitude (dB)", fontsize=FONTSIZE_LABELS)
        self.filter_ax.set_xlabel("Frequency (Hz)", fontsize=FONTSIZE_LABELS)
        self.filter_ax.minorticks_on()
        self.filter_ax.tick_params(labelsize=FONTSIZE_TICKS)
        self.filter_ax.grid(which="major", linestyle="-", alpha=0.4)
        self.filter_ax.grid(which="minor", linestyle="--", alpha=0.2)
        self.filter_ax.set_xscale("log")
        self.filter_ax.set_ylim([-13, 13])
        self.update_filter_ax()

        # Current STF
        self.stf_ax.set_title("Measured STF", fontsize=FONTSIZE_TITLES)
        self.stf_ax.set_ylabel("Magnitude (dB)", fontsize=FONTSIZE_LABELS)
        self.stf_ax.set_xlabel("Frequency (Hz)", fontsize=FONTSIZE_LABELS)
        self.stf_ax.minorticks_on()
        self.stf_ax.tick_params(labelsize=FONTSIZE_TICKS)
        self.stf_ax.grid(which="major", linestyle="-", alpha=0.4)
        self.stf_ax.grid(which="minor", linestyle="--", alpha=0.2)
        self.stf_ax.set_xscale("log")

        self.canvas.draw()

    def toggle_buttons_state(self, bg, alg, export, bypass):
        """
        Takes boolean arguments to set the state of the buttons in the interface.
        """
        self.bg_btn.setEnabled(bg)
        self.alg_btn.setEnabled(alg)
        self.export_btn.setEnabled(export)
        self.bypass_cbox.setEnabled(bypass)

    def toggle_bypass(self):
        """
        Function triggered when interacting with the "Bypass" checkbox.
        Allows the user to A/B between the filtered and unfiltered signals.
        """
        if self.bypass_cbox.isChecked():
            print("\nFilters bypassed!")
            self.toggle_buttons_state(False, False, False, True)
            self.bypass_live_chain.set_state(True)
        else:
            print("\nFilters active!")
            self.toggle_buttons_state(True, True, True, True)
            self.bypass_live_chain.set_state(False)
    
    def update_bg_model_ax(self):
        # Remove all existing lines
        self.bg_model_ax.lines = list()
        # Plot model
        self.bg_model_ax.semilogx(*self.bg_model, linestyle="-", linewidth=2, color="black")
        # Plot snippets
        for x, y in self.bg_snippets:
            self.bg_model_ax.semilogx(x, y, color="gray", zorder=-1, linewidth=1)
        self.bg_model_ax.legend(["Averaged background model", "{0} audio clips".format(len(self.bg_snippets))],
                                fontsize=FONTSIZE_LEGENDS)
        self.canvas.draw()
    
    def update_fitness_iter_ax(self):
        # Remove all existing lines
        self.fitness_iter_ax.lines = list()
        fitness_r, fitness_p, fitness_c = list(), list(), list()
        iter_r, iter_p, iter_c = list(), list(), list()
        # Plot all fitness as scatter, colour coded
        for i, population in enumerate(self.population_history):
            for chain in population:
                if chain.kind == "r":
                    fitness_r.append(chain.fitness)
                    iter_r.append(i + 1)
                elif chain.kind == "p":
                    fitness_p.append(chain.fitness)
                    iter_p.append(i + 1)
                elif chain.kind == "c":
                    fitness_c.append(chain.fitness)
                    iter_c.append(i + 1)

        self.fitness_iter_ax.plot(iter_r, fitness_r, linestyle="", marker="o", color="C0", label="Random")
        self.fitness_iter_ax.plot(iter_p, fitness_p, linestyle="", marker="o", color="C1", label="Selected")
        self.fitness_iter_ax.plot(iter_c, fitness_c, linestyle="", marker="o", color="C2", label="Crossover")
        
        # Plot best fitness
        self.fitness_iter_ax.plot(self.best_fitness_list, color="black", label=r"Best $\phi$")
        self.fitness_iter_ax.legend(fontsize=FONTSIZE_LEGENDS)

        # Dynamically set axes limits
        self.fitness_iter_ax.relim()
        self.fitness_iter_ax.autoscale_view()
        self.fitness_iter_ax.set_autoscale_on(True)

        self.canvas.draw()

    def update_filter_ax(self):
        # Remove all existing lines
        self.filter_ax.lines = list()
        # Plot live_chain
        f, H = self.live_chain.get_chain_tf()
        self.filter_ax.semilogx(f, convert_to_dbfs(H), label="Chain (id={0})".format(self.live_chain.id), linewidth=2,
                                color="C0")
        # Plot individual filters
        for d in self.live_chain.get_all_filters_settings_tf():
            f, h = d["tf"]
            # label = d["settings"]
            self.filter_ax.semilogx(f, convert_to_dbfs(h), linestyle="--", zorder=-1, linewidth=1)
        self.filter_ax.legend(fontsize=FONTSIZE_LEGENDS)

        # Dynamically set axes limits
        self.filter_ax.relim()
        self.filter_ax.autoscale_view()
        self.filter_ax.set_autoscale_on(True)
        self.filter_ax.set_xlim(self.plot_xlimits)

        self.canvas.draw()

    def update_stf_ax(self):
        self.stf_ax.lines = list()
        self.stf_ax.plot(*self.best_stf, color="black", label=r"Current best STF ($\phi$={0})"
                         .format(round(self.best_fitness, 2)), linestyle="-", linewidth=2, zorder=-1)
        self.stf_ax.plot(*self.initial_stf, color="gray", label=r"Initial RTF ($\phi$={0})"
                         .format(round(self.initial_fitness, 2)), linestyle="-", linewidth=2)

        self.stf_ax.legend(fontsize=FONTSIZE_LEGENDS)
        self.canvas.draw()
        self.plot_xlimits = self.stf_ax.get_xlim()

    def collect_algorithm_queue_data(self):
        """
        Triggered after every iteration. Collects the data from the algorithm back-end.
        """
        # Collect the data
        self.initial_stf = self.alg_queue.get()
        self.initial_fitness = self.alg_queue.get()
        self.best_stf = self.alg_queue.get()
        self.best_fitness = self.alg_queue.get()
        self.best_fitness_list = self.alg_queue.get()
        curr_population = self.alg_queue.get()
        # deepcopy is needed because the filter chains are mutable and change between iterations
        # TODO no longer needed because doing a deepcopy in backend?
        # self.population_history.append(deepcopy(curr_population))
        self.population_history.append(curr_population)
    
    def start_bg_model_measurement(self):
        # Deactivate buttons
        self.toggle_buttons_state(False, False, False, False)
        # Pause the main stream
        self.main_stream_paused.set_state(True)
        time.sleep(0.5)
        # Spawn the model generating thread
        self.bg_model_measurement = BackgroundModel(self.bg_model_queue, self.meas_in_buffer, self.meas_sync_event)
        self.bg_model_measurement.start()
        self.bg_model_measurement.finished.connect(self.collect_bg_model_measurement)

    def collect_bg_model_measurement(self):
        self.main_stream_paused.set_state(False)
        time.sleep(0.5)
        self.bg_model = self.bg_model_queue.get()
        self.bg_snippets = self.bg_model_queue.get()
        self.update_bg_model_ax()
        # Reactivate buttons
        self.toggle_buttons_state(True, True, True, True)
        # self.wait_for_bg.set()

    def closeEvent(self, event):
        """
        Overrides superclass method.
        Ensures streams exit nicely.
        """
        self.shutting_down.set_state(True)
        time.sleep(0.5)
        event.accept()

    def start_algorithm(self):
        # self.wait_for_bg = threading.Event()
        # self.wait_for_bg.clear()
        print("\nYou get 10 seconds to GTFO...")
        time.sleep(10)
        # if self.bg_model is None:
        #     self.start_bg_model_measurement()
        # self.wait_for_bg.wait()
        # time.sleep(1)
        # Deactivate buttons except algorithm button
        self.toggle_buttons_state(False, True, False, False)
        self.alg_btn.setText("Stop Algorithm")
        self.alg_btn.clicked.disconnect()
        self.alg_btn.clicked.connect(self.stop_algorithm)
        self.algorithm_running.set_state(True)
        
        self.algorithm = Algorithm(self.alg_queue, self.ref_in_buffer, self.meas_in_buffer, self.main_sync_event,
                                   self.meas_sync_event, self.bg_model, self.live_chain, self.update_filter_ax_signal,
                                   self.collect_algorithm_queue_data_signal, self.update_fitness_iter_ax_signal,
                                   self.update_stf_ax_signal, self.algorithm_running)
        self.algorithm.start()
        # Algorithm thread finishes when the user presses the Stop Algorithm-button
        self.algorithm.finished.connect(self.collect_algorithm)
    
    def stop_algorithm(self):
        """
        Triggered when user clicks the Stop Algorithm-button.
        """
        self.toggle_buttons_state(False, False, False, False)
        self.alg_btn.setText("Stopping Algorithm...")
        self.alg_btn.clicked.disconnect()
        self.algorithm_running.set_state(False)
        
    def collect_algorithm(self):
        """
        Triggered when the algorithm thread has actually finished.
        """
        # Reactivate buttons
        self.alg_btn.clicked.connect(self.start_algorithm)
        self.toggle_buttons_state(True, True, True, True)
        self.alg_btn.setText("Start Algorithm")

    def export_data(self):
        file_path = QFileDialog.getSaveFileName(self, filter="Text Files (*.txt)")[0]
        
        if file_path == "":
            return
        
        with open(file_path, 'w') as outfile:
            # Save date and time
            outfile.write("Exported at: {0}\n\n\n".format(time.strftime("%c")))

            # Save parameter choices
            outfile.write("/// PARAMETERS ///\n"
                          "RATE = {0}\n"
                          "BUFFER = {1}\n"
                          "SNIPPET_LENGTH = {2}\n"
                          "BACKGROUND_LENGTH = {3}\n"
                          "MEAS_REF_LATENCY = {4}\n"
                          "LATENCY_MEASUREMENT_LENGTH = {5}\n\n"
                          
                          "EXPORT_WAV = {6}\n"
                          "F_LIMITS = {7}\n"
                          "OCT_FRAC = {8}\n"
                          "MAX_CROSSOVER_ATTEMPTS = {9}\n\n"
                          
                          "POP_SIZE = {10}\n"
                          "NUM_FILTERS = {11}\n"
                          "GAIN_LIMITS = {12}\n"
                          "Q_LIMITS = {13}\n\n"
                          
                          "PROP_PROMOTED = {14}\n\n"
                          
                          "STDEV_FC = {15}\n"
                          "STDEV_GAIN = {16}\n"
                          "STDEV_Q = {17}\n\n"
            
                          "PROP_RND = {18}\n\n\n"
            
                          .format(RATE, BUFFER, SNIPPET_LENGTH, BACKGROUND_LENGTH, MEAS_REF_LATENCY,
                                  LATENCY_MEASUREMENT_LENGTH,
                                  EXPORT_WAV, F_LIMITS, OCT_FRAC, MAX_CROSSOVER_ATTEMPTS,
                                  POP_SIZE, NUM_FILTERS, GAIN_LIMITS, Q_LIMITS,
                                  PROP_PROMOTED,
                                  STDEV_FC, STDEV_GAIN, STDEV_Q,
                                  PROP_RND)
                          )
        
            # Save background measurement
            if self.bg_model:
                snippets = ""
                for i, (f, db) in enumerate(self.bg_snippets):
                    snippets += "db_snippet_{0} = {1}\n".format(i, db)
                    
                outfile.write("/// BACKGROUND MODEL ///\n"
                              "freq = {0}\n"
                              "db_model = {1}\n"
                              "{2}\n\n"
                              .format(*self.bg_model, snippets))
            
            # Save initial fitness and STF
            outfile.write("/// INITIAL STF AND FITNESS ///\n"
                          "freq = {0}\n"
                          "stf_init = {1}\n"
                          "fitness_init = {2}\n\n\n"
                          .format(*self.initial_stf, self.initial_fitness))
            
            # Save algorithm iteration data
            w = "/// ALGORITHM ///\n\n"
            for i, population in enumerate(self.population_history):
                best_chain = sorted(population, key=lambda x: x.fitness)[0]
                w += "/// iteration: {0}, best.fitness: {1}, best.id: {2}, best.kind: {3}\n"\
                     .format(i + 1, best_chain.fitness, best_chain.id, best_chain.kind)
                for chain in population:
                    w += "chain_id_{0}:\nstf = {1}\nfitness = {2}\nkind = {3}\n"\
                         .format(chain.id, chain.stf[1], chain.fitness, chain.kind)
                    w += "{0}\n\n".format(chain.get_chain_settings())
                w += "\n\n"
            outfile.write(w)
        
        print("\nExported to {0}.".format(file_path))
    
    # BELOW IS FUNCTIONALITY THAT IS NO LONGER NEEDED (OR AT LEAST NOT NEEDED AT PRESENT)
    #
    # def start_latency_measurement(self):
    #     # Deactivate buttons
    #     self.toggle_buttons_state()
    #     self.latency_measurement = LatencyCalibration(self.latency_queue, self.ref_in_buffer, self.meas_in_buffer,
    #                                                   self.main_sync_event, self.meas_sync_event)
    #     self.latency_measurement.start()
    #     self.latency_measurement.finished.connect(self.collect_latency_measurement)
    #
    # def collect_latency_measurement(self):
    #     self.latency_ref = self.latency_queue.get()
    #     self.latency_meas = self.latency_queue.get()
    #     self.update_latency_ax()
    #
    #     # Reactivate buttons
    #     self.toggle_buttons_state()
    #
    # def random_filter_settings(self):
    #     """
    #     Toy function to simulate a change in EQ.
    #     """
    #     for i in range(len(self.live_chain.get_filters())):
    #         fc, gain, q = Population.random_filter_params()
    #         self.live_chain.set_filter_params(i, fc, gain, q)
    #     self.update_filter_ax()
    #
    # def update_latency_ax(self):
    #     # Latency Measurement (to go in init)
    #     self.latency_ax.set_title("Latency Measurement", fontsize=FONTSIZE_TITLES)
    #     self.latency_ax.set_ylabel("Amplitude [a.u.]", fontsize=FONTSIZE_LABELS)
    #     self.latency_ax.set_xlabel("Time [s]", fontsize=FONTSIZE_LABELS)
    #     self.latency_ax.minorticks_on()
    #     self.latency_ax.tick_params(labelsize=FONTSIZE_TICKS)
    #     self.latency_ax.grid(which="major", linestyle="-", alpha=0.4)
    #     self.latency_ax.grid(which="minor", linestyle="--", alpha=0.2)
    #
    #     # Remove all existing lines
    #     self.latency_ax.lines = list()
    #     # Plot latencies
    #     self.latency_ax.plot(*self.latency_ref, color="C0", label="Reference In")
    #     self.latency_ax.plot(*self.latency_meas, color="C1", label="Measurement In")
    #     self.latency_ax.legend(fontsize=FONTSIZE_LEGENDS)
    #
    #     # Dynamically set axes limits
    #     self.latency_ax.relim()
    #     self.latency_ax.autoscale_view()
    #     self.latency_ax.set_autoscale_on(True)
    #
    #     self.canvas.draw()
