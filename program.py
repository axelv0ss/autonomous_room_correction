from backend import *
from queue import Queue

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# TODO is meas_out_buffer really needed?


class Program(QWidget):
    # Signal
    update_filter_ax_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.update_filter_ax_signal.connect(self.update_filter_ax)
        self.title = "Autonomous Room Correction  //  Rate: {0}Hz  //  Format: {1}  //  Buffer: {2}  //  " \
                     "Range: {3}Hz - {4}Hz".format(RATE, str(NP_FORMAT).split('\'')[1], BUFFER, *F_LIMITS)
        self.setWindowTitle(self.title)
        self.setGeometry(200, 50, 1200, 1000)

        self.init_gui()
        self.init_shared_objects()
        self.init_queues()
        self.init_live_chain()
        self.init_plots()
        self.show()
        self.init_streams()

    def init_shared_objects(self):
        self.ref_in_buffer = np.zeros(BUFFER, dtype=NP_FORMAT)
        self.meas_out_buffer = np.zeros(BUFFER, dtype=NP_FORMAT)
        self.meas_in_buffer = np.zeros(BUFFER, dtype=NP_FORMAT)
        self.bypass_live_chain = Flag(False)
        self.shutting_down = Flag(False)
        self.main_stream_paused = Flag(False)
        self.main_sync_event = threading.Event()  # Acts as a clock for synchronised recording
        self.meas_sync_event = threading.Event()  # Acts as a clock for synchronised recording
        self.bg_model = None


    def init_queues(self):
        self.bg_model_queue = Queue()
        self.latency_queue = Queue()
        self.filter_queue = Queue()
        self.stf_queue = Queue()

    def init_live_chain(self):
        filters = list()
        for i in range(NUM_FILTERS):
            fc, gain, q = Population.random_filter_params()
            filters.append(PeakFilter(fc, 0, 1))
        self.live_chain = FilterChain(*filters)

    def init_streams(self):
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

        (self.bg_model_ax, self.latency_ax), (self.filter_ax, self.stf_ax) = self.figure.subplots(2, 2)
        self.figure.tight_layout()

        self.bg_btn = QPushButton("Take Background Measurement")
        self.bg_btn.clicked.connect(self.start_bg_model_measurement)
        self.latency_btn = QPushButton("Take Latency Measurement")
        self.latency_btn.clicked.connect(self.start_latency_measurement)

        self.rndflt_btn = QPushButton("Randomise Filter")
        self.rndflt_btn.clicked.connect(self.random_filter_settings)
        self.alg_btn = QPushButton("Start Algorithm")
        self.alg_btn.clicked.connect(self.start_algorithm)

        self.bypass_cbox = QCheckBox("Bypass")
        self.bypass_cbox.clicked.connect(self.toggle_bypass)

        button_layout = QHBoxLayout()
        # button_layout.addStretch()
        button_layout.addWidget(self.bg_btn)
        button_layout.addWidget(self.latency_btn)
        button_layout.addWidget(self.rndflt_btn)
        button_layout.addWidget(self.alg_btn)
        button_layout.addWidget(self.bypass_cbox)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def init_plots(self):
        # Background Measurement
        self.bg_model_ax.set_title("Background Measurement", fontsize=FONTSIZE_TITLES)
        self.bg_model_ax.set_ylabel("Magnitude [dBFS]", fontsize=FONTSIZE_LABELS)
        self.bg_model_ax.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE_LABELS)
        self.bg_model_ax.minorticks_on()
        self.bg_model_ax.tick_params(labelsize=FONTSIZE_TICKS)
        self.bg_model_ax.grid(which="major", linestyle="-", alpha=0.4)
        self.bg_model_ax.grid(which="minor", linestyle="--", alpha=0.2)
        self.bg_model_ax.set_xscale("log")

        # Latency Measurement
        self.latency_ax.set_title("Latency Measurement", fontsize=FONTSIZE_TITLES)
        self.latency_ax.set_ylabel("Amplitude [a.u.]", fontsize=FONTSIZE_LABELS)
        self.latency_ax.set_xlabel("Time [s]", fontsize=FONTSIZE_LABELS)
        self.latency_ax.minorticks_on()
        self.latency_ax.tick_params(labelsize=FONTSIZE_TICKS)
        self.latency_ax.grid(which="major", linestyle="-", alpha=0.4)
        self.latency_ax.grid(which="minor", linestyle="--", alpha=0.2)

        # Current Filter Chain
        self.filter_ax.set_title("Current Filter Chain", fontsize=FONTSIZE_TITLES)
        self.filter_ax.set_ylabel("Transfer Function " + r"$H(z)$" + " [dBFS]", fontsize=FONTSIZE_LABELS)
        self.filter_ax.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE_LABELS)
        self.filter_ax.minorticks_on()
        self.filter_ax.tick_params(labelsize=FONTSIZE_TICKS)
        self.filter_ax.grid(which="major", linestyle="-", alpha=0.4)
        self.filter_ax.grid(which="minor", linestyle="--", alpha=0.2)
        self.filter_ax.set_xscale("log")
        self.filter_ax.set_ylim([-13, 13])
        self.update_filter_ax()

        # Current STF
        self.stf_ax.set_title("Current STF", fontsize=FONTSIZE_TITLES)
        self.stf_ax.set_ylabel("Magnitude [dBFS]", fontsize=FONTSIZE_LABELS)
        self.stf_ax.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE_LABELS)
        self.stf_ax.minorticks_on()
        self.stf_ax.tick_params(labelsize=FONTSIZE_TICKS)
        self.stf_ax.grid(which="major", linestyle="-", alpha=0.4)
        self.stf_ax.grid(which="minor", linestyle="--", alpha=0.2)
        self.stf_ax.set_xscale("log")

        self.canvas.draw()

    def toggle_buttons_state(self):
        new_state = not self.bg_btn.isEnabled()
        self.bg_btn.setEnabled(new_state)
        self.latency_btn.setEnabled(new_state)
        self.alg_btn.setEnabled(new_state)
        self.rndflt_btn.setEnabled(new_state)
        self.bypass_cbox.setEnabled(new_state)

    def toggle_bypass(self):
        if self.bypass_cbox.isChecked():
            print("\nFilters bypassed!")
            self.bypass_live_chain.set_state(True)
        else:
            print("\nFilters active!")
            self.bypass_live_chain.set_state(False)

    def update_bg_model_ax(self):
        # Remove all existing lines
        self.bg_model_ax.lines = list()
        # Plot model
        self.bg_model_ax.semilogx(*self.bg_model, linestyle="-", linewidth=2, color="black")
        # Plot snippets
        for x, y in self.bg_snippets:
            self.bg_model_ax.semilogx(x, y, color="gray", zorder=-1, linewidth=1)
        self.bg_model_ax.legend(["Log Binned Model", "N={0} Snippets".format(len(self.bg_snippets))],
                                fontsize=FONTSIZE_LEGENDS)
        self.canvas.draw()

    def update_latency_ax(self):
        # Remove all existing lines
        self.latency_ax.lines = list()
        # Plot latencies
        self.latency_ax.plot(*self.latency_ref, color="C0", label="Reference In")
        self.latency_ax.plot(*self.latency_meas, color="C1", label="Measurement In")
        self.latency_ax.legend(fontsize=FONTSIZE_LEGENDS)

        # Dynamically set axes limits
        self.latency_ax.relim()
        self.latency_ax.autoscale_view()
        self.latency_ax.set_autoscale_on(True)

        self.canvas.draw()

    # @QtCore.pyqtSlot()
    def update_filter_ax(self):
        # Remove all existing lines
        self.filter_ax.lines = list()
        # Plot live_chain
        f, H = self.live_chain.get_chain_tf()
        self.filter_ax.semilogx(f, convert_to_dbfs(H), label="Chain", linewidth=2, color="C0")
        # Plot individual filters
        for d in self.live_chain.get_all_filters_settings_tf():
            f, h = d["tf"]
            label = d["settings"]
            self.filter_ax.semilogx(f, convert_to_dbfs(h), label=label, linestyle="--", zorder=-1, linewidth=1)
        # self.filter_ax.legend(fontsize=FONTSIZE_LEGENDS)

        # Dynamically set axes limits
        self.filter_ax.relim()
        self.filter_ax.autoscale_view()
        self.filter_ax.set_autoscale_on(True)

        self.canvas.draw()

    def update_stf_ax(self):
        # TODO
        self.stf_ax.lines = list()
        self.stf_ax.plot(*self.stf, color="black", label="Best STF (MS={0})".format(int(self.ms)), linestyle="-", linewidth=2, zorder=-1)
        self.stf_ax.plot(*self.initial_stf, color="gray", label="Initial STF", linestyle="-",
                         linewidth=2)
        # self.stf_ax.plot(*self.ref, color="C0", label="Normalised Reference In", linestyle="-", zorder=-1, linewidth=1)
        # self.stf_ax.plot(*self.meas, color="C1", label="Normalised Measurement In", linestyle="-", zorder=-1, linewidth=1)
        self.stf_ax.legend(fontsize=FONTSIZE_LEGENDS)
        self.canvas.draw()
    
    def start_bg_model_measurement(self):
        # Deactivate buttons
        self.toggle_buttons_state()
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
        self.toggle_buttons_state()
    
    def start_latency_measurement(self):
        # Deactivate buttons
        self.toggle_buttons_state()
        self.latency_measurement = LatencyCalibration(self.latency_queue, self.ref_in_buffer, self.meas_in_buffer,
                                                      self.main_sync_event, self.meas_sync_event)
        self.latency_measurement.start()
        self.latency_measurement.finished.connect(self.collect_latency_measurement)
    
    def collect_latency_measurement(self):
        self.latency_ref = self.latency_queue.get()
        self.latency_meas = self.latency_queue.get()
        self.update_latency_ax()

        # Reactivate buttons
        self.toggle_buttons_state()

    def closeEvent(self, event):
        """
        Overrides superclass method.
        Ensures streams exit nicely
        """
        self.shutting_down.set_state(True)
        time.sleep(0.5)
        event.accept()

    # TODO: EVERYTHING BELOW IS UNFINISHED
    def random_filter_settings(self):
        """
        Toy function to simulate a change in EQ.
        """
        for i in range(len(self.live_chain.get_filters())):
            fc, gain, q = Population.random_filter_params()
            self.live_chain.set_filter_params(i, fc, gain, q)
        self.update_filter_ax()

    def start_algorithm(self):
        # Deactivate buttons
        self.toggle_buttons_state()
        # TODO Pass in start parameters here, like an initial population etc?
        
        self.algorithm = Algorithm(self.stf_queue, self.ref_in_buffer, self.meas_in_buffer, self.main_sync_event, 
                                   self.meas_sync_event, self.bg_model, self.live_chain, self.update_filter_ax_signal)
        self.algorithm.start()
        self.algorithm.finished.connect(self.collect_algorithm)

    def collect_algorithm(self):
        self.initial_stf = self.stf_queue.get()
        self.stf = self.stf_queue.get()
        self.ms = self.stf_queue.get()
        self.update_stf_ax()

        # Reactivate buttons
        self.toggle_buttons_state()

    def export_data(self):
        # TODO: Need this functionality for saving data and later replotting.
        # Preferably a log of the whole program, saving everything. Not crucial at this stage.
        pass

