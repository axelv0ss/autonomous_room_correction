from backend import *
from queue import Queue

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# TODO is meas_out_buffer really needed?


class Program(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Autonomous Room Correction ({0}Hz)".format(RATE)
        self.setWindowTitle(self.title)
        self.setGeometry(200, 50, 1200, 1000)

        self.init_gui()
        self.init_shared_objects()
        self.init_queues()
        self.init_filter_chain()
        self.init_plots()
        self.show()
        self.init_streams()

    def init_shared_objects(self):
        self.ref_in_buffer = np.zeros(BUFFER, dtype=NP_FORMAT)
        self.meas_out_buffer = np.zeros(BUFFER, dtype=NP_FORMAT)
        self.meas_in_buffer = np.zeros(BUFFER, dtype=NP_FORMAT)
        self.bypass_chain = Flag(False)
        self.shutting_down = Flag(False)
        self.main_stream_paused = Flag(False)
        self.main_sync_event = threading.Event()  # Acts as a clock for synchronised recording
        self.meas_sync_event = threading.Event()  # Acts as a clock for synchronised recording

    def init_queues(self):
        self.bg_model_queue = Queue()
        self.latency_queue = Queue()
        self.filter_queue = Queue()
        self.rtf_queue = Queue()

    def init_filter_chain(self):
        f1 = PeakFilter(fc=50, gain=0, q=1)
        f2 = PeakFilter(fc=120, gain=0, q=1)
        f3 = PeakFilter(fc=300, gain=0, q=1)
        f4 = PeakFilter(fc=625, gain=0, q=1)
        f5 = PeakFilter(fc=1250, gain=0, q=1)
        f6 = PeakFilter(fc=2500, gain=0, q=1)
        f7 = PeakFilter(fc=5000, gain=0, q=1)
        f8 = PeakFilter(fc=10000, gain=0, q=1)
        f9 = PeakFilter(fc=20000, gain=0, q=1)
        self.chain = FilterChain(f1, f2, f3, f4, f5, f6, f7, f8, f9)

    def init_streams(self):
        self.main_stream = MainStream(self.ref_in_buffer, self.meas_out_buffer, self.chain, self.bypass_chain,
                                      self.main_stream_paused, self.shutting_down, self.main_sync_event)
        self.main_stream.start()
        time.sleep(0.5)  # Required to avoid SIGSEGV error
        self.meas_stream = MeasStream(self.meas_in_buffer, self.shutting_down, self.meas_sync_event)
        self.meas_stream.start()

    def init_gui(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        (self.bg_model_ax, self.latency_ax), (self.filter_ax, self.rtf_ax) = self.figure.subplots(2, 2)
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
        self.update_filter_ax()

        # Current RTF
        self.rtf_ax.set_title("Current RTF", fontsize=FONTSIZE_TITLES)
        self.rtf_ax.set_ylabel("Magnitude [dBFS]", fontsize=FONTSIZE_LABELS)
        self.rtf_ax.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE_LABELS)
        self.rtf_ax.minorticks_on()
        self.rtf_ax.tick_params(labelsize=FONTSIZE_TICKS)
        self.rtf_ax.grid(which="major", linestyle="-", alpha=0.4)
        self.rtf_ax.grid(which="minor", linestyle="--", alpha=0.2)
        self.rtf_ax.set_xscale("log")

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
            self.bypass_chain.set_state(True)
        else:
            print("\nFilters active!")
            self.bypass_chain.set_state(False)

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
        self.canvas.draw()

    def update_filter_ax(self):
        # Remove all existing lines
        self.filter_ax.lines = list()
        # Plot chain
        f, H = self.chain.get_chain_tf()
        self.filter_ax.semilogx(f, convert_to_dbfs(H), label="Chain", linewidth=2)
        # Plot individual filters
        for d in self.chain.get_all_filters_settings_tf():
            f, h = d["tf"]
            label = d["settings"]
            self.filter_ax.semilogx(f, convert_to_dbfs(h), label=label, linestyle="--", zorder=-1, linewidth=1)
        # self.filter_ax.legend(fontsize=FONTSIZE_LEGENDS)
        self.canvas.draw()

    def update_rtf_ax(self):
        # TODO
        self.rtf_ax.lines = list()
        self.rtf_ax.plot(*self.ref)
        self.rtf_ax.plot(*self.meas)
        self.canvas.draw()
    
    def start_bg_model_measurement(self):
        # Deactivate buttons
        self.toggle_buttons_state()
        # Pause the main stream
        self.main_stream_paused.set_state(True)
        time.sleep(0.1)
        # Spawn the model generating thread
        self.bg_model_measurement = BackgroundModel(self.bg_model_queue, self.meas_in_buffer, self.meas_sync_event)
        self.bg_model_measurement.start()
        self.bg_model_measurement.finished.connect(self.collect_bg_model_measurement)

    def collect_bg_model_measurement(self):
        self.main_stream_paused.set_state(False)
        time.sleep(0.1)
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
        time.sleep(0.2)
        event.accept()

    # TODO: EVERYTHING BELOW IS UNFINISHED
    def random_filter_settings(self):
        """
        Toy function to simulate a change in EQ.
        """
        freqs = [50, 120, 300, 625, 1250, 2500, 5000, 10000, 20000]
        for i, fc in enumerate(freqs):
            self.chain.set_filter_params(i, fc, np.random.randint(-20, 5), np.random.random() + 0.5)
        self.update_filter_ax()

    def start_algorithm(self):
        # Deactivate buttons
        self.toggle_buttons_state()
        # TODO Pass in start parameters here, like an initial population etc?
        self.algorithm_iteration = AlgorithmIteration(self.rtf_queue, self.ref_in_buffer, self.meas_in_buffer,
                                                      self.main_sync_event, self.meas_sync_event)
        self.algorithm_iteration.start()
        self.algorithm_iteration.finished.connect(self.collect_algorithm)

    def collect_algorithm(self):
        self.ref = self.rtf_queue.get()
        self.meas = self.rtf_queue.get()
        self.update_rtf_ax()

        # Reactivate buttons
        self.toggle_buttons_state()

    def export_data(self):
        # TODO: Need this functionality for saving data and later replotting.
        # Preferably a log of the whole program, saving everything. Not crucial at this stage.
        pass

