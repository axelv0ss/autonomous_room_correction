from backend import *
from queue import Queue

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# TODO is meas_out_buffer_queue really needed?


class Program(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Autonomous Room Correction ({0}Hz)".format(RATE)
        self.setWindowTitle(self.title)
        self.setGeometry(200, 50, 1200, 1000)

        self.bypass_chain = Flag(False)

        self.init_queues()
        self.init_filter_chain()
        self.init_streams()
        self.init_gui()
        self.init_plots()
        self.show()

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
        # self.alg_btn = QPushButton("Start Algorithm")
        # self.alg_btn.clicked.connect(self.start_algorithm)

        self.alg_btn = QPushButton("Randomise Filter")
        self.alg_btn.clicked.connect(self.random_filter_settings)

        self.bypass_cb = QCheckBox("Bypass")
        self.bypass_cb.clicked.connect(self.toggle_bypass)

        button_layout = QHBoxLayout()
        # button_layout.addStretch()
        button_layout.addWidget(self.bg_btn)
        button_layout.addWidget(self.latency_btn)
        button_layout.addWidget(self.alg_btn)
        button_layout.addWidget(self.bypass_cb)

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

    def init_queues(self):
        self.ref_in_buffer_queue = Queue(maxsize=1)
        self.meas_out_buffer_queue = Queue(maxsize=1)
        self.meas_in_buffer_queue = Queue(maxsize=1)

        self.bg_model_queue = Queue()
        self.latency_queue = Queue()
        self.filter_queue = Queue()
        self.rtf_queue = Queue()

    def init_filter_chain(self):
        f1 = PeakFilter(fc=50, gain=-10, q=0.5)
        f2 = PeakFilter(fc=120, gain=-5, q=1)
        f3 = PeakFilter(fc=300, gain=5, q=1)
        f4 = PeakFilter(fc=625, gain=5, q=2)
        f5 = PeakFilter(fc=1250, gain=2, q=1)
        f6 = PeakFilter(fc=2500, gain=-20, q=0.8)
        f7 = PeakFilter(fc=5000, gain=-5, q=1)
        f8 = PeakFilter(fc=10000, gain=5, q=1)
        f9 = PeakFilter(fc=20000, gain=-1, q=3)
        self.chain = FilterChain(f1, f2, f3, f4, f5, f6, f7, f8, f9)

    def init_streams(self):
        self.main_stream = MainStream(self.ref_in_buffer_queue, self.meas_out_buffer_queue,
                                      self.chain, self.bypass_chain)
        self.main_stream.start()
        time.sleep(0.5)  # Required to avoid SIGSEGV error
        self.meas_stream = MeasStream(self.meas_in_buffer_queue)
        self.meas_stream.start()

    def toggle_buttons_state(self):
        new_state = not self.bg_btn.isEnabled()
        self.bg_btn.setEnabled(new_state)
        self.latency_btn.setEnabled(new_state)
        self.alg_btn.setEnabled(new_state)
        self.bypass_cb.setEnabled(new_state)

    def toggle_bypass(self):
        if self.bypass_cb.isChecked():
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
    
    def start_bg_model_measurement(self):
        # Deactivate buttons
        self.toggle_buttons_state()
        # Pause the main stream
        self.main_stream.toggle_pause()
        time.sleep(0.1)
        # Spawn the model generating thread
        self.bg_model_measurement = BackgroundModel(self.bg_model_queue, self.meas_in_buffer_queue)
        self.bg_model_measurement.start()
        self.bg_model_measurement.finished.connect(self.collect_bg_model_measurement)

    def collect_bg_model_measurement(self):
        self.main_stream.toggle_pause()
        time.sleep(0.1)
        self.bg_model = self.bg_model_queue.get()
        self.bg_snippets = self.bg_model_queue.get()
        self.update_bg_model_ax()
        # Reactivate buttons
        self.toggle_buttons_state()
    
    def start_latency_measurement(self):
        # Deactivate buttons
        self.toggle_buttons_state()
        self.latency_measurement = LatencyCalibration(self.latency_queue, self.ref_in_buffer_queue,
                                                      self.meas_in_buffer_queue)
        self.latency_measurement.start()
        self.latency_measurement.finished.connect(self.collect_latency_measurement)
    
    def collect_latency_measurement(self):
        self.latency_ref = self.latency_queue.get()
        self.latency_meas = self.latency_queue.get()
        self.update_latency_ax()

        # Reactivate buttons
        self.toggle_buttons_state()

    # TODO

    def random_filter_settings(self):
        """
        Toy function to simulate a change in EQ.
        """
        freqs = [50, 120, 300, 625, 1250, 2500, 5000, 10000, 20000]
        for i, fc in enumerate(freqs):
            self.chain.set_filter_params(i, fc, np.random.randint(-20, 5), np.random.random() + 0.5)

    def start_algorithm(self):
        # Deactivate buttons
        self.bg_btn.setEnabled(False)
        self.alg_btn.setEnabled(False)
        # TODO Pass in start parameters here, like an initial population
        self.algorithm_iteration = AlgorithmIteration(self.algorithm_queue, self.ref_in_buffer_queue,
                                                      self.meas_in_buffer_queue)
        self.algorithm_iteration.start()
        self.algorithm_iteration.finished.connect(self.collect_algorithm)

    def collect_algorithm(self):
        self.ref = self.algorithm_queue.get()
        self.meas = self.algorithm_queue.get()
        self.update_freq_resp_ax()

        # Reactivate buttons
        self.bg_btn.setEnabled(True)
        self.alg_btn.setEnabled(True)


    # def update_filter_ax(self):
    #     # Generate the plot data
    #     curr_chain = FilterChain(CHAIN_SETTINGS)
    #
    #     # Discard the old graph
    #     self.filter_ax.clear()
    #
    #     # Plot
    #     for filt in curr_chain.get_filters():
    #         f, h = filt.get_tf()
    #         self.filter_ax.semilogx(f, 20 * np.log10(abs(h)), label=filt.get_settings())
    #     f, H = curr_chain.get_tf_chain()
    #     self.filter_ax.semilogx(f, 20 * np.log10(abs(H)), label="Chain", linestyle="--")
    #
    #     self.filter_ax.set_title("Filter Curve")
    #     self.filter_ax.set_ylabel('Transfer Function ' + r'$H(z)$' + ' [dB]')
    #     self.filter_ax.set_xlabel('Frequency [Hz]')
    #     self.filter_ax.legend()
    #     self.filter_ax.minorticks_on()
    #     self.filter_ax.grid(which="major", linestyle="-", alpha=0.4)
    #     self.filter_ax.grid(which="minor", linestyle="--", alpha=0.2)
    #
    #     self.canvas.draw()

    def closeEvent(self, event):
        """
        Overrides superclass method.
        Ensures streams exit nicely
        """
        self.main_stream.shutdown()
        time.sleep(0.1)
        self.meas_stream.shutdown()
        time.sleep(0.1)
        event.accept()
