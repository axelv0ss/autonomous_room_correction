import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


class PeakFilter(object):
    def __init__(self, fc, gain, q, sampling_rate):
        """
        :param fc: The center frequency (in Hz) of the filter
        :param gain: The gain (in dB) of the filter
        :param q: The Q-factor of the filter
        :param sampling_rate: The sampling rate (in /s) of the input signal
        """
        # Save instance variables
        self.fc = fc
        self.gain = gain
        self.q = q
        self.sampling_rate = sampling_rate
        
        # Frequencies declared as fractions of sampling rate
        f_frac = self.fc / sampling_rate
        w_frac = 2 * np.pi * f_frac
        
        # Declare Transfer Function in S-domain
        num = [1, 10 ** (self.gain / 20) * w_frac / self.q, w_frac ** 2]
        den = [1, w_frac / self.q, w_frac ** 2]
        
        # Calculate coefficients in Z-domain using Bilinear transform
        self.b, self.a = signal.bilinear(num, den)
    
    def get_b_a(self):
        return self.b[:], self.a[:]
    
    def get_desc(self):
        return "fc={0}, g={1}, q={2}".format(self.fc, self.gain, self.q)
    
    def get_tf(self):
        w_f, h = signal.freqz(*self.get_b_a())
        f = w_f * self.sampling_rate / (2 * np.pi)
        return f, h


class FilterChain(object):
    def __init__(self, filters):
        """
        :param filters: List of individual PeakFilter objects to be applied
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
    
    def get_tf_chain(self):
        f, H = self.filters[0].get_tf()
        if len(self.filters) > 1:
            for filter in self.filters[1:]:
                H *= filter.get_tf()[1]
        return f, H


# For plots in interface
FONTSIZE_TITLES = 12
FONTSIZE_LABELS = 25
FONTSIZE_LEGENDS = 20
FONTSIZE_TICKS = 20

if __name__ == "__main__":
    RATE = 44100
    
    filters = [PeakFilter(700, 5, 1, RATE),
               PeakFilter(2000, 10, 2, RATE),
               PeakFilter(10000, -20, 0.5, RATE)]
    chain = FilterChain(filters)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # PLOT CHAIN
    f, H = chain.get_tf_chain()
    plt.semilogx(f, 20 * np.log10(abs(H)), linestyle="-", label="Chain", linewidth=2, color="black", zorder=5)
    
    # PLOT INDIVIDUAL
    first = True
    for i, filt in enumerate(filters):
        f, h = filt.get_tf()
        if first:
            first = False
            plt.semilogx(f, 20 * np.log10(abs(h)), linestyle="--", color="gray", label="Individual Filters")
        else:
            plt.semilogx(f, 20 * np.log10(abs(h)), linestyle="--", color="gray")

    plt.minorticks_on()
    plt.tick_params(labelsize=FONTSIZE_TICKS)
    plt.grid(which="major", linestyle="-", alpha=0.4)
    plt.grid(which="minor", linestyle="--", alpha=0.2)
    
    # plt.title('Digital Filter Frequency Response')
    plt.ylabel('Magnitude (dB)', fontsize=FONTSIZE_LABELS)
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE_LABELS)
    plt.legend(fontsize=FONTSIZE_LEGENDS)
    plt.grid()
    
    plt.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig("filter_chain.png", transparent=False, dpi=800)
    # plt.show()
