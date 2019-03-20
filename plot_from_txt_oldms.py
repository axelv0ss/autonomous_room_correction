from matplotlib import pyplot as plt
import numpy as np
from params import *
import ast

path = "/Users/axel/Documents/_Coursework/Y4/MSci_project/_MSci/algorithm_results/" \
       "010219_wb_centerListPos_ChasingForeverLoop/010219_wb_centerListPos_ChasingForeverLoop_export.txt"
# path = "/Users/axel/Desktop/test2.txt"


def get_array(file_path, arr_name):
    """
    Returns an array from the saved file named name.
    """
    collecting = False
    
    with open(file_path, "r") as infile:
        for line in infile:
            if arr_name in line:
                arr = line.split("[")[1]
                collecting = True
                
            elif collecting and "]" in line:
                arr += line.split("]")[0]
                break
            
            elif collecting:
                arr += line
        
        if not collecting:
            raise AttributeError("No array with name '{0}' found.".format(arr_name))
        
        arr = " ".join(arr.split())  # Remove linebreaks and spaces
        arr = np.fromstring(arr, sep=" ")  # Convert to np array
        
        return arr


def plot_background(file_path):
    # Find how many snippets are in the file
    with open(file_path, "r") as infile:
        for line in infile:
            if "db_snippet" in line:
                num_snippets = int(line.split(" = ")[0].split("_")[-1]) + 1
    
    freq = get_array(file_path, "freq")
    db_model = get_array(file_path, "db_model")
    
    snippets = list()
    for i in range(num_snippets):
        snippets.append(get_array(file_path, "db_snippet_{0}".format(i)))
    
    # Plot
    plt.semilogx(freq, db_model, linestyle="-", linewidth=2, color="black")
    for db_snippet in snippets:
        plt.semilogx(freq, db_snippet, color="gray", zorder=-1, linewidth=1)

    plt.title("Background Measurement", fontsize=FONTSIZE_TITLES)
    plt.ylabel("Magnitude [dBFS]", fontsize=FONTSIZE_LABELS)
    plt.xlabel("Frequency [Hz]", fontsize=FONTSIZE_LABELS)
    plt.minorticks_on()
    plt.tick_params(labelsize=FONTSIZE_TICKS)
    plt.grid(which="major", linestyle="-", alpha=0.4)
    plt.grid(which="minor", linestyle="--", alpha=0.2)
    
    plt.legend(labels=["Log Binned Model", "N={0} Snippets".format(num_snippets)], fontsize=FONTSIZE_LEGENDS)
    plt.show()


def plot_algorithm_progression(file_path):
    with open(file_path, "r") as infile:
        for line in infile:
            if "ms_init" in line:
                best_ms = [float(line.split(" = ")[1])]
            if "best.ms" in line:
                best_ms.append(float(line.split("best.ms: ")[1].split(",")[0]))
    
    plt.plot(range(len(best_ms)), best_ms)
    
    plt.title("Algorithm Progression", fontsize=FONTSIZE_TITLES)
    plt.ylabel("Mean-Squared Error", fontsize=FONTSIZE_LABELS)
    plt.xlabel("Iteration", fontsize=FONTSIZE_LABELS)

    plt.minorticks_on()
    plt.tick_params(labelsize=FONTSIZE_TICKS)
    plt.grid(which="major", linestyle="-", alpha=0.4)
    plt.grid(which="minor", linestyle="--", alpha=0.2)
    
    plt.show()


def plot_layman_algprog(file_path):
    with open(file_path, "r") as infile:
        for line in infile:
            if "ms_init" in line:
                best_ms = [float(line.split(" = ")[1])]
            if "best.ms" in line:
                best_ms.append(float(line.split("best.ms: ")[1].split(",")[0]))
    
    x = list(range(0, len(best_ms)*100, 100))
    plt.plot(x, best_ms)
    
    # plt.title("Algorithm Progression", fontsize=FONTSIZE_TITLES)
    plt.ylabel("Mean-Squared Error", fontsize=FONTSIZE_LABELS*1.5)
    plt.xlabel("Time (s)", fontsize=FONTSIZE_LABELS*1.5)
    
    plt.minorticks_on()
    plt.tick_params(labelsize=FONTSIZE_TICKS*1.5)
    plt.grid(which="major", linestyle="-", alpha=0.4)
    plt.grid(which="minor", linestyle="--", alpha=0.2)
    
    plt.show()


def plot_layman(file_path):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    ENLARGE = 1.2
    
    freq = get_array(file_path, "freq")
    stf_init = get_array(file_path, "stf_init")
    
    ax1.semilogx(freq, stf_init, label="Initial Transfer Function", color="black")
    
    # plt.title("Algorithm Progression", fontsize=FONTSIZE_TITLES)
    ax1.set_ylabel("Magnitude (dB)", fontsize=FONTSIZE_LABELS * ENLARGE)
    ax1.set_xlabel("Frequency (Hz)", fontsize=FONTSIZE_LABELS * ENLARGE)
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=FONTSIZE_TICKS * ENLARGE)
    ax1.grid(which="major", linestyle="-", alpha=0.4)
    ax1.grid(which="minor", linestyle="--", alpha=0.2)
    
    with open(file_path, "r") as infile:
        for line in infile:
            if "ms_init" in line:
                best_ms = [float(line.split(" = ")[1])]
            if "best.ms" in line:
                best_ms.append(float(line.split("best.ms: ")[1].split(",")[0]))

    x = list(range(0, len(best_ms) * 100, 100))
    ax2.plot(x, best_ms)

    # plt.title("Algorithm Progression", fontsize=FONTSIZE_TITLES)
    ax2.set_ylabel("Mean-Squared Error", fontsize=FONTSIZE_LABELS)
    ax2.set_xlabel("Time (s)", fontsize=FONTSIZE_LABELS)

    ax2.minorticks_on()
    ax2.tick_params(labelsize=FONTSIZE_TICKS)
    ax2.grid(which="major", linestyle="-", alpha=0.4)
    ax2.grid(which="minor", linestyle="--", alpha=0.2)
    
    plt.show()


def plot_efp(file_path):
    with open(file_path, "r") as infile:
        for line in infile:
            if "ms_init" in line:
                best_ms = [float(line.split(" = ")[1])]
            if "best.ms" in line:
                best_ms.append(float(line.split("best.ms: ")[1].split(",")[0]))
    
    f, ax = plt.subplots(figsize=(4, 2))

    import matplotlib.font_manager as fm
    font = fm.FontProperties(family='Gill Sans',
                             fname='/Library/Fonts/GillSans.ttc',
                             size=FONTSIZE_LABELS * 2)
    
    ax.plot(range(len(best_ms)), best_ms, linewidth=3, color="#3ebae5")
    
    # plt.title("Sound Improvement", fontsize=FONTSIZE_TITLES)
    plt.ylabel("Sound Error", fontsize=FONTSIZE_LABELS * 2, fontproperties=font, color="#e4e4e4")
    plt.xlabel("Time", fontsize=FONTSIZE_LABELS * 2, fontproperties=font, color="#e4e4e4")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#e4e4e4")
    ax.spines["left"].set_color("#e4e4e4")
    
    # plt.minorticks_on()
    # plt.tick_params(labelsize=FONTSIZE_TICKS)
    plt.xticks([])
    plt.yticks([])
    # plt.grid(which="major", linestyle="-", alpha=0.4)
    # plt.grid(which="minor", linestyle="--", alpha=0.2)

    plt.tight_layout()
    f.patch.set_alpha(0)
    # ax.patch.set_alpha(0)

    f.savefig("efp.png", transparent=True, dpi=800)
    
    plt.show()


def plot_poster(file_path):
    FONTSIZE_LABELS = 25
    FONTSIZE_LEGENDS = 20
    FONTSIZE_TICKS = 20
    
    all_ms = dict()
    
    with open(file_path, "r") as infile:
        for line in infile:
            if "ms_init" in line:
                best_ms = [float(line.split(" = ")[1])]
            if "best.ms" in line:
                best_ms.append(float(line.split("best.ms: ")[1].split(",")[0]))
            if "/// iteration: " in line:
                curr_i = int(line.split("/// iteration: ")[1].split(", best.ms")[0])
                all_ms[curr_i] = list()
            if "ms = " in line:
                ms = float(line.split("ms = ")[1])
                all_ms[curr_i].append(ms)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(range(len(best_ms)), best_ms, linewidth=2, label="Best Chain", color="black")
    
    first = True
    for iter, ms_vals in all_ms.items():
        if first:
            first = False
            ax.plot([iter] * len(ms_vals), ms_vals, linestyle="", marker="o", color="gray", label="Individual Chains",
                    markersize=6)
        else:
            ax.plot([iter] * len(ms_vals), ms_vals, linestyle="", marker="o", color="gray", markersize=6)
    
    # plt.title("Sound Improvement", fontsize=FONTSIZE_TITLES)
    plt.ylabel("Mean-Squared Error", fontsize=FONTSIZE_LABELS)
    plt.xlabel("Iteration", fontsize=FONTSIZE_LABELS)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    
    # plt.minorticks_on()
    # plt.tick_params(labelsize=FONTSIZE_TICKS)
    # plt.xticks([])
    # plt.yticks([])
    # plt.grid(which="major", linestyle="-", alpha=0.4)
    # plt.grid(which="minor", linestyle="--", alpha=0.2)
    plt.legend(fontsize=FONTSIZE_LEGENDS)
    plt.minorticks_on()
    plt.grid(which="major", linestyle="-", alpha=0.4)
    plt.grid(which="minor", linestyle="--", alpha=0.2)
    plt.tick_params(labelsize=FONTSIZE_TICKS)

    plt.tight_layout()
    fig.patch.set_alpha(0)
    # fig.savefig("plot_poster.png", transparent=False, dpi=800)
    
    plt.show()


# plot_poster(path)
# plot_background(path)
# plot_efp(path)

# x = np.arange(0.01, 10, 0.01)
# y = -np.log(x)
# plt.semilogx(x, y)
# plt.show()
