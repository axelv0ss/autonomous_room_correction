from matplotlib import pyplot as plt
import numpy as np
from params import *

path = "/Users/axel/Desktop/export_test.txt"
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
    

plot_background(path)
# plot_algorithm_progression(path)
