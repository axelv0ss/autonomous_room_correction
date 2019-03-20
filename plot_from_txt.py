from matplotlib import pyplot as plt
import numpy as np

path = "/Users/axel/Documents/_Coursework/Y4/MSci_project/_MSci/algorithm_results/" \
       "120319_whiteNoise/export.txt"


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
    

def plot_algorithm_progression(file_path):
    
    all_fitness = dict()
    all_kinds = dict()
    
    with open(file_path, "r") as infile:
        for line in infile:
            if "fitness_init" in line:
                best_fitness = [float(line.split(" = ")[1])]
            if "best.fitness" in line:
                best_fitness.append(float(line.split("best.fitness: ")[1].split(",")[0]))
            if "/// iteration: " in line:
                curr_i = int(line.split("/// iteration: ")[1].split(", best.fitness")[0])
                all_fitness[curr_i] = list()
                all_kinds[curr_i] = list()
            if "fitness = " in line:
                fitness = float(line.split("fitness = ")[1])
                all_fitness[curr_i].append(fitness)
            # if line[:7] == "kind = ":
            if "kind = " in line:
                kind = line.split("kind = ")[1]
                all_kinds[curr_i].append(kind)
            # print(all_kinds, all_fitness)

    fig, ax = plt.subplots()
    
    for (iter, fitness_vals), kinds in zip(all_fitness.items(), all_kinds.values()):
        for fitness, kind in zip(fitness_vals, kinds):
            if kind == "r\n":
                ax.plot([iter], fitness, linestyle="", marker="o", color="C0", markersize=6)
            elif kind == "p\n":
                ax.plot([iter], fitness, linestyle="", marker="o", color="C1", markersize=6)
            elif kind == "c\n":
                ax.plot([iter], fitness, linestyle="", marker="o", color="C2", markersize=6)
            elif kind == "m\n":
                ax.plot([iter], fitness, linestyle="", marker="o", color="C2", markersize=6)
    
    ax.plot(range(len(best_fitness)), best_fitness)
    
    ax.set_title("Algorithm Progression")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Iteration")
    
    plt.show()


def plot_layman(file_path):
    f, (ax1, ax2) = plt.subplots(1, 2)
    
    freq = get_array(file_path, "freq")
    stf_init = get_array(file_path, "stf_init")
    
    ax1.semilogx(freq, stf_init, label="Initial Transfer Function", color="black")
    
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_xlabel("Frequency (Hz)")
    
    with open(file_path, "r") as infile:
        for line in infile:
            if "fitness_init" in line:
                best_fitness = [float(line.split(" = ")[1])]
            if "best.fitness" in line:
                best_fitness.append(float(line.split("best.fitness: ")[1].split(",")[0]))
    
    x = list(range(0, len(best_fitness)))
    ax2.plot(x, best_fitness)
    
    ax2.set_ylabel("Fitness")
    ax2.set_xlabel("Iteration")
    
    plt.show()


plot_algorithm_progression(path)
plot_layman(path)
