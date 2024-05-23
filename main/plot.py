import matplotlib.pyplot as plt
from meta_data import *
import os


def compare_plot3(mean_lists, std_lists, traina_num, method, target, save_path, raw_size, title_data_name, legends=None, y_name="RMSE", imbalanced = True):
    # Create a figure for plotting
    plt.figure("compare_{}_{}".format(mean_lists, std_lists))
    plt.axhline(y=mean_lists[0][0], color='r', linestyle='-')
    
    # Define x-axis values
    x_axis = [x*0.2 for x in range(traina_num + 1)]
    
    # Check if legends list is provided and has the correct length
    if legends and len(legends) != len(mean_lists):
        raise ValueError("Length of legends list must be equal to the number of data sets (mean_lists)")
    
    if imbalanced:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    else:
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    # Define a list of markers
    markers = ['o', 's', 'D', '*', '^', 'v', '<', '>', 'p', 'h']
    
    # Ensure there are enough markers, cycle if not
    if len(markers) < len(mean_lists):
        markers = markers * (len(mean_lists) // len(markers) + 1)
        
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    
    # Plot each set of means and standard deviations
    for idx, (mean_list, std_list) in enumerate(zip(mean_lists, std_lists)):
        label = legends[idx] if legends else f"Data Set {idx+1}"
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]  # Select marker
        
        if std_list is not None:
            # If standard deviation data is available, plot with error bars
            plt.errorbar(x_axis, mean_list, yerr=std_list, fmt='-', marker=marker, capsize=5, label=label, color = color)
        else:
            # If no standard deviation data, plot a simple line with markers
            plt.plot(x_axis, mean_list, '-' + marker, label=label, color = color)
    
    # Set labels and title
    plt.xlabel('syn/raw', fontsize = 18)
    plt.ylabel(y_name, fontsize = 18)
    plt.title(f"{title_data_name}: {method} on {target}", fontsize = 18)
    
    # Display legend
    plt.legend(fontsize = 15)
    
    plt.grid(axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    print(save_path)
    

def read_results_file(file_path):
    file_name = os.path.basename(file_path)
    name_parts = file_name.split('_')
    current_post = name_parts[0]
    method = name_parts[1]
    data_name = name_parts[2].split('.')[0]  # Remove file extension
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        mean_line = lines[0].strip().split("Mean: ")[1]
        std_line = lines[1].strip().split("Std: ")[1]
        
        mean_values = eval(mean_line)
        std_values = eval(std_line)
    
    return current_post, method, data_name, mean_values, std_values


if __name__ == "__main__":
    
    # An example how you can use plot function
    mean_lists = [
        [0.541, 0.388, 0.3390000000000001, 0.29800000000000004, 0.29100000000000004, 0.269],
        [0.541, 0.489, 0.46799999999999997, 0.45899999999999996, 0.45200000000000007, 0.45199999999999996],
        [0.541, 0.5149999999999999, 0.517, 0.5189999999999999, 0.525, 0.5189999999999999]
    ]
    
    std_lists = [
        [0.040527768258318914, 0.027522717889045794, 0.037483329627982606, 0.04452527372178636, 0.05366563145999496, 0.029025850547399968],
        [0.040527768258318914, 0.056833088953531244, 0.07529940238806677, 0.08042076846188428, 0.08136338242723198, 0.07181573643707903],
        [0.040527768258318914, 0.03984344362627307, 0.05215361924162118, 0.038955102361564936, 0.027156951228000544, 0.02484954727957832]
    ]
    
    control_ls = {0: "imbalanced", 1: "spurious_minor", 2: "spurious_overall", 3: "spurious_major", 4: "spurious_worst"}
    
    data_name_ls = {0: "openmlDiabetes", 1: "gender", 2: "heart_failure", 3: "craft"}
    
    method_ls = {0: "logistic", 1: "catBoost", 2: "randomForest"}
    
    title_data_name_ls = {"openmlDiabetes": "Diabetes", "gender": "Gender", "heart_failure": "HeartFailure", "craft": "Crafted data for simulation"}
    
    control = control_ls[0]
    data_name = data_name_ls[2]
    title_data_name = title_data_name_ls[data_name]
    method = method_ls[0]
    
    
    info = info_dict[data_name]
    target = info["target"]
    metric = info["metric"]
    raw_size = info["raw_size"]
    title_data_name = title_data_name_ls[data_name]
    
    traina_num = 5
    
    legends = ["OPAL", "Smote", "Duplicate"]
    save_path = "./main/results/plots/{}_{}_{}.png".format(control, method, data_name)
    
    compare_plot3(mean_lists, std_lists, traina_num, method, target, save_path, raw_size, title_data_name, legends, y_name=metric, imbalanced=False)
