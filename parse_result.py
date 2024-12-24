import os
import json
import argparse

def find_best_log(root_dir):
    max_accuracy = -1
    best_log_path = ""
    best_log_info = {}

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == "log.txt":
                log_path = os.path.join(subdir, file)

                # if "endlayer" in log_path: continue
                
                with open(log_path, 'r') as log_file:
                    lines = log_file.readlines()[1:]
                    
                    local_max_accuracy = -1
                    last_item = None
                    invalid_line_count = 0 
                    total_lines = len(lines)
                    
                    for line in lines:
                        try:
                            item = json.loads(line.strip())
                            if "max_accuracy" in item:
                                if item["max_accuracy"] > max_accuracy:
                                    max_accuracy = item["max_accuracy"]
                                    best_log_path = log_path
                                    best_log_info = item
                                    best_n_parameters = item.get('n_parameters', 'N/A')  # 更新 n_parameters

                            elif "test_acc1" in item:
                                if item["test_acc1"] > local_max_accuracy:
                                    local_max_accuracy = item["test_acc1"]
                                    last_item = item
                        except json.JSONDecodeError:
                            invalid_line_count += 1
                            continue
                    
                    if local_max_accuracy > max_accuracy:
                        max_accuracy = local_max_accuracy
                        best_log_path = log_path
                        best_log_info = last_item
                        best_n_parameters = last_item.get('n_parameters', 'N/A')  

    if best_log_path:
        print("##################" )
        print(f"Max accuracy found: {max_accuracy}")
        print(f"Log file path: {best_log_path}")
        print(f"Details: max_accuracy={best_log_info.get('max_accuracy', 'N/A')}, "
              f"n_parameters={best_n_parameters}, "  
            #   f"epoch={best_log_info.get('epoch', 'N/A')}, "
              f"prompt_indices={best_log_info.get('prompt_indices', 'N/A')}")
    else:
        print("No valid log.txt files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the log with the highest max_accuracy or test_acc1.')
    parser.add_argument('root_dir', type=str, help='The root directory containing experiment folders.')
    
    args = parser.parse_args()
    
    find_best_log(args.root_dir)
