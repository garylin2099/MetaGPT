import re
import json
import os

def extract_highest_scores(file_paths):
    """
    Extract the highest scores from a list of JSON file paths.

    Parameters:
    file_paths (list of str): List of paths to JSON files.

    Returns:
    list of float: List of highest scores from each file.
    """
    scores = []
    num_files = len([name for name in os.listdir(file_paths) if name.endswith('{i:04d}.json')])
    for i in range(1,num_files):
        # Regular expression patterns for matching accuracy values
        accuracy_pattern = re.compile(r'(\w+(?: \w+)*): (\d\.\d{4})')
        best_model_pattern = re.compile(r'Best model on eval data: (\w+(?: \w+)*) with accuracy: (\d\.\d{4})')
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            
            # Extract the result field from the JSON data
            result_text = json_data['task_map'][str(len(json_data["task_map"]))]["result"]
            
            # Find all accuracies
            accuracies = accuracy_pattern.findall(result_text)
            best_model = best_model_pattern.search(result_text)
            
            # Determine the highest score in the current file
            current_highest = 0.0
            for model, accuracy in accuracies:
                accuracy_value = float(accuracy)
                if accuracy_value > current_highest:
                    current_highest = accuracy_value
            
            # Check the best model accuracy
            if best_model:
                best_model_accuracy = float(best_model.group(2))
                if best_model_accuracy > current_highest:
                    current_highest = best_model_accuracy
            
            # Append the highest score to the scores list
            scores.append(current_highest)

        return scores

def get_max_score_with_index(file_paths):
    """
    Get the maximum score and its index from a list of JSON file paths.

    Parameters:
    file_paths (list of str): List of paths to JSON files.

    Returns:
    tuple: The highest score and its index in the file_paths list.
    """
    highest_scores = extract_highest_scores(file_paths)
    if highest_scores:
        max_score = max(highest_scores)
        max_index = highest_scores.index(max_score)
        return max_score, max_index
    else:
        return None, None

def get_baseline_code_path(file_paths, code_base_path):
    """
    Get the baseline code path based on the highest score.

    Parameters:
    file_paths (list of str): List of paths to JSON files.
    code_base_path (str): Base directory path where code files are located.

    Returns:
    str: The path of the code file with the highest score.
    """
    _, max_index = get_max_score_with_index(file_paths)
    if max_index is not None:
        # Construct the full path to the code file based on the highest score index
        code_file_name = f"code_{max_index}.py"
        baseline_code_path = f"{code_base_path}{code_file_name}"
        return baseline_code_path
    else:
        return None