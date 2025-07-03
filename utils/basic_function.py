import json,os

py_directory = os.path.dirname(os.path.realpath(__file__))

def get_abs_path(relative_path):
    """
    Obtain the absolute path through the relative path
    """
    abs_path = os.path.abspath(os.path.join(py_directory, relative_path))
    return abs_path

def load_json(relative_json_path):
    """
    Load the JSON file through the relative path
    """
    abs_json_path = get_abs_path(relative_json_path)
    with open(abs_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(relative_json_path,param_to_save):
    """
    Save the content of a variable to a JSON file through a relative path
    """
    abs_json_path = get_abs_path(relative_json_path)
    with open(abs_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(param_to_save, json_file, ensure_ascii=False, indent=4)
        print("The JSON file was saved successfully")

