import nbformat
import json
import os

from datetime import datetime
from pathlib import Path
from metagpt.roles.role import Role
from metagpt.utils.common import read_json_file
from metagpt.const import DATA_PATH
from metagpt.utils.common import write_json_file

def load_history(save_dir: str = ""):
    """
    Load plan and code execution history from the specified save directory.

    Args:
        save_dir (str): The directory from which to load the history.

    Returns:
        Tuple: A tuple containing the loaded plan and notebook.
    """

    plan_path = Path(save_dir) / "plan.json"
    nb_path = Path(save_dir) / "history_nb" / "code.ipynb"
    plan = read_json_file(plan_path)
    nb = nbformat.read(open(nb_path, "r", encoding="utf-8"), as_version=nbformat.NO_CONVERT)
    return plan, nb

def save_code_file(name: str, code_context: str, file_format: str = "py") -> None:
    """
    Save code files to a specified path.

    Args:
    - name (str): The name of the folder to save the files.
    - code_context (str): The code content.
    - file_format (str, optional): The file format. Supports 'py' (Python file), 'json' (JSON file), and 'ipynb' (Jupyter Notebook file). Default is 'py'.

    Returns:
    - None
    """
    # Create the folder path if it doesn't exist
    os.makedirs(name= DATA_PATH / "output" / f"{name}", exist_ok=True)
    # Choose to save as a Python file or a JSON file based on the file format
    file_path = DATA_PATH / "output_1" / f"{name}/code.{file_format}"
    if file_format == "py":
        file_path.write_text(code_context + "\n\n", encoding="utf-8")
    elif file_format == "json":
        # Parse the code content as JSON and save
        data = {"code": code_context}
        write_json_file(file_path, data, encoding="utf-8", indent=2)
    elif file_format == "ipynb":
        nbformat.write(code_context, file_path)
    else:
        raise ValueError("Unsupported file format. Please choose 'py', 'json', or 'ipynb'.")

def save_history(role: Role, save_dir: str = "",round: str=""):
    """
    Save plan and code execution history to the specified directory.

    Args:
        role (Role): The role containing the plan and execute_code attributes.
        save_dir (str): The directory to save the history.

    Returns:
        Path: The path to the saved history directory.
    """
    record_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save_path = DATA_PATH / "output_1" / f"{record_time}"
    save_path = Path(save_dir)/ f"{round}_output/{record_time}"
    # overwrite exist trajectory
    save_path.mkdir(parents=True, exist_ok=True)

    plan = role.planner.plan.dict()

    with open(save_path / "plan.json", "w", encoding="utf-8") as plan_file:
        json.dump(plan, plan_file, indent=4, ensure_ascii=False)

    # save_code_file(name=Path(record_time), code_context=role.execute_code.nb, file_format="ipynb")
    return save_path

