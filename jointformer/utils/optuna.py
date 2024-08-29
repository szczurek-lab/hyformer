import json

def get_hparam_search_space(trial, hyperparameters_grid):
    """
    Generates a hyperparameter search space for a given trial.

    Args:
        trial (optuna.trial.Trial): The trial for which to generate the search space.
        hyperparameters_grid (dict): A dictionary defining the hyperparameters and their possible values.

    Returns:
        dict: A dictionary containing the hyperparameters and their suggested values for the trial.
    """
    hyperparams = {}
    for hp_name, hp_params in hyperparameters_grid.items():
        if hp_params['type'] == 'int':
            hyperparams[hp_name] = trial.suggest_int(hp_name, hp_params['low'], hp_params['high'])
        elif hp_params['type'] == 'categorical':
            hyperparams[hp_name] = trial.suggest_categorical(hp_name, hp_params['choices'])
        elif hp_params['type'] == 'float':
            hyperparams[hp_name] = trial.suggest_float(hp_name, hp_params['low'], hp_params['high'])
    return hyperparams

def load_json(file_path):
    """
    Loads a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)
    
def save_json(file_path, data):
    """
    Saves a dictionary to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        data (dict): The dictionary to save.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f)