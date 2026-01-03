import yaml
import os

def load_config(config_path="../train_config.yaml"):
    """
    Load YAML configuration file.
    Args:
        config_path (str): Relative or absolute path to the config file.
    Returns:
        dict: Configuration dictionary.
    """
    # Resolve absolute path if needed, assuming relative to this script or current working directory
    # If run from src/, ../train_config.yaml is valid.
    
    if not os.path.exists(config_path):
        # Try finding it relative to the script location if current dir fails
        script_dir = os.path.dirname(os.path.abspath(__file__))
        Project_root = os.path.dirname(script_dir)
        possible_path = os.path.join(Project_root, "train_config.yaml")
        if os.path.exists(possible_path):
            config_path = possible_path
            
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
