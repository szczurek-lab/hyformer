#!/usr/bin/env python
"""
Example script demonstrating how to use the configuration system for hyperparameter optimization.
This script shows how to load, modify, and save configurations for different experiments.
"""

import os
import json
import itertools
from typing import Dict, List, Any

from hyformer.configs.base import BaseConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.trainer import TrainerConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.dataset import DatasetConfig


def load_base_configs() -> Dict[str, BaseConfig]:
    """Load base configurations for an experiment."""
    # Define base configurations
    model_config = ModelConfig(
        model_name="Hyformer",
        embedding_dim=512,
        num_attention_heads=8,
        num_layers=6,
        max_seq_len=1024,
        vocab_size=1000,
    )
    
    trainer_config = TrainerConfig(
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_epochs=10,
        tasks={"pretrain": 1.0},
    )
    
    tokenizer_config = TokenizerConfig(
        vocabulary_path="path/to/vocab.json",
        max_molecule_length=512,
        tokenizer="SmilesTokenizer",
    )
    
    dataset_config = DatasetConfig(
        dataset_type="example_dataset",
        train_data_path="path/to/train.csv",
        val_data_path="path/to/val.csv",
        task_type="regression",
    )
    
    return {
        "model": model_config,
        "trainer": trainer_config,
        "tokenizer": tokenizer_config,
        "dataset": dataset_config,
    }


def generate_hyperparameter_grid() -> List[Dict[str, Any]]:
    """Generate a grid of hyperparameters to search over."""
    # Define hyperparameter search space
    param_grid = {
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size": [16, 32, 64],
        "num_layers": [4, 6, 8],
        "embedding_dim": [256, 512, 768],
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    # Convert to list of dictionaries
    return [dict(zip(keys, combo)) for combo in combinations]


def run_hyperparameter_optimization():
    """Run hyperparameter optimization by generating and saving multiple configurations."""
    # Create output directory
    output_dir = "hyformer/configs/experiments"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base configurations
    base_configs = load_base_configs()
    
    # Generate hyperparameter grid
    param_grid = generate_hyperparameter_grid()
    
    # Generate and save configurations for each hyperparameter combination
    for i, params in enumerate(param_grid):
        # Create a copy of the base configurations
        experiment_configs = {k: v.to_dict() for k, v in base_configs.items()}
        
        # Update configurations with hyperparameters
        for param_name, param_value in params.items():
            if param_name in ["learning_rate", "batch_size"]:
                experiment_configs["trainer"][param_name] = param_value
            elif param_name in ["num_layers", "embedding_dim"]:
                experiment_configs["model"][param_name] = param_value
        
        # Save the configuration
        experiment_name = f"experiment_{i+1}"
        output_path = os.path.join(output_dir, f"{experiment_name}.json")
        
        with open(output_path, "w") as f:
            json.dump(experiment_configs, f, indent=2)
        
        print(f"Saved configuration for {experiment_name} to {output_path}")
        print(f"Parameters: {params}")
        print("-" * 50)


def load_and_run_experiment(experiment_path: str):
    """Load a configuration and simulate running an experiment."""
    # Load the configuration
    with open(experiment_path, "r") as f:
        config_dict = json.load(f)
    
    # Convert dictionaries to configuration objects
    model_config = ModelConfig.from_dict(config_dict["model"])
    trainer_config = TrainerConfig.from_dict(config_dict["trainer"])
    tokenizer_config = TokenizerConfig.from_dict(config_dict["tokenizer"])
    dataset_config = DatasetConfig.from_dict(config_dict["dataset"])
    
    # Print configuration details
    print(f"Running experiment with configuration from {experiment_path}")
    print(f"Model: {model_config.model_name} with {model_config.num_layers} layers")
    print(f"Training with batch size {trainer_config.batch_size} and LR {trainer_config.learning_rate}")
    print(f"Using {tokenizer_config.tokenizer} with max length {tokenizer_config.max_molecule_length}")
    print(f"Dataset: {dataset_config.dataset_type} ({dataset_config.task_type})")
    
    # Here you would actually run the experiment
    print("Simulating experiment run...")
    
    # Example of updating a configuration parameter with mixed valid and invalid parameters
    print("\nDemonstrating the new update behavior:")
    
    # Create a dictionary with both valid and invalid parameters
    update_params = {
        "batch_size": 64,                # Valid parameter
        "learning_rate": 2e-4,           # Valid parameter
        "unknown_param": 42,             # Invalid parameter
        "another_unknown_param": "test", # Invalid parameter
    }
    
    # Update the trainer configuration - only valid parameters will be updated
    print(f"Before update: batch_size={trainer_config.batch_size}, learning_rate={trainer_config.learning_rate}")
    trainer_config.update(**update_params)
    print(f"After update: batch_size={trainer_config.batch_size}, learning_rate={trainer_config.learning_rate}")
    print("Note: Invalid parameters were silently ignored")
    
    # Try to access an invalid parameter (will raise an error)
    try:
        # Using direct attribute access instead of dictionary-like access
        value = getattr(trainer_config, "unknown_param", None)
        if value is not None:
            print(f"This should not print! Value: {value}")
        else:
            print("Attribute 'unknown_param' does not exist (returns None with getattr)")
    except AttributeError as e:
        print(f"Error when accessing invalid parameter: {e}")
    
    print("Experiment completed!")


if __name__ == "__main__":
    # Generate configurations for hyperparameter optimization
    run_hyperparameter_optimization()
    
    # Example of loading and running a specific experiment
    # Uncomment to run:
    # experiment_path = "hyformer/configs/experiments/experiment_1.json"
    # if os.path.exists(experiment_path):
    #     load_and_run_experiment(experiment_path) 