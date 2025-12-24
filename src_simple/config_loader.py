#!/usr/bin/env python3
"""
Config loader that allows specifying different config files.
This solves the race condition problem where queued jobs all use the same config.py
"""

import sys
import importlib.util
import os
import argparse
from pathlib import Path


# ==================================================
# Config Loading Functions
# ==================================================
def load_config(config_path=None):
    """
    Load config from specified path or default config.py
    """
    if config_path is None:
        # Default behavior - use config.py
        import config
        return config
    
    # Load custom config file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Get absolute path
    config_path = os.path.abspath(config_path)
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    
    # Add to sys.modules so imports work correctly
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    
    return config_module

def get_config_from_args():
    """
    Parse command line arguments to get config file path
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file to use instead of default config.py')
    
    # Parse only known args to avoid conflicts with other parsers
    args, remaining = parser.parse_known_args()
    
    # Update sys.argv to remove the config argument for other parsers
    sys.argv = [sys.argv[0]] + remaining
    
    return load_config(args.config)
