# config manger 09/25
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    # manages loading 
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_configs()
    
    def _load_configs(self):
        # ymal
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory {self.config_dir} not found")
        
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            with open(config_file, 'r') as f:
                self.configs[config_name] = yaml.safe_load(f)
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        # config by anmes
        if config_name not in self.configs:
            raise KeyError(f"Configuration '{config_name}' not found")
        return self.configs[config_name]
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        # model specifics 
        model_configs = self.get_config("model_configs")
        if model_name not in model_configs.get("models", {}):
            raise KeyError(f"Model '{model_name}' not found in configuration")
        return model_configs["models"][model_name]
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        # pipeline config 
        return self.get_config("pipeline_configs")
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        # updates configs
        if config_name in self.configs:
            self.configs[config_name].update(updates)