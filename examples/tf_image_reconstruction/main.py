"""Example of experiment specific code for image reconstruction task using the framework"""
from pix2pix import Pix2Pix
from xflow.utils.config import ConfigManager, load_validated_config
from xflow.trainers.trainer import BaseTrainer 
from xflow.data.provider import FileProvider
from xflow.data.pipeline import InMemoryPipeline
from xflow.data.transforms import build_transforms_from_config
from xflow.utils.helper import get_base_dir
from pathlib import Path
import os

cur_dir = get_base_dir()

# ====================================
# Training configuration
# ====================================
config_manager = ConfigManager(load_validated_config(os.path.join(cur_dir, "config.yaml")))
config = config_manager.get()
base = Path(config["paths"]["base"])

# ====================================
# Data pipeline
# ====================================
provider = FileProvider(base / config["data"]["root"])
print(f"total files found:{len(provider)}")
transforms = build_transforms_from_config(config['transforms'])
print(f"total transforms: {len(transforms)}")
pipeline = InMemoryPipeline(provider, transforms)


# ====================================
# Model definition
# ====================================
model = Pix2Pix(config["model"])


# ====================================
# Training pipeline
# ====================================
trainer = BaseTrainer(  # will use TF Trainer 
    model=model,
    pipeline=pipeline,
    config=config["training"]
)
trainer.fit()



# ====================================
# Save results
# ====================================
model.save(config["model"]["save_path"])


