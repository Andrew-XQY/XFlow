"""Example of experiment specific code for image reconstruction task using the framework"""
from pix2pix import Pix2Pix
from xflow.utils.config import ConfigManager, load_validated_config
from xflow.trainers.trainer import BaseTrainer 
from xflow.data.loader import BasePipeline      
import os

cur_dir = os.path.dirname(__file__)

# ====================================
# Training configuration
# ====================================
config_manager = ConfigManager(load_validated_config(os.path.join(cur_dir, "config.yaml")))
config = config_manager.get()
model_config = config.get("model", {})
training_config = config.get("training", {})
data_config = config.get("data", {})

print("Model configuration:", model_config)
print("Training configuration:", training_config)
print("Data configuration:", data_config)

# ====================================
# Data pipeline
# ====================================
pipeline = BasePipeline(data_config)



# ====================================
# Model definition
# ====================================
model = Pix2Pix(config)



# ====================================
# Training pipeline
# ====================================
trainer = BaseTrainer(
    model=model,
    pipeline=pipeline,
    config=training_config
)
trainer.fit()



# ====================================
# Save results
# ====================================
model.save("./checkpoints/pix2pix_model")
loaded_model = Pix2Pix.load("./checkpoints/pix2pix_model")

