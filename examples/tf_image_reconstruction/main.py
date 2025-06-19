from xflow import FileProvider, TensorFlowPipeline, BaseTrainer, ConfigManager, BaseModel
from xflow.data import build_transforms_from_config
from xflow.utils import get_base_dir, plot_image, load_validated_config
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
provider = FileProvider(base / config["data"]["root"], path_type="string").subsample(500)
train_provider, temp_provider = provider.split(train_ratio=config["data"]["first_split"], seed=config["seed"])
val_provider, test_provider = temp_provider.split(train_ratio=config["data"]["second_split"], seed=config["seed"])

transforms = build_transforms_from_config(config['data']['transforms']['tf_native']) 
def make_dataset(provider):
    return (
        TensorFlowPipeline(provider, transforms)
        .to_framework_dataset(config['framework']['name'], config["data"]["dataset_ops"])
        .cache()
    )
train_dataset = make_dataset(train_provider)
val_dataset   = make_dataset(val_provider)
test_dataset  = make_dataset(test_provider)

for re, inp in test_dataset.take(1):
    print(f"input sample shape: {inp.shape}, label sample shape: {re.shape}")
    plot_image(inp)
    plot_image(re)

# ====================================
# Model and Trainer
# ====================================
model = BaseModel(config["model"])

trainer = BaseTrainer(
    model=model,
    data_pipeline=train_dataset,
    config=config,
    output_dir=base / config["paths"]["output"]
)

trainer.fit()
trainer.save_model(base / config["paths"]["output"] / "model.h5")