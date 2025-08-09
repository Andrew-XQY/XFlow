import os
from pathlib import Path

import tensorflow as tf
from regressor import Encoder
from tensorflow.keras.optimizers import Adam

from xflow import ConfigManager, DataPipeline, FileProvider
from xflow.data import build_transforms_from_config
from xflow.trainers import build_callbacks_from_config
from xflow.utils import get_base_dir, load_validated_config, plot_image

cur_dir = get_base_dir()
# ====================================
# Configuration
# ====================================
config_manager = ConfigManager(
    load_validated_config(os.path.join(cur_dir, "regressor.yaml"))
)
config = config_manager.get()
base = Path(config["paths"]["base"])

# ====================================
# Data pipeline
# ====================================
provider = FileProvider(base / config["data"]["root"])
train_provider, temp_provider = provider.split(
    ratio=config["data"]["first_split"], seed=config["seed"]
)
val_provider, test_provider = temp_provider.split(
    ratio=config["data"]["second_split"], seed=config["seed"]
)

transforms = build_transforms_from_config(config["data"]["transforms"]["numpy"])


def make_dataset(provider):
    return DataPipeline(provider, transforms).to_numpy()


train_dataset = make_dataset(train_provider)
val_dataset = make_dataset(val_provider)
test_dataset = make_dataset(test_provider)

for inp, params, re in zip(*test_dataset):
    print(f"input sample shape: {inp.shape}, label sample shape: {len(re)} \n{params}")
    plot_image(inp)
    plot_image(re)
    break

# ====================================
# Model
# ====================================
cbs = build_callbacks_from_config(
    config=config["callbacks"],
    framework=config["framework"],
)
cbs[-1].set_dataset(test_dataset)  # add dataset closure to the last callback

model = Encoder()
adam_optimizer = Adam(learning_rate=config["training"]["learning_rate"])
model.compile(optimizer=adam_optimizer, loss=config["training"]["loss"])
model.summary()

# ====================================
# Training
# ====================================
history = model.fit(
    train_dataset[0],
    train_dataset[1],  # x, y only
    epochs=config["training"]["epochs"],
    batch_size=config["training"]["batch_size"],
    validation_data=(val_dataset[0], val_dataset[1]),
    callbacks=[cbs],
)
config_manager.save(Path(config["paths"]["output"]) / config["name"])
