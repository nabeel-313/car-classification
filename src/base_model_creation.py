import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.model import log_model_summary
import random
import tensorflow as tf

STAGE = "BASE MODEL CREATION"

logging.basicConfig(
    filename=os.path.join("logs", "running_logs1.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def main(config_path):
    config = read_yaml(config_path)
    params = config["params"]

    logging.info("layers defined")
    
    LAYERS = [
    tf.keras.layers.Input(shape=tuple(params["img_shape"])),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='same', activation="relu", ),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation="relu", ),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation="relu", ),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation="relu", ),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation="relu", ),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation="relu", ),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    
    
    tf.keras.layers.Dense(7, activation="softmax")
    
    ]
    
    classifier = tf.keras.Sequential(LAYERS)

    logging.info(f"base model summary:\n{log_model_summary(classifier)}")

    classifier.compile(
    optimizer=tf.keras.optimizers.Adam(params["lr"]),
    loss=params["loss"],
    metrics=params["metrics"]
    )
    
    model_dir = config["data"]["model_dir"] #create the dir for models
    create_directories([model_dir])
    
    path_to_model = os.path.join(
        model_dir, 
        config["data"]["init_model_file"])
    classifier.save(path_to_model)
    logging.info(f"model is saved at : {path_to_model}")
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
    