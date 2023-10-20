import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


STAGE = "Training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    ## get ready the data

    PARENT_DIR = os.path.join(
    config["data"]["unzip_data_dir"],
    config["data"]["parent_data_dir"]
    )
    #print("??????",PARENT_DIR)
    train_data = os.path.join(
    config["data"]["unzip_data_dir"],
    config["data"]["parent_data_dir"],
    config["data"]["train_data"]
    )
    print("TRAINING DATA: ", train_data)
    print("---"*30)
    test_data = os.path.join(
    config["data"]["unzip_data_dir"],
    config["data"]["parent_data_dir"],
    config["data"]["test_data"]
    )
    print("TESTINF DATA: ", test_data)
    print("---"*30)

    params = config["params"]

    logging.info(f"read the data from {PARENT_DIR}")
    ## Training
    train_data_gen = ImageDataGenerator( 
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20,
        vertical_flip = True)
    train_generator = train_data_gen.flow_from_directory(
        directory=train_data,
        target_size=tuple(params["targer_size"]),
        color_mode=params["color_mode"],
        batch_size=params["batch_size"],
        class_mode=params["class_mode"],  # Use 'categorical' for multi-class classification
        subset='training',
        shuffle=True,
        seed=params["seed"])
    
    
    #Validation
    validation_generator = train_data_gen.flow_from_directory(
        directory=train_data,
        target_size=tuple(params["targer_size"]),
        color_mode=params["color_mode"],
        batch_size=params["batch_size"],
        class_mode=params["class_mode"],  # Use 'categorical' for multi-class classification
        subset='validation',
        shuffle=True,
        seed=params["seed"])
    
    #Test
    test_data_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_data_gen.flow_from_directory(
        directory=test_data,
        target_size=tuple(params["targer_size"]),
        color_mode=params["color_mode"],
        batch_size=1,
        class_mode=params["class_mode"],  # Use 'categorical' for multi-class classification
        shuffle=False,
        seed=params["seed"])
    
    class_map = dict([(v,k) for k,v in train_generator.class_indices.items()])

    
    


    ## load the base model

    path_to_model = os.path.join(
        config["data"]["model_dir"], 
        config["data"]["init_model_file"])
    
    logging.info(f"load the base model from {path_to_model}")
    
    early_stop = os.path.join(
        config["data"]["model_dir"], 
        config["MODELS"]["early_stop_vgg16"])

    VGG16_classifier = tf.keras.models.load_model(path_to_model)
    
    #Early stop call back
    earlystop = tf.keras.callbacks.ModelCheckpoint(early_stop, 
                                     monitor='val_loss',
                                     patience=3, 
                                     save_best_only=False,
                                     save_weights_only=False, 
                                     restore_best_weights=True,
                                     mode='auto', 
                                     save_freq='epoch')
    #CSV logger call ack
    CSVfile = os.path.join(
        config["data"]["model_dir"], 
        config["data"]["CSV_file"])
    CSVLogger = tf.keras.callbacks.CSVLogger(CSVfile, 
                             separator=',', 
                             append=False)
    
    

    ## training
    logging.info(f"training started")

    hist = VGG16_classifier.fit_generator(train_generator,
                    validation_data = train_generator,
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = validation_generator.n//validation_generator.batch_size,
                    epochs=params["epochs"],
                    callbacks=[earlystop, CSVLogger])
    
    trained_model_file = os.path.join(
        config["data"]["model_dir"], 
        config["MODELS"]["trained_model_vg16"])

    VGG16_classifier.save(trained_model_file)
    logging.info(f"trained model is saved at : {trained_model_file}")
    
    # Plot the error and accuracy
    h = hist.history
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(h['loss'], c='red', label='Training Loss')
    plt.plot(h['val_loss'], c='red', linestyle='--', label='Validation Loss')
    plt.plot(h['accuracy'], c='blue', label='Training Accuracy')
    plt.plot(h['val_accuracy'], c='blue', linestyle='--', label='Validation Accuracy')
    plt.xlabel("Number of Epochs")
    plt.legend(loc='best')
    #plt.show()
    
    create_directories([config["data"]["graph_file"]])
    plt.savefig(os.path.join(config["data"]["graph_file"],'acc_val_plot_vgg16.jpg'), format='jpg')
    

    

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