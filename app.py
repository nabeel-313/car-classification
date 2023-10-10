import os
import argparse
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import logging
from src.utils.common import read_yaml, create_directories

STAGE = "Deploy using streamlit"


logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a")

def main(config_path):
    config = read_yaml(config_path)
    #data = config["data"]
    # test data path
    test_data = os.path.join(
        config["data"]["unzip_data_dir"],
        config["data"]["parent_data_dir"],
        config["data"]["test_data"]
        )
    #trained model path
    model_path = os.path.join(
        config["data"]["model_dir"], 
        config["data"]["trained_model_file"])
    
    class_map = {0: 'Audi',
                1: 'Hyundai Creta',
                2: 'Mahindra Scorpio',
                3: 'Rolls Royce',
                4: 'Swift',
                5: 'Tata Safari',
                6: 'Toyota Innova'}
    
    model = tf.keras.models.load_model(model_path)
    
    
    # Streamlit app
    st.title("Car Image Classifier")

    # Upload a new image
    uploaded_image = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Make predictions on the uploaded image
        test_img = image.load_img(uploaded_image, 
                             target_size=tuple(config["params"]["targer_size"]))
        test_img_arr = image.img_to_array(test_img)/255.0
        test_img_input = test_img_arr.reshape((1, test_img_arr.shape[0], test_img_arr.shape[1], test_img_arr.shape[2]))

        # 2. Make Predictions
        predicted_label = np.argmax(model.predict(test_img_input))
        #print(predicted_label)
        predicted_car = class_map[predicted_label]
        # Display the prediction
        st.title(f"Predicted Car Name: {predicted_car}")
        
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
    



