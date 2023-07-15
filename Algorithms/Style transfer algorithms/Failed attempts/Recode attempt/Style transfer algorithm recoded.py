import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from scipy.optimize import fmin_l_bfgs_b as lbfgs
from PIL import Image
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input as preprocess
from keras.applications.vgg19 import VGG19
import keras.backend as K

tf.enable_eager_execution()
print(tf.executing_eagerly())

### Main Variables###

content_image_path = "/Users/adithyanarayanan/Documents/High School/Grade 10/Non Subject related/Personal Project/Code/Assets/San Fransisco skyline/San Fransisco skyline.png"
style_image_path = "/Users/adithyanarayanan/Documents/High School/Grade 10/Non Subject related/Personal Project/Code/Assets/San Fransisco skyline/Vision of Warsaw, Poland.png"

content_layers, style_layers = (["block5_conv2"], ["block1_conv1",
                                                   "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"])

### End of Main Variables###

### Helper function taken from https://colab.research.google.com/github/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb#scrollTo=mjzlKRQRs_y2###


def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

###End of helper function###


def preprocessing_input_image(path_to_img):
    image = load_img(path_to_img)
    image = preprocess(image)
    return image


def deprocessing_input_image(previously_processed_image):
    if len(previously_processed_image.shape) == 4:
        np.squeeze(previously_processed_image, axis=0)
    previously_processed_image[:, :, 0] += 103.939
    previously_processed_image[:, :, 1] += 116.779
    previously_processed_image[:, :, 2] += 123.68
    deprocessed_input_image = np.clip(
        previously_processed_image[:, :, :, ::-1], 0, 255).astype(uint8)
    return deprocessed_input_image


def load_VGG_model():
    VGG19_model = VGG19(include_top=false, weights="imagenet")
    VGG19_model.trainable = False
    for every_layer in style_layers:
        Style_layer_output = VGG19_model.get_layer[every_layer].output
        content_layer_output = VGG19_model.get_layer[every_layer].output
    Total_output = (style_layer_output, content_layer_output)
    return models.Model(VGG19_model.input, Total_output)

### LOSS FUNCTIONS ####


def content_loss(content_img, activations_content_l):
    content_cost = tf.reduce_mean(tf.square(content_img - activations_content_l))
    return content_cost


def total_variation_loss(x):
    assert K.ndim(x) == 4
    a = tf.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, 1:, :img_height-1])
    b = tf.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, :img_width-1, 1:])
    return tf.sum(tf.pow(a + b, 1.25))
