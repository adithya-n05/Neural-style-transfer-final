import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
import keras.applications.vgg19 as CNN_network
from keras.applications.vgg19 import VGG19
from keras import backend as K


content_img_path = "/content/brad_pitt.jpg"
style_img_path = "/content/Picasso Self Portrait 1907.jpg"
epochs_or_iterations_whichever_idk = 100
save_directory = "/content/MY_Out"

# these are the weights of the different loss components
c_weight = 0.025
s_weight = 1.0
tv_weight = 1.0

# dimensions of the generated picture.
img_height = np.shape(load_img(content_img_path))[0]
img_width = np.shape(load_img(content_img_path))[1]

print("The content image is of size " + str(img_height) + " by " + str(img_width) +
      " pixels. Adjust the generated image size accordingly.")

output_img_height = 400
output_img_width = int((output_img_height/img_height) * img_width)

print("The generated image will be of size " +
      str(output_img_height) + " by " + str(output_img_width) + ".")

generated_img = K.placeholder((1, output_img_height, output_img_width, 3))

# util function to open, resize and format pictures into appropriate tensors


def preprocess_img(preprocess_input_img_path):
    preprocess_input_img = load_img(preprocess_input_img_path,
                                    target_size=(output_img_height, output_img_width))
    preprocess_input_img = img_to_array(preprocess_input_img)
    preprocess_input_img = np.expand_dims(preprocess_input_img, axis=0)
    preprocess_input_img = CNN_network.preprocess_input(preprocess_input_img)
    return preprocess_input_img

# util function to convert a tensor into a valid image


def deprocess_img(output_img):
    output_img = output_img.reshape((output_img_height, output_img_width, 3))
    output_img[:, :, 0] += 103.939
    output_img[:, :, 1] += 116.779
    output_img[:, :, 2] += 123.68
    output_img = output_img[:, :, ::-1]
    output_img = np.clip(output_img, 0, 255).astype('uint8')
    return output_img

# get tensor representations of our images


def create_img_tensors(content_img_path, style_img_path):
    content_img_tensor = K.variable(preprocess_img(content_img_path))
    style_img_tensor = K.variable(preprocess_img(style_img_path))
    return (content_img_tensor, style_img_tensor)


# this will contain our generated image
generated_img = K.placeholder((1, output_img_height, output_img_width, 3))

# combine the 3 images into a single Keras tensor


def input_to_VGG(content_img_path, style_img_path, generated_img):
    content_img_tensor = K.variable(create_img_tensors(content_img_path, style_img_path)[0])
    style_img_tensor = K.variable(create_img_tensors(content_img_path, style_img_path)[1])
    VGG_input = K.concatenate([content_img_tensor, style_img_tensor, generated_img], axis=0)
    return VGG_input


VGG_input = input_to_VGG(content_img_path, style_img_path, generated_img)

# build the VGG19 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights


def build_vgg(VGG_input):
    vgg_model = VGG19(input_tensor=VGG_input, weights="imagenet", include_top=False)
    return vgg_model


VGG_model = build_vgg(VGG_input)

# get the symbolic outputs of each "key" layer (we gave them unique names).

layer_activations = dict([(every_layer.name, every_layer.output)
                          for every_layer in VGG_model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)


def gm(input_tensor):
    assert K.ndim(input_tensor) == 3
    features = K.batch_flatten(K.permute_dimensions(input_tensor, (2, 0, 1)))
    gm_matrix = K.dot(features, K.transpose(features))
    return gm_matrix

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def s_loss(style_img_features, generated_img_features):
    assert K.ndim(style_img_features) == 3
    assert K.ndim(generated_img_features) == 3
    generated = gm(generated_img_features)
    style = gm(style_img_features)
    style_loss = K.sum(K.square(style - generated)) / (4.0 * (3 ** 2)
                                                       * ((output_img_height*output_img_width) ** 2))
    return style_loss

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def c_loss(content_img_features, generated_img_features):
    content_loss = K.sum(K.square(generated_img_features - content_img_features))
    return content_loss

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def tv_loss(input_img):
    val1 = K.square(
        input_img[:, :output_img_height - 1, :output_img_width - 1, :] - input_img[:, 1:, :output_img_width - 1, :])
    val2 = K.square(
        input_img[:, :output_img_height - 1, :output_img_width - 1, :] - input_img[:, :output_img_height - 1, 1:, :])
    val3 = K.sum(K.pow(val1 + val2, 1.25))
    return val3


# combine these loss functions into a single scalar
t_loss = K.variable(0.0)
c_g_features = layer_activations['block5_conv2']
content_img_features = c_g_features[0, :, :, :]
generated_img_features = c_g_features[2, :, :, :]
t_loss = t_loss + (c_weight * c_loss(content_img_features,
                                     generated_img_features))

s_g_features = ['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1',
                'block5_conv1']
for every_layer in s_g_features:
    singular_layer_features = layer_activations[every_layer]
    style_minimisation_features = singular_layer_features[1, :, :, :]
    generated_minimisation_features = singular_layer_features[2, :, :, :]
    style_loss = s_loss(style_minimisation_features, generated_minimisation_features)
    t_loss = t_loss + ((s_weight/len(s_g_features)) * style_loss)

t_loss = t_loss + (tv_weight * tv_loss(generated_img))

# get the gradients of the generated image wrt the loss
gradients = K.gradients(t_loss, generated_img)

model_outputs = [t_loss]
if isinstance(gradients, (list, tuple)):
    model_outputs = model_outputs + gradients
else:
    model_outputs.append(gradients)


calculated_outputs = K.function([generated_img], model_outputs)


def grads_eval(input_img):
    input_img = input_img.reshape(1, output_img_height, output_img_width, 3)
    outputs = calculated_outputs([input_img])
    loss_val = outputs[0]
    if len(outputs[1:]) == 1:
        gradient_val = outputs[1].flatten().astype('float64')
    else:
        gradient_val = np.array(outputs[1:]).flatten().astype("float64")
    return loss_val, gradient_val

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

    def __init__(self):
        self.loss_val = None
        self.gradient_val = None

    def loss(self, input_img):
        assert self.loss_val is None
        loss_val, gradient_val = grads_eval(input_img)
        self.loss_val = loss_val
        self.gradient_val = gradient_val
        return self.loss_val

    def grads(self, input_img):
        assert self.loss_val is not None
        gradient_val = np.copy(self.gradient_val)
        self.loss_val = None
        self.gradient_val = None
        return gradient_val


evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
content_img = preprocess_img(content_img_path)

for i in range(epochs_or_iterations_whichever_idk):
    print('Start of iteration', i)
    content_img, min_val, info = fmin_l_bfgs_b(
        evaluator.loss, content_img.flatten(), fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_img(x.copy())
    fname = save_directory + '_at_iteration_%d.png' % i
    save_img(fname, img)
    print('Image saved as', fname)
