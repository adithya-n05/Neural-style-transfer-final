import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
import keras.applications.vgg19 as CNN_network
from keras.applications.vgg19 import VGG19
from keras import backend as K

content_img_path = "/content/stata.jpg"
style_img_path = "/content/Hokusai's the great wave.jpg"
epochs_or_iterations_whichever_idk = 100
save_directory = "/generated_img"

t_loss = K.variable(0.0)

c_weight = 0.25
s_weight = 0.75
tv_weight = 1.0


s_g_features = ['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1',
                'block5_conv1']

img_height = np.shape(load_img(content_img_path))[0]
img_width = np.shape(load_img(content_img_path))[1]

print("The content image is of size " + str(img_height) + " by " + str(img_width) +
      " pixels. Adjust the generated image size accordingly.")

output_img_height = 400
output_img_width = int((output_img_height/img_height) * img_width)

print("The generated image will be of size " +
      str(output_img_height) + " by " + str(output_img_width) + ".")

generated_img = K.placeholder((1, output_img_height, output_img_width, 3))


def preprocess_img(preprocess_input_img_path):
    preprocess_input_img = load_img(preprocess_input_img_path,
                                    target_size=(output_img_height, output_img_width))
    preprocess_input_img = img_to_array(preprocess_input_img)
    preprocess_input_img = np.expand_dims(preprocess_input_img, axis=0)
    preprocess_input_img = CNN_network.preprocess_input(preprocess_input_img)
    return preprocess_input_img


def deprocess_img(output_img):
    output_img = output_img.reshape((output_img_height, output_img_width, 3))
    output_img[:, :, 0] += 103.939
    output_img[:, :, 1] += 116.779
    output_img[:, :, 2] += 123.68
    output_img = output_img[:, :, ::-1]
    output_img = np.clip(output_img, 0, 255).astype('uint8')
    return output_img


def create_img_tensors(content_img_path, style_img_path):
    content_img_tensor = K.variable(preprocess_img(content_img_path))
    style_img_tensor = K.variable(preprocess_img(style_img_path))
    return (content_img_tensor, style_img_tensor)


def input_to_VGG(content_img_path, style_img_path, generated_img):
    content_img_tensor = K.variable(create_img_tensors(content_img_path, style_img_path)[0])
    style_img_tensor = K.variable(create_img_tensors(content_img_path, style_img_path)[1])
    VGG_input = K.concatenate([content_img_tensor, style_img_tensor, generated_img], axis=0)
    return VGG_input


VGG_input = input_to_VGG(content_img_path, style_img_path, generated_img)


def build_vgg(VGG_input):
    vgg_model = VGG19(input_tensor=VGG_input, weights="imagenet", include_top=False)
    return vgg_model


VGG_model = build_vgg(VGG_input)

layer_activations = dict([(every_layer.name, every_layer.output)
                          for every_layer in VGG_model.layers])

c_g_features = layer_activations['block5_conv2']


def gm(input_tensor):
    assert K.ndim(input_tensor) == 3
    features = K.batch_flatten(K.permute_dimensions(input_tensor, (2, 0, 1)))
    gm_matrix = K.dot(features, K.transpose(features))
    return gm_matrix


def s_loss(style_img_features, generated_img_features):
    assert K.ndim(style_img_features) == 3
    assert K.ndim(generated_img_features) == 3
    generated = gm(generated_img_features)
    style = gm(style_img_features)
    style_loss = K.sum(K.square(style - generated)) / (4.0 * (3 ** 2)
                                                       * ((output_img_height*output_img_width) ** 2))
    return style_loss


def c_loss(content_img_features, generated_img_features):
    content_loss = K.sum(K.square(generated_img_features - content_img_features))
    return content_loss


def tv_loss(input_img):
    val1 = K.square(
        input_img[:, :output_img_height - 1, :output_img_width - 1, :] - input_img[:, 1:, :output_img_width - 1, :])
    val2 = K.square(
        input_img[:, :output_img_height - 1, :output_img_width - 1, :] - input_img[:, :output_img_height - 1, 1:, :])
    val3 = K.sum(K.pow(val1 + val2, 1.25))
    return val3


content_img_features = c_g_features[0, :, :, :]
generated_img_features = c_g_features[2, :, :, :]
t_loss = t_loss + (c_weight * c_loss(content_img_features,
                                     generated_img_features))

for every_layer in s_g_features:
    singular_layer_features = layer_activations[every_layer]
    style_minimisation_features = singular_layer_features[1, :, :, :]
    generated_minimisation_features = singular_layer_features[2, :, :, :]
    style_loss = s_loss(style_minimisation_features, generated_minimisation_features)
    t_loss = t_loss + ((s_weight/len(s_g_features)) * style_loss)


t_loss = t_loss + (tv_weight * tv_loss(generated_img))

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

content_img = preprocess_img(content_img_path)

for iter in range(epochs_or_iterations_whichever_idk):
    print("Iteration " + str(iter))
    content_img, min_val, info = fmin_l_bfgs_b(
        evaluator.loss, content_img.flatten(), fprime=evaluator.grads, maxfun=20)
    print("Loss at end of iteration " + str(iter) + " = " + str(min_val))
    deprocessed_img = deprocess_img(content_img.copy())
    img_save = (save_directory + "_{}.jpg".format(iter))
    save_img(img_save, deprocessed_img)
