# Libraries
import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import VGG19
from keras import backend as K

# Paths
content_img_path = "/content/MIT garden area.jpg"
style_img_path = "/content/Mads Berg Family.jpg"
epochs_or_iterations_whichever_idk = 100
save_directory = "/content/candy"

# Weights
c_weight = 0.02
s_weight = 3200.0
tv_weight = 400.0

# Image size configurations
img_height = np.shape(load_img(content_img_path))[0]
img_width = np.shape(load_img(content_img_path))[1]

print("The content image is of size " + str(img_height) + " by " + str(img_width) +
      " pixels. Adjust the generated image size accordingly.")

output_img_height = 400
output_img_width = int((output_img_height/img_height) * img_width)

print("The generated image will be of size " +
      str(output_img_height) + " by " + str(output_img_width) + ".")

# Main functions


def add_vgg_mean(img):
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    return img


def subtract_vgg_mean(img):
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img


def preprocess_img(preprocess_input_img_path):
    preprocess_input_img = load_img(preprocess_input_img_path,
                                    target_size=(output_img_height, output_img_width))
    preprocess_input_img = img_to_array(preprocess_input_img)
    preprocess_input_img = preprocess_input_img[:, :, ::-1]
    preprocess_input_img = subtract_vgg_mean(preprocess_input_img)
    preprocessed_input_img = np.expand_dims(preprocess_input_img, axis=0)
    return preprocessed_input_img


def deprocess_img(output_img):
    output_img = output_img.reshape((output_img_height, output_img_width, 3)).astype("float64")
    output_img = add_vgg_mean(output_img)
    output_img = output_img[:, :, ::-1]
    output_img = np.clip(output_img, 0, 255).astype('uint8')
    return output_img


def create_img_tensors(content_img_path, style_img_path):
    content_img_tensor = tf.Variable(preprocess_img(content_img_path))
    style_img_tensor = tf.Variable(preprocess_img(style_img_path))
    return (content_img_tensor, style_img_tensor)


def input_to_VGG(content_img_path, style_img_path, generated_img):
    content_img_tensor = tf.Variable(create_img_tensors(content_img_path, style_img_path)[0])
    style_img_tensor = tf.Variable(create_img_tensors(content_img_path, style_img_path)[1])
    VGG_input = tf.concat([content_img_tensor, style_img_tensor, generated_img], axis=0)
    return VGG_input


def build_vgg(VGG_input):
    vgg_model = VGG19(input_tensor=VGG_input, weights="imagenet", include_top=False)
    return vgg_model


def gm(input_tensor_features):
    gm_matrix = tf.matmul(input_tensor_features, tf.transpose(input_tensor_features))
    return gm_matrix


def s_loss(style_img_tensor, generated_img_tensor):
    generated_img_features = K.batch_flatten(K.permute_dimensions(generated_img_tensor, (2, 0, 1)))
    style_img_features = K.batch_flatten(K.permute_dimensions(style_img_tensor, (2, 0, 1)))
    generated = gm(generated_img_features)
    style = gm(style_img_features)
    style_loss = tf.reduce_sum(tf.square(style - generated)) / (4.0 * (3 ** 2)
                                                                * ((output_img_height*output_img_width) ** 2))
    return style_loss


def c_loss(content_img_features, generated_img_features):
    content_loss = tf.reduce_sum(tf.square(generated_img_features - content_img_features))
    return content_loss


def tv_loss(input_img):
    val1 = tf.square(
        input_img[:, :output_img_height - 1, :output_img_width - 1, :] - input_img[:, 1:, :output_img_width - 1, :])
    val2 = tf.square(
        input_img[:, :output_img_height - 1, :output_img_width - 1, :] - input_img[:, :output_img_height - 1, 1:, :])
    val3 = tf.reduce_sum(tf.pow(val1 + val2, 1.25))
    return val3

# Calls to functions


generated_img = K.placeholder((1, output_img_height, output_img_width, 3))

VGG_input = input_to_VGG(content_img_path, style_img_path, generated_img)

VGG_model = build_vgg(VGG_input)

t_loss = tf.Variable(0.0)


layer_activations = dict([(every_layer.name, every_layer.output)
                          for every_layer in VGG_model.layers])

s_g_features = [('block1_pool', 1.0), ('block2_pool', 1.0),
                ('block4_pool', 1.0), ('block3_conv3', 1.0)]

c_g_features = layer_activations['block4_conv2']

content_img_features = c_g_features[0, :, :, :]
generated_img_features = c_g_features[2, :, :, :]
t_loss = t_loss + (c_weight * c_loss(content_img_features,
                                     generated_img_features))

for every_layer, weight in s_g_features:
    singular_layer_features = layer_activations[every_layer]
    style_minimisation_features = singular_layer_features[1, :, :, :]
    generated_minimisation_features = singular_layer_features[2, :, :, :]
    style_loss = s_loss(style_minimisation_features, generated_minimisation_features)
    t_loss = t_loss + ((s_weight/len(s_g_features)) * weight * style_loss)


t_loss = t_loss + (tv_weight * tv_loss(generated_img))

model_outputs = [t_loss]
gradients = tf.gradients(t_loss, generated_img)

if isinstance(gradients, (list, tuple)):
    model_outputs = model_outputs + gradients
else:
    model_outputs.append(gradients)

calculated_outputs = K.function([generated_img], model_outputs)

# Gradient Descent evaluation


def grads_eval(input_img):
    input_img = input_img.reshape(1, output_img_height, output_img_width, 3)
    outputs = calculated_outputs([input_img])
    loss_val = outputs[0]
    if len(outputs[1:]) == 1:
        gradient_val = outputs[1].flatten().astype('float64')
    else:
        gradient_val = np.array(outputs[1:]).flatten().astype("float64")
    return loss_val, gradient_val

# Evaluator class


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

# Cost minimization via L-BFGS

for iter in range(epochs_or_iterations_whichever_idk):
    print("Iteration " + str(iter))
    content_img, min_val, info = fmin_l_bfgs_b(
        evaluator.loss, content_img.flatten(), fprime=evaluator.grads, maxfun=20)
    print("Loss at end of iteration " + str(iter) + " = " + str(min_val))
    deprocessed_img = deprocess_img(content_img.copy())
    img_save = (save_directory + "_{}.jpg".format(iter))
    save_img(img_save, deprocessed_img)
