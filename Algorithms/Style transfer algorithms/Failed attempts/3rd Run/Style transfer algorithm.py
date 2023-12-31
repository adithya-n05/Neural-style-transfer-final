import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

content_image = scipy.misc.imread(
    "/Users/adithyanarayanan/Documents/High School/Grade 10/Non Subject related/Personal Project/Code/Assets/Man walking away from flames/Man Walking Away.g")
plt.imshow(content_image)
content_image = reshape_and_normalize_image(content_image)


style_image = scipy.misc.imread(
    "/Users/adithyanarayanan/Documents/High School/Grade 10/Non Subject related/Personal Project/Code/Assets/James Bond/Hokusai's the great wave.jpg")
plt.imshow(style_image)
style_image = reshape_and_normalize_image(style_image)

STYLE_LAYERS = [("conv1_1", 1.0), ("conv1_2", 1.0), ("conv2_1", 1.0),
                ("conv3_2", 0.5), ("conv3_3", 0.2)]

### MAIN FUNCTIONS ###


def compute_content_cost(a_C, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, [n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [n_H * n_W, n_C])

    J_content = (tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)))/(4 * n_H * n_W * n_C)

    return J_content


def gram_matrix(A):

    GA = tf.matmul(A, tf.transpose(A))

    return GA


def compute_layer_style_cost(a_S, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = (1./(4 * n_C**2 * (n_H*n_W)**2)) * tf.reduce_sum(tf.pow((GS - GG), 2))

    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS):

    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        out = model[layer_name]

        a_S = sess.run(out)

        a_G = out

        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=30, beta=40):

    J = (alpha * J_content) + (beta * J_style)

    return J


tf.reset_default_graph()

sess = tf.InteractiveSession()

generated_image = generate_noise_image(content_image)
plt.imshow(generated_image[0])

model = load_vgg_model(
    "/Users/adithyanarayanan/Documents/High School/Grade 10/Non Subject related/Personal Project/Code/imagenet-vgg19-verydeep-19.mat")

sess.run(model['input'].assign(content_image))

out = model['conv5_4']

a_C = sess.run(out)

a_G = out

J_content = compute_content_cost(a_C, a_G)

sess.run(model['input'].assign(style_image))

J_style = compute_style_cost(model, STYLE_LAYERS)

J = total_cost(J_content, J_style, alpha=1.5, beta=0.0005)

learning_rate_value = 5.0

optimizer = tf.train.AdamOptimizer(learning_rate_value)

train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations=200):

    sess.run(tf.global_variables_initializer())

    sess.run(model["input"].assign(input_image))

    for i in range(num_iterations):

        sess.run(train_step)

        generated_image = sess.run(model["input"])

        Jt, Jc, Js = sess.run([J, J_content, J_style])
        print("Iteration " + str(i) + " :")
        print("total cost = " + str(Jt))
        print("content cost = " + str(Jc))
        print("style cost = " + str(Js))

        save_image("/Users/adithyanarayanan/Documents/High School/Grade 10/Non Subject related/Personal Project/Code/output/Style Transfer/Man walking away from flames/Attempt 1/Output_" +
                   str(i) + ".jpg", generated_image)

    save_image('/Users/adithyanarayanan/Documents/High School/Grade 10/Non Subject related/Personal Project/Code/output/Style Transfer/Man walking away from flames/Attempt 1/generated_image.jpg', generated_image)

    return generated_image


model_nn(sess, generated_image, num_iterations=2000)
