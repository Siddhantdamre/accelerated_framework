import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

def compute_gradcam(model, image, last_conv_layer_name):
    # Create a model that maps input to the activations of the last conv layer + output
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for the input image
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply pooled grads with activations
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

def display_gradcam(image, heatmap, alpha=0.4):
    # Rescale the heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)

    # Apply the heatmap on the image
    jet = plt.cm.get_cmap("jet")
    jet_heatmap = jet(heatmap)[:, :, :3] * 255
    overlayed_image = jet_heatmap * alpha + image

    # Display the image with the Grad-CAM overlay
    plt.imshow(overlayed_image.astype("uint8"))
    plt.axis("off")
    plt.show()
