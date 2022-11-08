from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io

def image_grid(image, idx):

    """
    Return a 5x5 grid of the Hyperspectral band images as a matplotlib figure.
    image: CWH, here C > 0
    """
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i + 1, title=f"idx:{idx} band no.:{i}")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image[i].detach().cpu().numpy(), cmap=plt.cm.binary)
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    # image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    # image = tf.expand_dims(image, 0)
    
    image = np.array(Image.open(buf))
    # image = torchvision.io.decode_png(buf.getvalue())
    return image