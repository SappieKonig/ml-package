import os
from typing import Generator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_images_from_folder(folder: str) -> Generator[str, None, None]:
    """Get all images from a folder."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                yield os.path.join(root, file)


class ImageShower:
    """
    Class that takes an generator, and shows rows x cols images in a grid.
    Supposed to be used in jupyter notebooks.
    """
    def __init__(self, images: Generator[str, None, None], rows: int = 3, cols: int = 3):
        self.images = images
        self.rows = rows
        self.cols = cols

    def show(self):
        """Display the list of image paths in a grid."""
        fig, ax = plt.subplots(self.rows, self.cols, figsize=(15, 15))
        for i in range(self.rows):
            for j in range(self.cols):
                try:
                    image = next(self.images)
                    img = mpimg.imread(image)
                    ax[i, j].imshow(img)
                    ax[i, j].axis('off')
                    filename = os.path.basename(image)
                    shape = list(img.shape)
                    if len(shape) == 2:
                        resolution = f"{img.shape[0]}x{img.shape[1]}"
                    else:
                        resolution = f"{img.shape[0]}x{img.shape[1]}x{img.shape[2]}"
                    ax[i, j].set_title(f"{filename}\n{resolution}", fontsize=10)
                except StopIteration:
                    break
        plt.tight_layout()
        plt.show()
