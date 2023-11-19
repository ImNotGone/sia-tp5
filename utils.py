from typing import Any, Dict, List
import json
import numpy as np
from PIL import Image

def get_batch_size(config: Dict[str, Any], data_size: int) -> int:
    strategy = config["training_strategy"]

    if strategy == "batch":
        return data_size
    elif strategy == "mini_batch":
        batch_size = config["batch_size"]

        if batch_size > data_size:
            raise ValueError("Batch size must be smaller than the dataset size")

        return batch_size
    elif strategy == "online":
        return 1
    else:
        raise ValueError("Invalid training strategy")


def pretty_print_font(bitmap):
    delta = 0.1
    for row in bitmap:
        for pixel in row:
            if pixel < delta:
                print(" ", end="")
            else:
                print("█", end="")
        print()

def create_image(fonts, path, size=(7, 5)):
    # Add a border around each font, 1 pixel wide
    fonts = np.pad(fonts, ((0, 0), (1, 1), (1, 1)), "constant", constant_values=0)

    bitmap_height, bitmap_width = len(fonts[0]), len(fonts[0][0])
    cols, rows = size
    composite_width = cols * bitmap_width
    composite_height = rows * bitmap_height

    composite_image = Image.new('L', (composite_width, composite_height))

    for i, font in enumerate(fonts):
        font_image = Image.fromarray(font * 255)
        x = i % cols
        y = i // cols
        composite_image.paste(font_image, (x * bitmap_width, y * bitmap_height))

    # Add a border around each image

    # Scale the image up
    composite_image = composite_image.resize((composite_width * 10, composite_height * 10))

    composite_image.save(path)
    composite_image.show()
   
    

def serialize_weights(weights, path="weights.json"):
    # Convert the weights to a list of lists
    weights = [[w.tolist() for w in layer] for layer in weights]

    with open(path, "w") as f:
        json.dump(weights, f)


def deserialize_weights(path="weights.json") -> List[np.ndarray]:
    with open(path, "r") as f:
        weights = json.load(f)

        # Convert the weights to numpy arrays
        weights = [[np.array(w) for w in layer] for layer in weights]
        weights = [np.array(layer) for layer in weights]

        return weights
