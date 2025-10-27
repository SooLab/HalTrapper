import base64
from PIL import Image
import io

from ._path import PathObj


def encode_image_to_base64(image_path: PathObj) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode("utf-8")


def decode_base64_to_image(base64_string: str, target_size: int = -1) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image
