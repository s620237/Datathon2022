from __future__ import print_function
from google.cloud import vision
from pathlib import Path
from natsort import natsorted
import io

def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    for index, text in enumerate(texts):
        if index == 0:
            print(text.description)
        
if __name__ == "__main__":
    dir = "training-strips"
    index = 1
    filelist = Path(dir).glob('*')
    for file in natsorted(filelist):
        detect_text(file)
        index += 1