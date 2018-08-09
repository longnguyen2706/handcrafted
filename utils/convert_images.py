import numpy as np
from glob import glob
import os
from PIL import Image
import scipy.misc

IMG_HEIGHT = 382
IMG_WIDTH = 382
TARGET_HEIGHT = 382 #299
TARGET_WIDTH = 382 #299
OUTPUT_DIR = "/home/long/Desktop/processed_image/"
IMAGE_EXTENSION='.tif'

def _read_and_resize_img(image_path, is_resize=False):
    image_name = image_path.split(IMAGE_EXTENSION)[0].split("/")[-1]
    image_class = image_path.split(IMAGE_EXTENSION)[0].split("/")[-2]
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    outdir = OUTPUT_DIR + image_class + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = outdir + image_name + ".jpg"
    print(outfile)
    if os.path.isfile(outfile):
        print("A jpeg file already exists for %s" % image_path.split('/')[-1])
        return None
    else:
        try:
            im = Image.open(image_path)
            print("Generating jpeg for %s" % image_path.split('/')[-1])
            # choose the correct format
            if IMAGE_EXTENSION == '.tif':
                im.mode = 'I'
                im = im.point(lambda i: i * (1. / 256)).convert('L')
            elif IMAGE_EXTENSION == '.BMP':
                pass
            elif IMAGE_EXTENSION == '.png':
                if not im.mode == 'RGB':
                    im = im.convert('RGB')

            # choose whether resize the imag
            if is_resize:
                im = _image_expasion(im, IMG_WIDTH, IMG_HEIGHT)
                im.thumbnail((TARGET_HEIGHT, TARGET_WIDTH))
            else:
                im.thumbnail(im.size)

            # convert to 3 channels and save
            im = convert_to_3_channels(im)
            im.save(outfile, "JPEG", quality=100)
        except Exception as e:
            print("Exception: " + str(e))
        return outfile


def _image_expasion(PIL_img, target_width, target_height):
    img_width = PIL_img.size[0]
    img_height = PIL_img.size[1]

    # expand the image to the target size. The new image will have the original image
    # in the center and the expasion pixels filled with black
    if (img_width <= target_width and img_height <= target_height):
        width_expansion = target_width - img_width
        height_expansion = target_height - img_height
        img_expanded = PIL_img.crop((int(-width_expansion / 2), int(-height_expansion / 2),
                                     int(img_width + width_expansion / 2), int(img_height + height_expansion / 2)))

        return img_expanded
    else:
        print("The original image has bigger size than the target size!")
        return PIL_img


def convert_to_3_channels(PIL_img):
    img_np = np.asarray(PIL_img)
    if len(img_np.shape) == 3:
        _, _, channel = img_np.shape
    else:
        channel = 1 # consider as gray

    if channel ==1:
        stacked_img = np.stack((img_np,) * 3, -1)
        img = Image.fromarray(stacked_img)
    else:
        img = PIL_img
    return img


patterns = "/mnt/6B7855B538947C4E/Stage_1_To_Long/image/hela/*/*"+IMAGE_EXTENSION
files = glob(patterns)

dirs = glob("/mnt/6B7855B538947C4E/Stage_1_To_Long/image/hela/*")

for i, f in enumerate(files):
    print(i)
    resized_image_dir = _read_and_resize_img(f)