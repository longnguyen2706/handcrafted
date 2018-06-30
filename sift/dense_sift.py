import cyvlfeat as vl
from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data


img_data = load_image('/home/long/Desktop/gray.jpg')
frame, desc = vl.sift.dsift(img_data)
print (frame)
print(desc)