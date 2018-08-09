import numpy as np
import scipy.io as sio
import os
def read_feature_file(filepath):
    try:
        mat = sio.loadmat(filepath)
        features = mat['fea']
        features = np.asarray(features).flatten()
        return features
    except:
        print(str(Exception))
        return None


def get_features(list_images, label_names, feature_dir):
    features_list = []
    for i, image_path in enumerate(list_images):
        label_name = label_names[i]
        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.jpg')[0]
        prefix = os.path.join(feature_dir, label_name, image_name)
        # print('prefix: ', prefix)
        features = read_feature_file(prefix+".mat")
        features_list.append(features)
    print('loaded handcrafted with shape: ', np.asarray(features_list).shape)
    return np.asarray(features_list)