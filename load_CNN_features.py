import os
# /tmp/bottleneck/Nucleus/r06aug97.gidap.12--1---2.dat.jpg_inception_resnet_v2.txt
import numpy as np
def read_feature_file(filepath):
    with open(filepath, 'r') as feature_file:
        feature_string = feature_file.read()
    try:
        feature_values = [float(x) for x in feature_string.split(',')]
    except ValueError:
        print('Invalid float found')
    return np.asarray(feature_values)

def get_features(list_images, label_names, feature_dir, feature_type='concat'):
    features_list = []
    for i, image_path in enumerate(list_images):
        label_name = label_names[i]
        image_name = image_path.split('/')[-1]
        prefix = os.path.join(feature_dir, label_name, image_name)

        inception_features = read_feature_file(prefix+"_inception_v3.txt")
        resnet_features = read_feature_file(prefix+"_resnet_v2.txt")
        inception_resnet_features = read_feature_file(prefix+"_inception_resnet_v2.txt")
        if feature_type == 'concat':
            features= np.concatenate((inception_features, resnet_features, inception_resnet_features))
            assert (features.shape  == ((2048 *2 + 1536),)) # if no exception -> correct
            # print (features.shape)
        elif feature_type == 'inception_v3':
            features = inception_features
        elif feature_type == 'resnet_v2':
            features = resnet_features
        elif feature_type == 'inception_resnet_v2':
            features = inception_resnet_features
        else:
            raise Exception
        features_list.append(features)
    print('loaded cnn features with shape: ', np.asarray(features_list).shape)
    return np.asarray(features_list)


def main():
    pass

if __name__ == '__main__':
    main()


