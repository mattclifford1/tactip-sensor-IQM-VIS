from skimage.transform import resize, rotate
import cv2
import os
import numpy as np


def load_image(image_path):
    return cv2.imread(image_path)

def load_real_image(image_path):
    image = cv2.imread(image_path)
    # Convert to gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def crop_to_square(image):
    # crop to square
    x, y = image.shape
    cut = (y - x)//2
    image = image[:, cut:cut+x]
    # downscale
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    # Add channel axis
    image = image[..., np.newaxis]
    image = np.concatenate([image, image, image], axis=2)
    return image.astype(np.float32) / 255.0

def process_im(image, data_type='sim'):
    if data_type == 'real':
        # Crop to specified bounding box
        bbox = [80,25,530,475]
        x0, y0, x1, y1 = bbox
        image = image[y0:y1, x0:x1]
        # Resize to specified dims
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        # Add channel axis
        image = image[..., np.newaxis]
        # threshold_image
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -30)
        image = image[..., np.newaxis]
    elif data_type == 'sim':
        # Convert to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Add channel axis
        image = image[..., np.newaxis]
    image = np.concatenate([image, image, image], axis=2) 
    return image.astype(np.float32) / 255.0

def get_real_csv_given_sim(sim_csv):
    ''' try find equivelant real csv given a sim
        only works for standard dir structing
        will return sim_csv if can't find real csv'''
    dirs = sim_csv.split(os.sep)
    dirs[0] = os.sep
    dirs.pop(-3)       # remove 128x128
    dirs[-5] = 'real'  # swap sim for real
    real_csv = os.path.join(*dirs)
    if os.path.isfile(real_csv):
        return real_csv
    else:
        print(f'Cannot find real csv dir: {real_csv}')
        return sim_csv


if __name__ == '__main__':
    csv_file_sim = '/home/matt/summer-project/data/Bourne/tactip/sim/edge_2d/shear/128x128/csv_train/targets.csv'
    print(get_real_csv_given_sim(csv_file_sim))

    # dev image transformations
    image_file = '/home/matt/summer-project/data/Bourne/tactip/sim/edge_2d/shear/128x128/csv_train/images/image_6.png'
    image = load_image(image_file)
    params_dict = {'rotation':90, 'zoom_factor':.75, 'x_shift': 0.2, 'y_shift':0.1}
    params_dict = {'x_shift': 0.3, 'y_shift':-0.2}
    image_trans = transform_image(image, params_dict)

    # show images
    images_list = [image, image_trans]
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(images_list))
    fig.suptitle('image transformations')
    for i, im in enumerate(images_list):
        axs[i].imshow(im)
    plt.show()
