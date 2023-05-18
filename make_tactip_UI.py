import os
import numpy as np
import IQM_Vis
import image_utils


def run():
    # metrics functions must return a single value
    metric = {'MAE': IQM_Vis.metrics.MAE(),
              'MSE': IQM_Vis.metrics.MSE(),
              '1-SSIM': IQM_Vis.metrics.SSIM()}

    # metrics images return a numpy image
    metric_images = {'MSE': IQM_Vis.metrics.MSE(return_image=True),
                     'SSIM': IQM_Vis.metrics.SSIM(return_image=True)}

    data = custom_image_handler(get_image_list(),
                                  image_utils.load_real_image,
                                  metric,
                                  metric_images)
    # define the transformations
    transformations = {
        'rotation':{'min':-10, 'max':10, 'function':IQM_Vis.transforms.rotation},    # normal input
        'blur':{'min':1, 'max':41, 'normalise':'odd', 'function':IQM_Vis.transforms.blur},  # only odd ints
        'brightness':{'min':-1.0, 'max':1.0, 'function':IQM_Vis.transforms.brightness},   # normal but with float
        'x_shift':{'min':-0.05, 'max':0.05, 'function':IQM_Vis.transforms.x_shift, 'num_values':41},
        'y_shift':{'min':-0.05, 'max':0.05, 'function':IQM_Vis.transforms.y_shift, 'num_values':41},
        'zoom':{'min':0.9, 'max':1.1, 'function':IQM_Vis.transforms.zoom_image, 'init_value': 1, 'num_values':41},  # requires non standard slider params
        'threshold':{'min':-40, 'max':40, 'function':IQM_Vis.transforms.binary_threshold, 'num_values':41},
        }

    # use the API to create the UI
    IQM_Vis.make_UI(data,
                transformations,
                metrics_avg_graph=True)


def get_image_list():
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data',
                             'real',
                             'surface_3d',
                             'tap',
                             'csv_train',
                             'images')
    image_paths = [os.path.join(image_dir, 'image_'+str(i)+'.png') for i in range(1, 101)]
    return image_paths


class custom_image_handler(IQM_Vis.dataset_holder):
    '''
    modify the image handler for the data API to give a difference reference image
    '''
    def __init__(self, image_list: list, # list of image file names
                       image_loader,     # function to load image files
                       metrics: dict,
                       metric_images: dict):
        super().__init__(image_list, metrics, metric_images, image_loader)

    def _load_image_data(self, i):
        # add to this method since the reference and transform image require different preprocessing
        super()._load_image_data(i)
        # apply different cropping
        image = self.image_loader(self.current_file)
        self.image_reference = self.image_storer(self.image_reference.name, image_utils.process_im(image, data_type='real'))
        self.image_to_transform = self.image_storer(self.image_to_transform.name, image_utils.crop_to_square(image))

    # def get_reference_image(self):
    #     # preprocess the reference image
    #     processed_im = image_utils.process_im(self.image_to_transform[1], data_type='real')
    #     return processed_im

if __name__ == '__main__':
    run()
