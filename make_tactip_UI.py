import os
import numpy as np
import IQM_VIS
import image_utils


def run():
    # metrics functions must return a single value
    metric = {'MAE': IQM_VIS.metrics.MAE,
              'MSE': IQM_VIS.metrics.MSE,
              '1-SSIM': IQM_VIS.metrics.ssim()}

    # metrics images return a numpy image
    metric_images = {'MSE': IQM_VIS.metrics.MSE_image,
                     'SSIM': IQM_VIS.metrics.SSIM_image()}

    data = custom_image_handler(get_image_list(),
                                  image_utils.load_and_crop_raw_real,
                                  metric,
                                  metric_images)
    # define the transformations
    transformations = {
               'rotation':{'min':-180, 'max':180, 'function':IQM_VIS.transforms.rotation},    # normal input
               'blur':{'min':1, 'max':41, 'normalise':'odd', 'function':IQM_VIS.transforms.blur},  # only odd ints
               'brightness':{'min':-1.0, 'max':1.0, 'function':IQM_VIS.transforms.brightness},   # normal but with float
               }

    # use the API to create the UI
    IQM_VIS.make_UI(data,
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


class custom_image_handler(IQM_VIS.dataset_holder):
    '''
    modify the image handler for the data API to give a difference reference image
    '''
    def __init__(self, image_list: list, # list of image file names
                       image_loader,     # function to load image files
                       metrics: dict,
                       metric_images: dict):
        super().__init__(image_list, image_loader, metrics, metric_images)

    def get_reference_image(self):
        processed_im = image_utils.process_im(self.image_to_transform[1], data_type='real')
        return processed_im

if __name__ == '__main__':
    run()
