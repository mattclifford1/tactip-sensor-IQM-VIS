'''
generic image and metric data class constructor
'''
# Author: Matt Clifford <matt.clifford@bristol.ac.uk>
import os
import numpy as np
import pandas as pd

import sys; sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import image_utils
import networks

'''
extension of data_holder that allows to iterate through a dataset
'''
class dataset_holder:
    def __init__(self, data_set_csv,
                       metrics: dict,
                       metric_images: dict,
                       pose_path=os.path.join(os.path.expanduser('~'), 'summer-project/models/pose_estimation/surface_3d/shear/sim_LR:0.0001_BS:16/run_0/checkpoints/best_model.pth'),
                       sim=True):
        self.df = pd.read_csv(data_set_csv)
        self.sim = sim
        if self.sim:
            self.im_dir = os.path.join(os.path.dirname(data_set_csv), 'images')
        else:
            self.im_dir = os.path.join(os.path.dirname(image_utils.get_real_csv_given_sim(data_set_csv)), 'images')
        self._load_image_data(0)   # load the first image
        self.metrics = metrics
        self.metrics['pose_error'] = self._get_pose_error
        self.metric_images = metric_images
        self._check_inputs()
        self.pose_esimator_sim = networks.pose_estimation(pose_path, sim=self.sim)

    def _load_image_data(self, i):
        image = self.df.iloc[i]['sensor_image']
        image_path = os.path.join(self.im_dir, image)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if self.sim:
            raw_image_data = image_utils.load_image(image_path)
            self.image_reference = (image_name, image_utils.process_im(raw_image_data, data_type='sim'))
            self.image_to_transform = self.image_reference
        else:
            raw_image_data = image_utils.load_and_crop_raw_real(image_path)
            self.image_reference = (image_name, image_utils.process_im(raw_image_data, data_type='real'))
            self.image_to_transform = self.image_reference

        self._load_pose_data(i)

    def _load_pose_data(self, i):
        poses = ['pose_'+str(i) for i in range(1,7)]
        self.pose_data = {}
        for pose in poses:
            self.pose_data[pose] = self.df.iloc[i][pose]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        self._load_image_data(i)

    def get_reference_image_name(self):
        return self.image_reference[0]

    def get_reference_image(self):
        return self.image_reference[1]

    def get_transform_image_name(self):
        return self.image_to_transform[0]

    def get_transform_image(self):
        return self.image_to_transform[1]

    def get_metrics(self, transformed_image):
        results = {}
        for metric in self.metrics.keys():
            results[metric] = self.metrics[metric](self.get_reference_image(), transformed_image)
        return results

    def _get_pose_error(self, im, transformed_image):
        pose_error_results_dict = self.pose_esimator_sim.get_error(transformed_image, self.pose_data)
        mae = pose_error_results_dict['MAE'][0]
        return mae

    def get_metric_images(self, transformed_image):
        results = {}
        for metric in self.metric_images.keys():
            results[metric] = self.metric_images[metric](self.get_reference_image(), transformed_image)
        return results

    def _check_inputs(self):
        input_types = [(self.image_reference[0], str),
                       (self.image_reference[1], np.ndarray),
                       (self.image_to_transform[0], str),
                       (self.image_to_transform[1], np.ndarray),
                       (self.metrics, dict),
                       (self.metric_images, dict)]
        for item in input_types:
            if type(item[0]) != item[1]:
                var_name = f'{item[0]=}'.split('=')[0]
                raise TypeError('holder input: '+var_name+' should be a '+str(item[1])+' not '+str(type(item[0])))
