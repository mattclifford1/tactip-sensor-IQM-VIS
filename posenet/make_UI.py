import os
import numpy as np
import IQM_VIS
import data_holder
import sys; sys.path.append('..'); import image_utils


def run():
    file_path = os.path.dirname(os.path.abspath(__file__))

    # metrics functions must return a single value
    metric = {'MAE': IQM_VIS.metrics.MAE,
              'MSE': IQM_VIS.metrics.MSE,
              '1-SSIM': IQM_VIS.metrics.ssim()}

    # metrics images return a numpy image
    metric_images = {'MSE': IQM_VIS.metrics.MSE_image,
                     'SSIM': IQM_VIS.metrics.SSIM_image()}

    # first row of images
    sim_dataset_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','data/sim/surface_3d/tap/128x128/csv_train/targets.csv')

    data_sim = data_holder.dataset_holder(sim_dataset_csv,
                                  metric,
                                  metric_images,
                                  pose_path=os.path.join(os.path.expanduser('~'), 'summer-project/models/pose_estimation/surface_3d/shear/sim_LR:0.0001_BS:16/run_0/checkpoints/best_model.pth'),
                                  sim=True)
    data_real = data_holder.dataset_holder(sim_dataset_csv,
                                  metric,
                                  metric_images,
                                  pose_path=os.path.join(os.path.expanduser('~'), 'summer-project/models/pose_estimation/surface_3d/shear/sim_LR:0.0001_BS:16/run_0/checkpoints/best_model.pth'),
                                  sim=False)
    # second row of images
    # define the transformations
    transformations = {
               'rotation':{'min':-180, 'max':180, 'function':IQM_VIS.transforms.rotation},    # normal input
               'blur':{'min':1, 'max':41, 'normalise':'odd', 'function':IQM_VIS.transforms.blur},  # only odd ints
               'brightness':{'min':-1.0, 'max':1.0, 'function':IQM_VIS.transforms.brightness},   # normal but with float
               'zoom':{'min':0.1, 'max':4.0, 'function':image_utils.zoom_image, 'init_value': 1, 'num_values':21},  # requires non standard slider params
               'x_shift':{'min':-1.0, 'max':1.0, 'function':image_utils.translate_x},
               'y_shift':{'min':-1.0, 'max':1.0, 'function':image_utils.translate_y},
               }

    # use the API to create the UI
    IQM_VIS.make_UI([data_sim, data_real],
                transformations,
                metrics_avg_graph=True)


if __name__ == '__main__':
    run()
