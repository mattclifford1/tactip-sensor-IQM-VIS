'''utils to handle any pytorch related task eg. generator or pose estimation'''
import torch
import os
import numpy as np
import json
import sys; sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gan_models.models_128 import GeneratorUNet, weights_init_normal, weights_init_pretrained
import downstream_task.networks.model_128 as pose_128
from downstream_task.networks.model_128 import load_weights


def preprocess_numpy_image(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)   # make into batch one 1
    image = torch.from_numpy(image)
    return image

def post_process_torch_image(image):
    image = image.cpu().detach().numpy()
    image = image[0, :, :, :]
    image = image.transpose((1, 2, 0))
    return np.clip(image, 0, 1)

'''generator model loader and get prediction of a single image'''
class generator():
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = GeneratorUNet(in_channels=1, out_channels=1)
        self.generator.to(self.device)
        self.generator.eval()
        if os.path.isfile(weights_path):
            weights_init_pretrained(self.generator, weights_path)
        else:
            print('Could not find weights path: '+str(weights_path)+'\nInitialising generator with random weights')
            self.generator.apply(weights_init_normal)

    def get_prediction(self, image):
        '''preprocess image'''
        image = preprocess_numpy_image(image).to(device=self.device, dtype=torch.float)
        ''' get prediction'''
        pred = self.generator(image)
        return post_process_torch_image(pred)


'''pose estimation model on single image'''
class pose_estimation():
    def __init__(self, weights_path, sim=True):
        self.use_sim = sim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_task_from_checkpoint(weights_path)
        self.net = pose_128.network(final_size=self.net_out_dims, task=self.task[0])
        self.net.to(self.device)
        self.net.eval()
        weights_path = self.sim_or_real_net(weights_path)
        if os.path.isfile(weights_path):
            self.load_normalisation(weights_path)
            load_weights(self.net, weights_path)
        else:
            print('*******\n*******\nCannot load pose estimation weights path: '+str(weights_path))

    def get_task_from_checkpoint(self, weights_path):
        '''get the task eg. surface_3d shear from the dir structure of the checkpoint path'''
        dirs = weights_path.split(os.sep)
        dirs[0] = os.sep
        self.task = (dirs[-6], dirs[-5])
        self.net_out_dims = int(self.task[0][-2])
        if self.task[0] == 'surface_2d' or self.task[0] == 'edge_2d':
            self.y_names = ['pose_2', 'pose_6']
        elif self.task[0] == 'surface_3d':
            self.y_names = ['pose_3', 'pose_4', 'pose_5']
        else:
            raise Exception('Incorrect task: '+str(self.task[0]))

    def sim_or_real_net(self, weights_path):
        '''change weights to real network if required'''
        dirs = weights_path.split(os.sep)
        dirs[0] = os.sep
        if self.use_sim == True and dirs[-4][:3] == 'sim':
            return weights_path
        if self.use_sim == False and dirs[-4][:3] == 'sim':
            dirs[-4] = 'real' + dirs[-4][3:]
            return os.path.join(*dirs)
        if self.use_sim == True and dirs[-4][:4] == 'real':
            dirs[-4] = 'sim' + dirs[-4][4:]
            return os.path.join(*dirs)
        if self.use_sim == False and dirs[-4][:4] == 'real':
            return weights_path

    def load_normalisation(self, weights_path):
        '''get the interpolation y labels range that the model was trined on'''
        model_dir = os.path.dirname(weights_path)
        normalisation_file = os.path.join(model_dir, 'output_normilisation.json')
        with open(normalisation_file) as f:
            self.normalisation = json.load(f)

    def get_prediction(self, image):
        '''preprocess image'''
        image = preprocess_numpy_image(image).to(device=self.device, dtype=torch.float)
        ''' get prediction'''
        return self.net(image)

    def normalise_y_labels(self, y_labels):
        # normalise groudn thruth y labels to range (-1,1)
        y_labels_normalised = []
        for label in self.y_names:
            y_labels_normalised.append(np.interp(y_labels[label], self.normalisation[label], (-1,1)))
        return torch.tensor(y_labels_normalised).to(device=self.device, dtype=torch.float)

    def get_error(self, image, y_labels):
        pred_y = self.get_prediction(image)
        labels = self.normalise_y_labels(y_labels)
        ae = torch.abs(pred_y - labels).cpu().detach().numpy()
        mae = ae.mean()
        return {'MAE': [mae], 'ABS Error':[list(ae.squeeze())]}


if __name__ == '__main__':
    path = os.path.join(os.path.expanduser('~'), 'summer-project/models/pose_estimation/surface_3d/shear/sim_LR:0.0001_BS:16/run_0/checkpoints/best_model.pth')
    e = pose_estimation(path)

    import pandas as pd
    import sys; sys.path.append('..'); sys.path.append('.')
    import gui_utils
    sensor_data = {}
    csv_file = os.path.join(os.path.expanduser('~'),'summer-project/data/Bourne/tactip/sim/surface_3d/shear/128x128/csv_train/targets.csv')
    df = pd.read_csv(csv_file)
    im_sim_dir = os.path.join(os.path.dirname(csv_file), 'images')
    im_num = 10
    image = df.iloc[im_num]['sensor_image']
    image_path = os.path.join(im_sim_dir, image)
    poses = ['pose_'+str(i) for i in range(1,7)]
    sensor_data['poses'] = {}
    for pose in poses:
        sensor_data['poses'][pose] = df.iloc[im_num][pose]
    sensor_data['im_reference'] = gui_utils.load_image(image_path)
    sensor_data['im_reference'] = gui_utils.process_im(sensor_data['im_reference'], data_type='sim')

    errors = e.get_error(sensor_data['im_reference'], sensor_data['poses'])
    print(errors)
