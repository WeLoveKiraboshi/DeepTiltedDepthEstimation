import yaml
import os
from utils.tools import recreate_dirs
import shutil
import numpy as np

class Config:

    def __init__(self, cfg_id, create_tb_dirs=False):
        self.id = cfg_id
        self.data_dir = '/media/yukisaito/ssd2/ScanNetv2/'
        self.cfg_name = 'config/%s.yml' % cfg_id
        if not os.path.exists(self.cfg_name):
            print("Config file doesn't exist: %s" % self.cfg_name)
            exit(0)
        cfg = yaml.load(open(self.cfg_name, 'r'), Loader=yaml.FullLoader)
        #self.ignore_index = yaml.load(open('%s/meta/invalid.yaml' % (self.data_dir)), Loader=yaml.FullLoader)
        self.imsize = cfg.get('imsize', [240, 320])

        # create dirs
        self.base_dir = '/home/yukisaito/TiltedDepthEstimation'
        self.mode = cfg.get('mode', 'sr_only')

        self.input_augmentation = cfg.get('input_augmentation', 'random_warp_input')
        self.image_padding_mode = cfg.get('image_padding_mode', 'zeros')

        self.result_dir = '%s/results' % self.base_dir
        self.cfg_dir = '%s/%s' % (self.result_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        if create_tb_dirs:
            recreate_dirs(self.tb_dir)
        shutil.copyfile(self.cfg_name, os.path.join(self.cfg_dir, '%s.yml' % cfg_id))

        self.save_model_interval = cfg.get('save_model_interval', 1000)
        self.tb_log_interval = cfg.get('tb_log_interval', 1000) # unit iter

        self.seed = cfg['seed']
        self.network = cfg['network']
        self.dropout_p = cfg.get('dropout', 0.0) #0.3

        self.optimizer = cfg.get('optimizer', 'Adam')
        self.lr = cfg.get('lr', 1.e-4)
        self.epochs = cfg.get('epochs', 50)
        self.start_epochs = cfg.get('start_epochs', 0)
        self.shuffle = cfg.get('shuffle', False)
        self.augment = cfg.get('augment', False)

        self.bs = cfg.get('bs', 1)
        self.weightdecay = cfg.get('weight_decay', 0.0)
        self.loss = cfg.get('loss', 'loss')
        self.loss_pose_w = cfg.get('loss_pose_w', 0.01)
        self.loss_net_w = cfg.get('loss_net_w', 1)


        self.pre_train = cfg.get('pre_train', True)
        #self.metrics = cfg.get('eval_metrics', 'mse')
        #self.augment_type = cfg.get('augment_type', ['jitter', 'rotate', 'perspective', 'center_crop', 'gaussian_blur'])

        self.sr_checkpoint_path = cfg.get('sr_checkpoint_path', None)
        self.checkpoint_path = cfg.get('checkpoint_path', None)
        self.full_checkpoint_path = cfg.get('full_checkpoint_path', None)
        if self.sr_checkpoint_path != None:
            assert os.path.isfile(self.sr_checkpoint_path), "=> no model found at '{}'".format(self.sr_checkpoint_path)
        if self.checkpoint_path != None:
            assert os.path.isfile(self.checkpoint_path), "=> no model found at '{}'".format(self.checkpoint_path)
        if self.full_checkpoint_path != None:
            assert os.path.isfile(self.full_checkpoint_path), "=> no model found at '{}'".format(self.full_checkpoint_path)
        if self.checkpoint_path == None and self.sr_checkpoint_path == None:
            print("No checkpoint loaded... training from init...")
        #if train from scratch, please turn on
        self.init_train = cfg.get('init_train', True)
        #num_workers if any for especially dataloader
        self.num_worker = cfg.get('num_workers', 16)

        self.train_dataset_split = cfg.get('train_dataset', None)
        self.test_dataset_split = cfg.get('test_dataset', None)
        if 'my_scannet_standard_train_test_val_split.pkl' in self.train_dataset_split:
            self.train_dataset_split_id = 'standard'
        elif 'my_full_2dofa_scannet.pkl' in self.train_dataset_split:
            self.train_dataset_split_id = 'full2dofa'
        else:
            print('There is no valid test_dataset_train_test_split......')
            exit(0)




        self.dataset = cfg.get('dataset', 'scannet')
        self.K = cfg.get('K', np.array([[577.87061 * 0.5, 0., 319.87654 * 0.5], [0, 577.87061 * 0.5, 239.87603 * 0.5], [0., 0., 1.]], dtype=np.float32))
        self.scale_mode = 'tanh' #'depth' use Depth Scaling processing in Dataloader


