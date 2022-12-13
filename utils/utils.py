import matplotlib
import matplotlib.cm
import numpy as np
import torch
import math

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class BinAverageMeter(object):
    def __init__(self, upper=90, lower=-90, step=10):
        self.reset()
        self.bins = list(range(lower, upper, step))

        self.label = [0] * int(len(self.bins)+1)
        self.total_values = [0] * int(len(self.bins)+1)
        self.avg_values = [0] * int(len(self.bins)+1)
        self.step = step

    def reset(self):
        self.step = 0

    def update(self, val, x=0):
        bin_x = math.ceil(x / self.step) + int(len(self.bins)/2) -1
        #print('x:{} -> bin_x:{}'.format(x, bin_x))
        self.total_values[bin_x] += val
        self.label[bin_x] += 1
        self.avg_values[bin_x] = self.total_values[bin_x] / self.label[bin_x]

        
        
        

def colorize(value, cmap='jet'):
    value = value.cpu().numpy()[0,:,:]
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))
    
    
def check_nan_ckpt(cnn):
    is_nan_flag = False
    for name, param in cnn.named_parameters():
        if torch.sum(torch.isnan(param.data)):
            is_nan_flag = True
            break
    return is_nan_flag


def abs_rel_loss(pred, gt, mask, eps=1.e-10):
    pred *= mask
    gt *= mask
    gt[gt == 0] = eps
    pred[pred == 0] = eps
    bs = gt.shape[0]
    if bs != 1:
        print('Error in util.py 74 line... : invalid batch size')
        exit(0)
    abs_rel = (np.abs(gt - pred) / gt).mean()
    return abs_rel



def compute_errors(gt, pred, mask, eps=1.e-10):
    """Computation of error metrics between predicted and ground truth depths
        """
    if mask.shape[1] == 3:
        mask = mask[:, 0, :, :].unsqueeze(1)
    elif len(mask.shape) == 3:
        mask = mask.unsqueeze(1)

    bs = gt.shape[0]
    pred *= mask
    gt *= mask
    gt[gt == 0] = eps
    pred[pred == 0] = eps


    abs_rel_list = []
    sq_rel_list = []
    rmse_list = []
    rmse_log_list = []
    a1_list = []
    a2_list = []
    a3_list = []

    for i in range(bs):
        thresh = np.maximum((gt[i] / pred[i]), (pred[i] / gt[i]))
        # depth_mask_tensor = np.where(thresh == Nan, 0, 1.0)
        a1 = (thresh < 1.25).to(torch.float32).mean()
        a2 = (thresh < 1.25 ** 2).to(torch.float32).mean()
        a3 = (thresh < 1.25 ** 3).to(torch.float32).mean()

        rmse = (gt[i] - pred[i]) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt[i]) - np.log(pred[i])) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = (np.abs(gt[i] - pred[i]) / gt[i]).mean()

        #print('gt = ', torch.amax(gt[i]))
        #print('pred = ', torch.amax(pred[i]))

        sq_rel = (((gt[i] - pred[i]) ** 2) / gt[i]).mean()

        abs_rel_list.append(abs_rel)
        sq_rel_list.append(sq_rel)
        rmse_list.append(rmse)
        rmse_log_list.append(rmse_log)
        a1_list.append(a1)
        a2_list.append(a2)
        a3_list.append(a3)


    return np.array(abs_rel_list).mean(), np.array(sq_rel_list).mean(), np.array(rmse_list).mean(), np.array(rmse_log_list).mean(), np.array(a1_list).mean(), np.array(a2_list).mean(), np.array(a3_list).mean()
