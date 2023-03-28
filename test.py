import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.IMPCSRNet3_model import CSRNet
#from impmodel.impghostnet import IMPGhostVGGNet
from models.vgg import vgg19
#from models.CSRNet3_model import CSRNet
import argparse

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='./target-data',
                        help='training data directory')
    parser.add_argument('--save-dir', default='./outputmodel/0629-085008',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=0, pin_memory=False)#num_workers=8
    # model = CSRGhostNet()
    #model = IMPGhostVGGNet()
    model =CSRNet()
    #model=vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device), False)
    epoch_minus = []

    x = []
    ground_truth = []
    predict = []

    # fifteen = []
    # twenty = []

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            ground_truth.append(count[0].item())
            predict.append(torch.sum(outputs).item())
            epoch_minus.append(temp_minu) 
            # if(np.abs(temp_minu) > 15):
            #     fifteen.append(name)
            #     if(np.abs(temp_minu) > 20):
            #         twenty.append(name)

    # print(ground_truth)
    # print(predict)
    # print(fifteen)
    # print(twenty)
    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
