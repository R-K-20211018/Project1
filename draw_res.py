import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from datasets.crowd import Crowd
from models.IMPCSRNet3_model import CSRNet
#from impmodel.impghostnet import IMPGhostVGGNet
from models.vgg import vgg19
import argparse

import PIL.Image as Image

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='./target-data',
                        help='training data directory')
    parser.add_argument('--save-dir', default='./outputmodel/BL+IMPCSRNet(0-199)',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=0, pin_memory=False)
    #model = IMPGhostVGGNet()
    model = CSRNet()
    #model=vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device), False)
    epoch_minus = []

    x = []
    ground_truth = []
    predict = []

    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])

    from matplotlib import cm as c

    path = 'E:/bean_counting/TestBean_data/test_data/images/IMG_34782.jpg'
    img = transform(Image.open(path).convert('RGB')).cuda()

    output = model(img.unsqueeze(0))
    print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
    plt.imshow(temp, cmap=c.jet)
    plt.show()
