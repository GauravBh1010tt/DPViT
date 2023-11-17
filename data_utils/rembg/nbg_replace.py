import os
import typing

#import moviepy.editor as mpy
import numpy as np
import argparse
import requests
import torch
import torch.nn.functional
import torch.nn.functional
from hsh.library.hash import Hasher
from tqdm import tqdm
import glob
from PIL import Image
import pickle
import csv
import pdb

import matplotlib.pyplot as plt
import pandas as pd
import glob

from u2net import u2net

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def iter_frames(path):
    return mpy.VideoFileClip(path).resize(height=320).iter_frames(dtype="uint8")


class Net(torch.nn.Module):
    def __init__(self, model_name, path=''):
        super(Net, self).__init__()
        hasher = Hasher()

        model, hash_val, drive_target, env_var = {
            'u2netp':          (u2net.U2NETP,
                                'e4f636406ca4e2af789941e7f139ee2e',
                                '1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy',
                                'U2NET_PATH'),
            'u2net':           (u2net.U2NET,
                                '09fb4e49b7f785c9f855baf94916840a',
                                '1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P',
                                'U2NET_PATH'),
            'u2net_human_seg': (u2net.U2NET,
                                '347c3d51b01528e5c6c071e3cff1cb55',
                                '1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
                                'U2NET_PATH')
            }[model_name]
        if not path:
            path = os.environ.get(env_var, os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")))
    
            #print (path, hasher.md5(path), hash_val)

            if not os.path.exists(path) or hasher.md5(path) != hash_val:
                head, tail = os.path.split(path)
                os.makedirs(head, exist_ok=True)

                URL = "https://docs.google.com/uc?export=download"

                session = requests.Session()
                response = session.get(URL, params={"id": drive_target}, stream=True)

                token = None
                for key, value in response.cookies.items():
                    if key.startswith("download_warning"):
                        token = value
                        break

                if token:
                    params = {"id": drive_target, "confirm": token}
                    response = session.get(URL, params=params, stream=True)

                total = int(response.headers.get("content-length", 0))

                with open(path, "wb") as file, tqdm(
                    desc=f"Downloading {tail} to {head}",
                    total=total,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                    ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = file.write(data)
                        bar.update(size)

        net = model(3, 1)

        net.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
        net.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
        net.eval()
        self.net = net

    def forward(self, block_input: torch.Tensor):
        image_data = block_input.permute(0, 3, 1, 2)
        original_shape = image_data.shape[2:]
        image_data = torch.nn.functional.interpolate(image_data, (320, 320), mode='bilinear')
        image_data = (image_data / 255 - 0.485) / 0.229
        out = self.net(image_data)[:, 0:1]
        ma = torch.max(out)
        mi = torch.min(out)
        out = (out - mi) / (ma - mi) * 255
        out = torch.nn.functional.interpolate(out, original_shape, mode='bilinear')
        out = out[:, 0]
        out = out.to(dtype=torch.uint8, device=torch.device('cpu'), non_blocking=True).detach()
        return out


@torch.no_grad()
def remove_many(image_data: typing.List[np.array], net: Net):
    image_data = np.stack(image_data)
    image_data = torch.as_tensor(image_data, dtype=torch.float32, device=DEVICE)
    return net(image_data).numpy()

net = Net('u2netp','u2netp.pth')

def get_class_list(split='train'):
    filename = 'tim_train.csv'
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        category_list = []
        class_list = []
        for row in csv_reader:
            category_list.append(row[1])
            class_list.append(row[0])
        category_list = list(set(category_list))
    return class_list, category_list 
#print (img_dir,folders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rembg')
    parser.add_argument('--dataset', default='miniIM', type=str, choices=['miniIm', 'in9', 'tIM'])
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--img_dir', default='', type=str)

    args = parser.parse_args()

    #dataset = "tiered_imagenet" # miniIM, tIM, cub 

    #batch = 120

    #if dataset == 'mini':
    #img_dir='../'+args.dataset+"/train/"
    #img_dir="../"+dataset
    #folders,_ = get_class_list()

    folders = glob.glob(args.img_dir+'/*')

    for fold in folders:
        
        #files = glob.glob(fold+"/*")
        #print (fold)
        this_class_target_dir = args.img_dir + '/' + fold + '/'

        files = glob.glob(os.path.join(this_class_target_dir, '*'))

        print (fold, len(files), this_class_target_dir)
        for idx in tqdm(range(0, len(files), batch)):
            data, save_paths = [], []
            for im_path in files[idx:idx+batch]:
                img = Image.open(im_path)
                data.append(np.array(img))
                save_paths.append(im_path)

            out = remove_many(data,net)
            
            for idx, im in enumerate(save_paths):
                img = data[idx]
                fore = np.stack([out[idx],out[idx],out[idx]]).transpose(1,2,0)
                #pdb.set_trace()
                plt.imsave(im, np.hstack([img,fore]))
            #print (a.shape)