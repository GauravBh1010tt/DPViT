import os
import typing

#import moviepy.editor as mpy
import numpy as np
import requests
import torch
import torch.nn.functional
import torch.nn.functional
from hsh.library.hash import Hasher
from tqdm import tqdm
import glob
from PIL import Image
import pickle

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

#print (img_dir,folders)

dataset = "miniIM" # miniIM, tIM, cub 

batch = 120

if dataset == 'miniIM':
    img_dir="../data/mini_imagenet/train/"
    folders = glob.glob(img_dir+"*")

    if not os.path.exists('../data/mini_imagenet/train_fore'):
        os.mkdir('../data/mini_imagenet/train_fore')

    for fold in folders:
        out_path = "../data/mini_imagenet/train_fore/"+fold.split('/')[-1]
        
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
        files = glob.glob(fold+"/*")

        print (len(files))
        for idx in tqdm(range(0, len(files), batch)):
            data, save_paths = [], []
            for im_path in files[idx:idx+batch]:
                img = Image.open(im_path)
                data.append(np.array(img))
                save_paths.append(im_path.replace('train','train_fore'))

            out = remove_many(data,net)
            
            for idx, im in enumerate(save_paths):
                plt.imsave(im, np.stack([out[idx],out[idx],out[idx]]).transpose(1,2,0))
            #print (a.shape)