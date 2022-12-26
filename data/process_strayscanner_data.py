import csv
# import pickle
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import os
import numpy as np
# import imageio
# import json
# from transforms3d.quaternions import quat2mat
# from skimage import img_as_ubyte
from PIL import Image
import skvideo.io
import cv2
import random


# DEPTH_WIDTH = 256
# DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0
np.random.seed(0)
"""
conda activate StrayVisualizer-main
python data/process_strayscanner_data.py --basedir ./data/hallway02 --num_train=200

"""

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./data/strayscanner/computer',
                        help='input data directory')
    parser.add_argument("--num_train", type=int, default=120,
                        help='number of train data')
    parser.add_argument("--num_test", type=int, default=12,
                        help='number of train data')

    parser.add_argument("--near_range", type=int, default=2,
                        help='near near range')
    parser.add_argument("--far_range", type=int, default=6,
                        help='far near range')

    parser.add_argument("--use_confi0_depth", type=int, default=1,
                        help='far near range')
    parser.add_argument("--depth_bound2", type=float, default=0.4,
                        help='condi2 depth range')
    parser.add_argument("--depth_bound1", type=float, default=1,
                        help='condi1 depth range')

    return parser

# def load_depth(path, confidence=None):
#     depth_mm = np.array(Image.open(path))
#     depth_m = depth_mm.astype(np.float32) / 1000.0
#     return depth_m

def load_depth(path, confidence=None):
    extension = os.path.splitext(path)[1]
    if extension == '.npy':
        depth_mm = np.load(path)
    elif extension == '.png':
        depth_mm = np.array(Image.open(path))
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m


def load_confidence(path):
    return np.array(Image.open(path))

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def process_stray_scanner(args, data,split='train'):
    #generate folder
    split_path = "{}/{}".format(args.basedir, split)
    rgb_path = "{}/images".format(split_path)
    sparse_path = "{}/sparse".format(split_path)

    make_dir(split_path)
    make_dir(rgb_path)
    make_dir(sparse_path)

    # #cameras.txt
    H,W = data['rgb'][0].shape[:-1]
    intrinsic = data['intrinsics']
    camera_file = open( os.path.join(sparse_path, 'cameras.txt'), 'w')
    # camera_id, model, W, H , fx, fy , cx ,cy
    line = "1 PINHOLE "+ str(W) + ' ' + str(H) + ' ' + str(intrinsic[0][0]) + ' ' + str(intrinsic[1][1]) + ' ' + str(intrinsic[0][2]) + ' ' + str(intrinsic[1][2])
    camera_file.write(line)
    camera_file.close()

    #points3D.txt
    points3D_file = open(os.path.join(sparse_path, 'points3D.txt'),'w')
    points3D_file.close()



    # sampling
    n = data['odometry'].shape[0]
    num_train = args.num_train
    num_val = 100
    all_index = np.arange(n)
    train_index = np.linspace(0, n, num_train, endpoint=False, dtype=int)
    # # if random sampling
    val_index = np.delete(all_index,train_index)
    val_index = np.random.choice(val_index,num_val,replace=False)
    select_index = []

    if split == 'train':
        select_index = train_index
    elif split == 'val':
        select_index = val_index
    # rgbs = np.array(data['rgb'])[select_index.astype(int)]
    # poses = data['odometry'][select_index]
    rgbs = data['rgb']
    poses = data['odometry']

    #TODO : 코드 미완이라 확인해야함.
    #images folder and images.txt
    pose_file = "{}/images.txt".format(sparse_path)
    lines = []
    for i in select_index:  #for i, (rgb, pose) in enumerate(zip(rgbs,poses)):
        rgb = rgbs[i]
        pose = poses[i]

        #pose :  timestamp, frame, x, y, z, qx, qy, qz, qw
        skvideo.io.vwrite(os.path.join(rgb_path, str(int(i)).zfill(5) + '.png'), rgb)
        # pose : # timestamp, frame(float ex 1.0), tx, ty, tz, qx, qy, qz, qw
        # image_id(1,2,3,...),  qw, qx, qy, qz ,tx, ty, tz , camera_id, name(file name)
        line = []
        line.append(str(i+1)) #TODO: i+1 ???

        # rt = pose[2:]
        # rt[0] = pose[-1] # qw
        # qxyz = pose[-4:-1]
        # rt[1:4] = pose[-4:-1] # qx, qy, qz
        # txyz = pose[2:5]
        # rt[4:] = pose[2:5] # tx, ty, tz
                        #tx, ty, tz, qx, qy, qz, qw
        # for j in rt :  # qw, qx, qy, qz ,tx, ty, tz
        #     line.append(str(j))

        index = [-1, -4, -3, -2, 2, 3, 4] # qw, qx, qy, qz ,tx, ty, tz
        for j in index:
            line.append(str(pose[j]))


        line.append(str(1)) # camera_id
        line.append( str(int(pose[1])).zfill(5) + '.png') # name
        lines.append(' '.join(line) + '\n' + '\n')
    # pose_file.close()

    with open(pose_file, 'w') as f:
        f.writelines(lines)




def precompute_depth_sampling(origin_near,origin_far,depth,confidence):

    depth_min, depth_max = origin_near, origin_far
    # [N,H,W]
    depth = torch.tensor(depth)
    confidence = torch.tensor(confidence)

    depth = depth[...,None]  #[N,H,W,1]
    confidence = confidence[..., None] #[N,H,W,1]
    near = torch.ones_like(depth) #[N,H,W,1]
    far = torch.ones_like(depth)

    condi2 = confidence[..., 0] == 2 #[N,H,W]
    bound2 = args.depth_bound2
    near[condi2]= torch.clamp(depth[condi2]-bound2 ,min=0)
    far[condi2] = depth[condi2]+bound2

    condi1 = confidence[..., 0] == 1
    bound1 = args.depth_bound1
    near[condi1] = torch.clamp(depth[condi1]-bound1 ,min=0)
    far[condi1] = depth[condi1]+bound1

    condi0 = confidence[..., 0] == 0
    if args.use_confi0_depth > 0:
        # consider depth
        near[condi0] = 2 #torch.clamp(depth[condi0]-0.3,max=4)
        far[condi0] = torch.clamp(depth[condi0]+0.3,min=depth_max)
    else:
        # near = 4
        near[condi0]= 4 #torch.clamp(depth[condi0]+0.3,min=4)
        far[condi0] = torch.clamp(depth[condi0]+0.3,min=depth_max)

    test_near , test_far = near[...,0],far[...,0]
    test_depth = depth[...,0]
    return near[...,0],far[...,0]  #[B,H*W]


def main(args):
    # data load
    data = {}
    intrinsics = np.loadtxt(os.path.join(args.basedir, 'camera_matrix.csv'), delimiter=',')
    data['intrinsics'] = intrinsics
    odometry = np.loadtxt(os.path.join(args.basedir, 'odometry.csv'), delimiter=',', skiprows=1)
    data['odometry'] = odometry


    rgbs=[]
    video_path = os.path.join(args.basedir, 'rgb.mp4')
    video = skvideo.io.vreader(video_path)
    rgb_img_path = os.path.join(args.basedir, 'rgb')
    # make_dir(rgb_img_path)
    for i, (T_WC, rgb) in enumerate(zip(data['odometry'], video)):
        # load depth
        print(f"Integrating frame {i:06}", end='\r')

        #rgb image
        rgb = Image.fromarray(rgb)
        # rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb = np.array(rgb)
        rgbs.append(rgb)


    data['rgb']=rgbs
    # split = ['train', 'val', 'test'] # conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    split = ['train', 'val']
    for mode in split:
        process_stray_scanner(args,data,mode)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
