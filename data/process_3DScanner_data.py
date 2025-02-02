import csv
import json

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
import glob
import PIL
import imageio.v2 as imageio
import shutil



IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1440
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0
np.random.seed(0)
"""

python data/process_3DScanner_data.py --basedir ./data/scanner3D/hallway6 --num_train=170 --num_val=25 --point_fname=XYZ_color.txt

python data/process_3DScanner_data.py --basedir ./data/scanner3D/obj3 --num_train=120 --num_val=35 --point_fname=obj3_18_56_22.txt

python data/process_3DScanner_data.py --basedir ./data/scanner3D/renaissance --num_train=120 --num_val=25 --point_fname=XYZ_color.txt

python data/process_3DScanner_data.py --basedir ./data/scanner3D/test --num_train=39 --num_val=0 --point_fname=colmap_test_20_41_49.txt


"""

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./data/strayscanner/computer',
                        help='input data directory')
    parser.add_argument("--point_fname", type=str, default='XYZ_color.txt',
                        help='input data directory')


    parser.add_argument("--num_train", type=int, default=120,
                        help='number of train data')
    parser.add_argument("--num_val", type=int, default=100,
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


def process_3DScanner_data(args,mode,selected_index):
    mode_path = "{}/{}".format(args.basedir, mode)
    images_path = os.path.join(mode_path, "images")
    depth_path = os.path.join(mode_path, "depth")
    confi_path = os.path.join(mode_path, "confidence")

    make_dir(mode_path)
    make_dir(images_path)
    make_dir(depth_path)
    make_dir(confi_path)


    # point cloud
    shutil.copy2(os.path.join(args.basedir, args.point_fname),os.path.join(args.basedir, "point_color.txt"))
    shutil.copy2(os.path.join(args.basedir, "point_color.txt"),os.path.join(mode_path, "point_color.txt"))

    poses = []
    for i in selected_index:
        # img
        img_fname = "{}/frame_{}.jpg".format(args.basedir, str(i).zfill(5) )
        img = PIL.Image.fromarray(imageio.imread(img_fname))
        # img = Image.fromarray(img)
        img = img.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        img = np.array(img)
        save_img_fame = os.path.join(images_path, "frame_{}".format(str(i)).zfill(5) + '.jpg')
        skvideo.io.vwrite(save_img_fame, img)

        #depth
        depth_fname = "{}/depth_{}.png".format(args.basedir, str(i).zfill(5))
        depth = load_depth(depth_fname)
        depth_fname = "{}/dpeth_{}.npy".format(depth_path, str(i).zfill(5) )
        np.save(depth_fname, depth)

        #load confidence
        confi_fname = "{}/conf_{}.png".format(args.basedir, str(i).zfill(5)) #os.path.join(args.basedir, 'confidence', f'{i:06}.png')
        confidence = load_confidence(confi_fname)
        confi_fname = "{}/conf_{}.png".format(confi_path, str(i).zfill(5))
        np.save(confi_fname, confidence)

        #pose
        pose_fname = "{}/frame_{}.json".format(args.basedir, str(i).zfill(5))
        f = json.load(open(pose_fname, 'r'))
        rt = np.array(f["cameraPoseARFrame"]) # rotation matrix line
        rt[-1] = 1
        rt_line = [str(e) for e in rt]
        poses.append(' '.join(rt_line) + '\n')

    #pose save
    camera_pose_file = os.path.join(mode_path,'poses.txt') # 4X4 matrix -> line
    with open(camera_pose_file, 'w') as f:
        f.writelines(poses)

    #intrinsic save
    pose_fame = "{}/frame_{}.json".format(args.basedir, str(0).zfill(5))
    f = json.load(open(pose_fame,'r'))
    intr = np.array(f["intrinsics"],dtype=float)
    intr[0] = intr[0] * float(DEPTH_WIDTH) / IMAGE_WIDTH
    intr[2] = intr[2] * float(DEPTH_WIDTH) / IMAGE_WIDTH
    intr[4] = intr[4] * float(DEPTH_HEIGHT) / IMAGE_HEIGHT
    intr[5] = intr[5] * float(DEPTH_HEIGHT) / IMAGE_HEIGHT
    intr[-1] = 1
    intr_fname = os.path.join(mode_path,'intrinsic.txt')
    f = open(intr_fname, 'w')
    for e in intr:
        f.write(str(e)+" ")
    f.close()

def main(args):
    #rgb images
    all_index = np.array([int(fname[-9:-4]) for fname in glob.glob(os.path.join(args.basedir, 'frame*.jpg'))])
    all_index.sort()
    # 이미지 파일이 있는 pose 파일 골라내기
    # frame_paths = list(sorted(glob(os.path.join(args.basedir, "*.json"))))
    # frame_paths = list(filter(lambda x: "frame_" in x, frame_paths))



    n = len(all_index)
    print("N : ",n)
    train_index = np.linspace(0, n, args.num_train, endpoint=False, dtype=int)
    # # if random sampling
    val_index = np.delete(np.arange(n),train_index)
    val_index = np.random.choice(val_index,args.num_val,replace=False)

    #save
    modes = ['train', 'valid']
    for mode in modes:
        selected_index = (all_index[train_index] if mode == 'train' else all_index[val_index])
        process_3DScanner_data(args, mode, selected_index )




if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
