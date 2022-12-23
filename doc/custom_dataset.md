# Custom Dataset
In the following, we provide how to apply our method to customized image dataset. ([ref](https://github.com/zhihao-lin/neurmips/issues/2))
you should probably follow the stepsbelow: 


## 1. Run COLMAP dense reconstruction on the image set ([ref](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses))
1. camera.txt,images.txt,points3D.txt 생성
2. 파일 구성 후 command 실행  : point_triangulator 까지 실행하면 sparse 모델 

After this step, you should have the output (format is described in [output format](https://colmap.github.io/format.html#output-format)). 
(1). camera intrinsics (camera.txt)
(2). camera extrinsics (rotation & translation) (images.txt)
(3). point cloud (points3D.txt)

## 2. Transform the camera model to PyTorch3D convention
(details in https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md)

여긴 아직 안해봐서 모르겠다.
