# Unsupervised Deep Learning for Phase Retrieval via Teacher-Student Distillation
Here we provide the official implementation of the AAAI-23 paper, unsupervised deep learning for phase retrieval via teacher-student distillation.

## Information
- Authors: Yuhui Quan (csyhquan@scut.edu.cn); Zhile Chen (cszhilechen@mail.scut.edu.cn); Tongyao Pang (matpt@nus.edu.sg); Hui Ji (matjh@nus.edu.sg)
- Institutes: School of Computer Science and Engineering, South China University of Technology; Department of Mathematics, National University of Singapore
- For any question, please send to **cszhilechen@mail.scut.edu.cn**
- For more information, please refer to: [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25306) [[website]](https://csyhquan.github.io/)

## Requirements
Here lists the essential packages needed to run this package:
* python 3.6
* pytorch 1.9.1
* torchvision 0.10.1
* opencv-python 4.7
* pillow 9.5
* scipy 0.10.1

## Start Training
1. Download the dataset provided in the [Google Drive](https://drive.google.com/drive/folders/1UBp1wI-witB_Vdbs-yqAXJnZ-1tNYb9F?usp=drive_link), which includes the measurements for training and the testsets. 
Place them under the directory './data', e.g., './data/CDP_uniform' and './data/PrDeep_12_128'.
2. Excute the training script, e.g.,
```
python train.py --optimizer 'Adam' --gpu_list 0 --stage_numT 5 --stage_numS 5 --hidden_channel 64 --lr 5e-4 --batch_size 8 --expe_name 'CDP_uniformx4' --scheduler 'multistep' --gamma 0.5 --start_epoch 0 --end_epoch 300 --data_dir 'data' --measurements 'CDP_uniform' --mask_x 4 --noise_alpha 9 --test_name 'PrDeep12' 'BSD68' --eval
```

## Citation
```
@inproceedings{quan2023unsupervised,
  title={Unsupervised deep learning for phase retrieval via teacher-student distillation},
  author={Quan, Yuhui and Chen, Zhile and Pang, Tongyao and Ji, Hui},
  booktitle={Proceedings of AAAI Conference on Artificial Intelligence},
  volume={3},
  year={2023}
}
```
