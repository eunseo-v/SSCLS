import os
os.sys.path.append(
    os.path.abspath('.')[
        :os.path.abspath('.').find('sscls') + len('sscls')
    ]
)
import numpy as np
import random
import math
import argparse
from mmcv import dump, load
from einops import rearrange

def gen_ske_anno(name, smpl):
    data = smpl['keypoint'] # [T, V, C=3]
    data[np.isnan(data)] = 0.0
    kpscore = smpl['keypoint_score'] # [T, V, C=3]
    # 单人动作
    joints = data[None, :] # [1, T, V, C]
    # 坐标轴方向变化
    joints[..., 2] = -joints[..., 2]
    # 减去三个轴的最小值
    x_min = joints[..., 0].min()
    y_min = joints[..., 1].min()
    z_min = joints[..., 2].min()
    # 我们需要的是x轴(水平)，z轴(垂直)
    min_vec = np.array([x_min, y_min, z_min])
    joints = joints - min_vec
    # 放缩值在屏幕内并有余量
    multi_value = max(
        joints[...,0].max()/float(1000), 
        joints[...,2].max()/float(1000)
    )
    # 只选0(水平)和2(垂直)通道
    kp = joints[..., 0:3:2]
    kp = kp/ multi_value + 50 # [1, T, V, C=2]
    assert kp[...,0].max()<1920
    assert kp[...,1].max()<1080
    kp_score = kpscore[...,0:3:2] # [T, V, C=2]
    anno = dict()
    anno['keypoint'] = kp # [N=1, T, V, C=2]
    anno['keypoint_score'] = np.array(
        kp_score[None,:][...,0], # [N=1, T, V]
        dtype=np.float32
    )
    anno['frame_dir'] = name
    anno['img_shape'] = (1080, 1920)
    anno['original_shape'] = (1080, 1920)
    anno['total_frames'] = kp.shape[1]
    anno['label'] =  smpl['label']
    return anno

if __name__ == '__main__':
    mocap_train_dict = np.load(
        '/home/yl/public_datasets/nursing/mocap_dset_train.npy', 
        allow_pickle=True, 
        encoding='latin1'
    ).item()
    mocap_test_dict = np.load(
        '/home/yl/public_datasets/nursing/mocap_dset_test.npy', 
        allow_pickle=True, 
        encoding='latin1'
    ).item()
    train_set_pkl = list() # 用于存储最终生成的样本
    test_set_pkl = list()
    anno_all = list()
    for name, mocap in mocap_train_dict.items():
        train_anno = gen_ske_anno(name, mocap)
        train_set_pkl.append(train_anno['frame_dir'])
        anno_all.append(train_anno)
    for name, mocap in mocap_test_dict.items():
        test_anno = gen_ske_anno(name, mocap)
        test_set_pkl.append(test_anno['frame_dir'])
        anno_all.append(test_anno)
    save_data = dict(
        [
            (
                'split', dict(
                    [
                        ('train',train_set_pkl),
                        ('test', test_set_pkl)
                    ]
                )
            ),
            ('annotations', anno_all)
        ]
    )
    project_folder_path = os.path.abspath('.')[
        :os.path.abspath('.').find('sscls') + len('sscls')
    ]
    ds_ncrc_folder_path = os.path.join(
        project_folder_path, 'data/ds_ncrc'
    )
    if not os.path.exists(ds_ncrc_folder_path):
        os.makedirs(ds_ncrc_folder_path)
    dump(
        save_data,
        os.path.join(
            ds_ncrc_folder_path, 'ncrc.pkl'
        )
    )