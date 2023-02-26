import os
import re
import pandas as pd
import numpy as np


file_paths={
    'mocap_train_path': '/home/yl/public_datasets/nursing/train/mocap/',
    'mocap_test_path' :'/home/yl/public_datasets/nursing/test/mocap/',

    'tr_labels_path':'/home/yl/public_datasets/nursing/train/activities_train.csv',
    'tst_labels_path':'/home/yl/public_datasets/nursing/activities_test.csv'

}

data_params = {
"MOCAP_SEGMENT" : 6000,
}

def pose2idlabel(mocap_path, labels_path):
    """
        记录训练集/测试集骨架数据集的路径和样本标签
    """
    labels_map = get_ncrc_labels(labels_path=labels_path)
    pose2id = dict() # 'id-i':'骨架csv'路径
    id2label = dict() # 'id-i': int(label)
    i = 0
    samples = os.listdir(mocap_path)
    for smpl in samples:
        # smpl: segment**.csv
        mocap_smpl_path = os.path.join(
            mocap_path, smpl
        )
        if mocap_smpl_path.endswith('.csv'):
            segment_id = int(re.findall(r'\d+', smpl)[0]) # 获取csv文件名上的segmentid
            pose2id['id-'+str(i)] = mocap_smpl_path
            id2label['id-'+str(i)] = labels_map[segment_id]
            i+=1
    return pose2id, id2label
    # pose2id 训练集/测试集骨架路径
    # id2label 训练集/测试集标签

def get_ncrc_labels(labels_path):
    # 获取训练集中每一个样本对应的类别标签0-5
    lbl_map = {
        2:0,    # vital signs measurement 生命体征测量
        3:1,    # blood collection 血液采集
        4:2,    # blood glucose measurement 血糖测量
        6:3,    # indwelling drop retention and connection 输液
        9:4,    # oral care
        12:5    # diaper exchange and cleaning of area 尿布更换和区域清洁
    }
    df = pd.read_csv(labels_path)
    df.drop('subject', axis = 1, inplace=True) # 去掉医护(subject)信息
    labels_map = {}
    for _, row in df.iterrows():
        labels_map[row['segment_id']] = lbl_map[row['activity_id']]
    return labels_map

def save_mocap(dset_path, labels_path, mode='train'):
    """
        将原始训练集/测试集的数据和标签存到npy文件中，
        骨架数据只保存原始的内容，不进行插值等操作
    """
    dset_mocap = dict()
    for idi, mocap_path in dset_path.items():
        raw_folder_pt = os.path.join(
            mocap_path.split('nursing')[0],
            'nursing' 
        )
        file_pt = mocap_path.split('/')[-1]
        df = pd.read_csv(mocap_path)
        # 删除无关行
        df.drop('time_elapsed',axis=1,inplace=True)
        df.drop('segment_id',axis=1,inplace=True)
        # 直接转换成numpy，然后创建keypoint_score矩阵，对于nan数值置信度赋0，其余赋1
        data = df.to_numpy()
        frames = data.shape[0] # T
        data = data.reshape((frames, 29, 3))
        kpscore = np.ones_like(data, dtype=np.float32)
        kpscore[np.isnan(data)] = 0
        label = labels_path[idi]
        smpl_dict = {
            file_pt.split('.')[0]: {
                'keypoint': data,
                'keypoint_score': kpscore,
                'label': label
            }
        }
        dset_mocap.update(smpl_dict)
    assert len(dset_mocap) == len(dset_path)
    np.save(
        os.path.join(
            raw_folder_pt,
            'mocap_dset_{}.npy'.format(mode)
        ), dset_mocap
    )
        

if __name__ == '__main__':
    raw_mocap_train_path = file_paths['mocap_train_path']
    raw_mocap_test_path = file_paths['mocap_test_path']
    tr_labels_path = file_paths['tr_labels_path']
    tst_labels_path = file_paths['tst_labels_path']
    # 训练集信息
    raw_tr_pose2id, tr_labels = pose2idlabel(
        mocap_path=raw_mocap_train_path,
        labels_path= tr_labels_path
    )
    # 测试集信息
    raw_tst_pose2id, tst_labels = pose2idlabel(
        mocap_path=raw_mocap_test_path,
        labels_path= tst_labels_path
    )
    print('--------------DATA SPLIT----------')
    print("Train Samples: ", len(raw_tr_pose2id))
    print("Test Samples: ", len(raw_tst_pose2id))
    # 存储原始骨架数据，置信度以及标签
    save_mocap(raw_tr_pose2id, tr_labels, mode = 'train')
    save_mocap(raw_tst_pose2id, tst_labels, mode = 'test')
    pass
