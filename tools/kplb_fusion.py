import os
os.sys.path.append(
    os.path.abspath('.')[
        :os.path.abspath('.').find('sscls') + len('sscls')
    ]
)
import numpy as np
from mmcv import load
from sklearn.metrics import precision_recall_fscore_support

def cal_metrics(scores, labels):
    """
    Args:
        scores (list[np.ndarray])
        labels (list[int])
    """
    # 1.计算topk1
    top_labels = np.array(labels)[:, np.newaxis]
    max_1_preds = np.argsort(scores, axis = 1)[:, -1:][:, ::-1]
    match_array = np.logical_or.reduce(max_1_preds == top_labels, axis=1)
    topk_acc_score = match_array.sum() / match_array.shape[0]
    # 2. 计算f1, precision, recall
    all_pred = np.argmax(scores, axis = 1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, all_pred, average=None
    )
    mean_prec = np.mean(prec)
    mean_rec = np.mean(rec)
    mean_f1 = np.mean(f1)
    pass

if __name__ == '__main__':
    project_folder_path = os.path.abspath('.')[
        :os.path.abspath('.').find('sscls') + len('sscls')
    ]
    kp_path = os.path.join(
        project_folder_path, 
        # 'model_pth/exp1/t9nsnr_nfwc/test_result/results.pkl'
        'model_pth/exp6/t9nswr_nfnc_ft/test_result/results.pkl'
    )
    lb_path = os.path.join(
        project_folder_path,
        'model_pth/exp4/t9nsnr_wfwc_5p_lb/test_result/results.pkl'
    )
    kp_results = load(kp_path)
    lb_results = load(lb_path)
    labels = list() # 存测试集每一个样本的label
    fusion_scores = list()
    for name in kp_results['sample_names']:
        kp_idx = kp_results['sample_names'].index(name)
        lb_idx = lb_results['sample_names'].index(name)
        kp_score = kp_results['outputs'][kp_idx] # [10]
        lb_score = lb_results['outputs'][lb_idx]
        fusion_score = kp_score + lb_score
        fusion_scores.append(fusion_score)
        labels.append(
            int(name[1:4]) - 1
        )
    cal_metrics(fusion_scores, labels)