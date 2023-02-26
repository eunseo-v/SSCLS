'''
    my taichi test, DDP is not allowed
    需要完成能够获取网络的中间层的输出
    编写钩子函数，如何注册，数据输入时，记录中间的输出
    可选输出内容：
        results.pkl 测试集样本的分类概率和样本名 'outputs','sample_names'
        test_metric.json 测试集的评价指标 'top_k_accuracy', 'mean_class_accuracy'
        confusion_matrix.png  confusion_matrix.csv 混淆矩阵
        wrong_sample_statistics.json 分类错误的测试样本
        t_sne_vis_out.pkl 中间层的输出和对应label 'target_outputs', 'target_labels'
        t_sne_vis.png 特征层的t-SNE可视化结果
'''

import mmcv
from mmcv import Config, ProgressBar, dump
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from pyskl.core.evaluation import confusion_matrix

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
from sklearn.metrics import precision_recall_fscore_support

import os
import argparse
import json
import numpy as np
import torch
import pandas as pd
from einops import repeat
from collections import OrderedDict

os.sys.path.append(
    os.path.abspath('.')[
        :os.path.abspath('.').find('sscls') + len('sscls')
    ]
)

from project_utils.dset_class_info import pt_confmat_info_10, pt_confmat_info_60, pt_confmat_info_6
from project_utils.utils import plot_confusion_matrix, draw_pic

def parse_args():
    parser = argparse.ArgumentParser(
        description='my eval test py'
    )
    parser.add_argument(
        'config', 
        # '--config',
        default= 'configs/sscls/exp7/ncrc_adamw5e-4.py',
        help = 'configuration file path'
    )
    parser.add_argument(
        'checkpoint',
        # '--checkpoint',
        default = 'model_pth/exp7/ncrc_adamw5e-4/best_top1_acc_epoch_9.pth',
        help = 'pretrained model path'
    )
    args = parser.parse_args()
    return args

def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)

class Infer_With_Hook:
    '''
        得到每一组输入的最终输出和target_layer_name的输出
    '''
    def __init__(self, model, target_layer_name):
        self.model = MMDataParallel(model, device_ids = [0])
        self.model.eval()
        self.final_output = []
        self.target_output = [] # 用来存储中间层的输出
        self.target_label = [] # 用来存储中间层的输出对应的label, t-SNE可视化使用
        self._register_hooks(target_layer_name)
    
    def _register_hooks(self, layer_name):
        '''
            Register forward hook to a layer, 
            given layer_name to obtain activations
            eg: layer_name 'cls_head/avg_pool'的输出为B*double*num_clips
        '''
        def get_activations(module, input, output):
            self.target_output.extend(output.clone().detach().cpu().squeeze().numpy()) # [B*num_segs, C]
        layer_ls = layer_name.split('/')
        prev_module = self.model.module
        for layer in layer_ls:
            prev_module = prev_module._modules[layer]
        target_layer = prev_module
        target_layer.register_forward_hook(get_activations)
        pass

    def __call__(self, data_loader):
        dataset = data_loader.dataset
        prog_bar = ProgressBar(len(dataset))
        for data in data_loader:
            num_segs = data['imgs'].shape[1] # double*num_clips
            self.target_label.extend(
                repeat(
                    data['label'], 'b -> (b m)',
                    m = num_segs
                ) # 正确的数据重复方式
            )
            with torch.no_grad():
                # 因为注册了钩子，也能执行get_activations
                result = self.model(return_loss=False, **data) 
            self.final_output.extend(result)
            batch_size = len(next(iter(data.values())))
            for _ in range(batch_size):
                prog_bar.update()

def inference_pytorch(cfg, data_loader):
    '''
        Get predictions by pytorch models.
    '''
    turn_off_pretrained(cfg.model)
    # build the model and load checkpoint
    model = build_model(
        cfg.model, 
        # train_cfg=None,
        # test_cfg = cfg.get('test_cfg')
    )
    # 此时 model._modules['backbone']._modules['conv1']._modules['conv']._parameters['weight'].device = 'cpu'
    load_checkpoint(model, cfg.checkpoint, map_location='cpu')
    # 测试集验证，加上钩子函数，得到cls_head/avg_pool层的输出
    test_taichi = Infer_With_Hook(model, target_layer_name='cls_head/pool')
    test_taichi(data_loader=data_loader)
    return {
        'output': test_taichi.final_output,
        'target_output': test_taichi.target_output,
        'target_label': test_taichi.target_label
    }


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.checkpoint:
        cfg.setdefault('checkpoint', args.checkpoint)
    else:
        raise ValueError('plz infer the model checkpoint')
    output_config = cfg.get('output_config', {})
    eval_config = cfg.get('eval_config', {})
    # build the dataloader
    dataset = build_dataset(
        cfg.data.test, 
        dict(test_mode = True)
    )
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        # dist=False,
        shuffle=False)
    dataloader_setting = dict(
        dataloader_setting,
        **cfg.data.get('test_dataloader', {})
    )
    data_loader = build_dataloader(dataset, **dataloader_setting)
    # outputs.keys() ['output', 'target_output', 'target_label']
    outputs = inference_pytorch(cfg=cfg, data_loader= data_loader)
    '''
        想存储的数据内容:
        1. 对测试集的各类分类概率
        2. 测试集各样本的名称，1,2可以存成一个pkl文件   results.pkl
        3. 中间层的特征输出和对应标签，存成一个pkl文件  t_sne_vis_out.pkl
        可视化内容:
        1. 混淆矩阵                 confusion_matrix.png
        2. 哪些样本分错了            wrong_sample_statistics.json
        3. 中间层的特征输出的可视化   t_sne_vis.png
    '''
    if output_config.get('out', None):
        out_path = output_config['out']
        mmcv.mkdir_or_exist(os.path.dirname(out_path))
        sample_names = [
            ann['frame_dir'] for ann in dataset.video_infos
        ] # 每个样本的样本名
        out = {
            'outputs': outputs['output'],
            'sample_names': sample_names
        }
        dump(
            out, out_path
        ) # 测试集的各类分类概率和样本名称
    if eval_config:
        # 对于top_k_acc和mean_class_acc，按照dataset.evaluate计算
        # confusion_matrix计算混淆矩阵，并且统计错误的样本
        # t_sne_vis，先将数据保存，然后通过PCA降维后，t-sne可视化
        mmcv.mkdir_or_exist(eval_config.get('metric_out'))
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 
            'confusion_matrix', 't_sne_vis'
        ]
        eval_metrics = eval_config.get('eval')
        for metric in eval_metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        eval_results = OrderedDict()
        for metric in eval_metrics:
            if (metric == 'top_k_accuracy') or (metric == 'mean_class_accuracy'):
                eval_results_temp = dataset.evaluate(outputs['output'], metric)
                for key in eval_results_temp.keys():
                    eval_results[key] = eval_results_temp[key]
            # 添加计算precision, recall, f1-score
            all_preds = np.argmax(outputs['output'], axis=1)
            all_targets = np.array(
                [ann['label'] for ann in dataset.video_infos],
                dtype=np.int64
            )
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average = None
            )
            eval_results['Class Wise Precision'] = np.round(prec, 3).tolist()
            eval_results['Class Wise Recall'] = np.round(rec, 3).tolist()
            eval_results['Class Wise F1-score'] = np.round(f1, 3).tolist()
            eval_results['Overall Precision'] = np.mean(prec).tolist()
            eval_results['Overall Recall'] = np.mean(rec).tolist()
            eval_results['Overall F1-score'] = np.mean(f1).tolist()
            if metric == 'confusion_matrix':
                gt_labels = [ann['label'] for ann in dataset.video_infos]
                confusion_mat = confusion_matrix(
                    y_pred=np.argmax(outputs['output'], axis=1),
                    y_real = gt_labels
                ).astype(float)
                wrong_sample_statistics=dict() # 用于统计错误样本
                if cfg.model.cls_head.num_classes == 60:
                    class_dict_info = pt_confmat_info_60()
                elif cfg.model.cls_head.num_classes ==10:
                    class_dict_info = pt_confmat_info_10()
                elif cfg.model.cls_head.num_classes ==6:
                    class_dict_info = pt_confmat_info_6()
                else:
                    raise ValueError(
                        'invalid {cfg.model.cls_head.num_classes} datset class number'
                    )
                # 画混淆矩阵并保存
                ax = plot_confusion_matrix(
                    cm = confusion_mat,
                    classes = class_dict_info.values(),
                    normalize = False
                )
                fig = ax.get_figure()
                fig.savefig(
                    os.path.join(
                        eval_config.get('metric_out'),
                        'confusion_matrix.png'
                    )
                )
                df = pd.DataFrame(
                    data=confusion_mat,
                    index = list(class_dict_info.values()),
                    columns = list(class_dict_info.values())
                )
                df.to_csv(
                    os.path.join(
                        eval_config.get('metric_out'),
                        'confusion_matrix.csv'
                    )
                )
                # 记录错误的样本和类
                sample_names = [
                    ann['frame_dir'] for ann in dataset.video_infos
                ] # 每个样本的样本名
                pred_labels = np.argmax(outputs['output'], axis=1) # 预测值
                labels = gt_labels # 真值
                for i, name in enumerate(sample_names):
                    if pred_labels[i] != labels[i]:
                        wrong_sample_statistics.update(
                            {name: '{} but wrong predicted as {}'.format(
                                class_dict_info[labels[i]], 
                                class_dict_info[pred_labels[i]]
                            )}
                        )
                with open(
                    os.path.join(
                        eval_config.get('metric_out'),
                        'wrong_sample_statistics.json'
                    ), 'w'
                ) as f:
                    json.dump(wrong_sample_statistics, f, indent=1)                
            if metric == 't_sne_vis':
                vis_out = {
                    'target_outputs':outputs['target_output'],
                    'target_labels': outputs['target_label']
                }
                dump(
                    vis_out,
                    os.path.join(
                        eval_config.get('metric_out'),
                        't_sne_vis_out.pkl'
                    )
                )
                pca = PCA(n_components=50, random_state=400)
                tsne = TSNE(n_components=2, init='pca', n_iter=3000, random_state=400)
                pca_datas = pca.fit_transform(
                    np.array(outputs['target_output']) # [N, C]
                ) # 可能会出现复数
                tsne_datas = tsne.fit_transform(pca_datas.real)
                draw_pic(
                    datas= tsne_datas, labs = np.array(outputs['target_label']),
                    name = os.path.join(
                        eval_config.get('metric_out'),
                        'pca50-t_sne_vis.png'
                    )
                )                
        with open(
            os.path.join(
                eval_config.get('metric_out'),
                'test_metric.json'
            ), 'w'
        ) as f:
            json.dump(eval_results, f, indent=1)
    pass

if __name__ == '__main__':
    main()