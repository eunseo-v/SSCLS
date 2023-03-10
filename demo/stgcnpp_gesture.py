graph = 'handmp'
modality = 'j'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        in_channels=2,
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        num_stages=6,
        down_stages=[6],
        inflate_stages=[6],
        graph_cfg=dict(layout=graph, mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=40, in_channels=128))

test_pipeline = [
    dict(type='PreNormalize2D', threshold=0, mode='auto'),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSample', clip_len=10, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
