_base_ = './zyc_faster_rcnn_r50_fpn_1x_coco.py'


model = dict(
    neck=dict(
        type='GPAFPN',
        in_channels=[256, 512, 1024, 2048],   #50ceng
        #in_channels=[64,128,256,512],
        out_channels=256,
        num_outs=5,
        attention_ways='SE',
        using_ssac=True,
        where_using_ssac=3,
    ))
