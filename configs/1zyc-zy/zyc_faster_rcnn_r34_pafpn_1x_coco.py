_base_ = './zyc_faster_rcnn_r34_fpn_1x_coco.py'


model = dict(
    neck=dict(
        type='PAFPN',
        #in_channels=[256, 512, 1024, 2048],   50ceng
        in_channels=[64,128,256,512],
        out_channels=256,
        num_outs=5))






