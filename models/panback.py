import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .head import build_head
from .neck import build_neck
from .utils import Conv_BN_ReLU


class OneMap(nn.Module):
    def __init__(self, backbone, neck, detection_head):
        super(OneMap, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                training_masks=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, training_masks)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        return outputs


class TwoMap(nn.Module):
    def __init__(self, backbone, neck, detection_head, recognition_head=None):
        super(TwoMap, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)

        if recognition_head:
            self.rec_head = build_head(recognition_head)
            self.rec_head2 = build_head(recognition_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_vertical=None,
                gt_horizontal=None,
                training_masks=None,
                img_metas=None,
                cfg=None,
                gt_texts=None,
                gt_kernels=None,
                gt_instances=None,
                gt_bboxes=None,
                gt_words=None,
                word_masks=None,
                data_rcg_extra=None
                ):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_vertical, gt_horizontal, training_masks)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        if self.rec_head is not None:
            if self.training:
                if cfg.train_cfg.use_ex:
                    x_crops_h, gt_words_h = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        gt_instances * gt_kernels[:, 0] * training_masks,
                        gt_bboxes, gt_words, data_rcg_extra['word_masks_h'])
                    x_crops_v, gt_words_v = self.rec_head2.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        gt_instances * gt_kernels[:, 0] * training_masks,
                        gt_bboxes, gt_words, data_rcg_extra['word_masks_v'])
                else:
                    x_crops_h, gt_words_h = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        gt_instances * training_masks, gt_bboxes, gt_words,
                        data_rcg_extra['word_masks_h'])
                    x_crops_v, gt_words_v = self.rec_head2.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        gt_instances * training_masks, gt_bboxes, gt_words,
                        data_rcg_extra['word_masks_v'])

                if x_crops_h is not None:
                    out_rec_h = self.rec_head(x_crops_h, gt_words_h)
                    loss_rec_h = self.rec_head.loss(out_rec_h,
                                                  gt_words_h,
                                                  reduce=True)
                else:
                    loss_rec_h = {
                        'loss_rec': f.new_full((1, ), -1, dtype=torch.float32),
                        'acc_rec': f.new_full((1, ), -1, dtype=torch.float32)
                    }
                if x_crops_v is not None:
                    out_rec_v = self.rec_head2(x_crops_v, gt_words_v)
                    loss_rec_v = self.rec_head2.loss(out_rec_v,
                                                  gt_words_v,
                                                  reduce=True)
                else:
                    loss_rec_v = {
                        'loss_rec': f.new_full((1, ), -1, dtype=torch.float32),
                        'acc_rec': f.new_full((1, ), -1, dtype=torch.float32)
                    }
                loss_rec = {'loss_rec': loss_rec_h['loss_rec'] + loss_rec_v['loss_rec'],
                            'acc_rec': loss_rec_h['acc_rec'] + loss_rec_v['acc_rec']}
                outputs.update(loss_rec)
            else:
                if len(det_res['bboxes']) > 0:
                    x_crops, _ = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        f.new_tensor(det_res['label'],
                                     dtype=torch.long).unsqueeze(0),
                        bboxes=f.new_tensor(det_res['bboxes_h'],
                                            dtype=torch.long),
                        unique_labels=det_res['instances'])
                    words, word_scores = self.rec_head.forward(x_crops)
                else:
                    words = []
                    word_scores = []

                if cfg.report_speed:
                    torch.cuda.synchronize()
                    outputs.update(dict(rec_time=time.time() - start))
                outputs.update(
                    dict(words=words, word_scores=word_scores, label=''))

        return outputs


class TwoMapTowRec(nn.Module):
    def __init__(self, backbone, neck, detection_head, recognition_head=None):
        super(TwoMapTowRec, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)

        if recognition_head:
            self.rec_head1 = build_head(recognition_head)

            # recognition_head2 = recognition_head
            # recognition_head2['feature_size'] = ()
            self.rec_head2 = build_head(recognition_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_vertical=None,
                gt_horizontal=None,
                training_masks=None,
                img_metas=None,
                cfg=None,
                gt_texts=None,
                gt_kernels=None,
                gt_instances=None,
                gt_bboxes=None,
                gt_words=None,
                word_masks=None,
                ):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_vertical, gt_horizontal, training_masks)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        if self.rec_head is not None:
            if self.training:
                if cfg.train_cfg.use_ex:
                    x_crops, gt_words = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        gt_instances * gt_kernels[:, 0] * training_masks,
                        gt_bboxes, gt_words, word_masks)
                else:
                    x_crops, gt_words = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        gt_instances * training_masks, gt_bboxes, gt_words,
                        word_masks)

                if x_crops is not None:
                    out_rec = self.rec_head(x_crops, gt_words)
                    loss_rec = self.rec_head.loss(out_rec,
                                                  gt_words,
                                                  reduce=False)
                else:
                    loss_rec = {
                        'loss_rec': f.new_full((1, ), -1, dtype=torch.float32),
                        'acc_rec': f.new_full((1, ), -1, dtype=torch.float32)
                    }
                outputs.update(loss_rec)
            else:
                if len(det_res['bboxes']) > 0:
                    x_crops, _ = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        f.new_tensor(det_res['label'],
                                     dtype=torch.long).unsqueeze(0),
                        bboxes=f.new_tensor(det_res['bboxes_h'],
                                            dtype=torch.long),
                        unique_labels=det_res['instances'])
                    words, word_scores = self.rec_head.forward(x_crops)
                else:
                    words = []
                    word_scores = []

                if cfg.report_speed:
                    torch.cuda.synchronize()
                    outputs.update(dict(rec_time=time.time() - start))
                outputs.update(
                    dict(words=words, word_scores=word_scores, label=''))

        return outputs


class ThreeMap(nn.Module):
    def __init__(self, backbone, neck, detection_head):
        super(ThreeMap, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_vertical=None,
                gt_horizontal=None,
                gt_two_lines=None,
                training_masks=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_vertical, gt_horizontal, gt_two_lines, training_masks)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        return outputs

