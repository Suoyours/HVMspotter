import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..loss import build_loss, iou, ohem_batch
from ..post_processing import pa


class ThreeMap_Head(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, loss_text):
        super(ThreeMap_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               hidden_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim,
                               num_classes,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.text_loss = build_loss(loss_text)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        return out

    def get_results(self, out, img_meta, cfg):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score_vertical = torch.sigmoid(out[:, 0, :, :])
        score_horizontal = torch.sigmoid(out[:, 1, :, :])
        score_two_lines = torch.sigmoid(out[:, 2, :, :])
        text_vertical = out[:, 0, :, :] > 0
        text_horizontal = out[:, 1, :, :] > 0
        text_two_lines = out[:, 2, :, :] > 0
        score_vertical = score_vertical.data.cpu().numpy()[0].astype(np.float32)
        score_horizontal = score_horizontal.data.cpu().numpy()[0].astype(np.float32)
        score_two_lines = score_two_lines.data.cpu().numpy()[0].astype(np.float32)

        text_vertical = text_vertical.data.cpu().numpy()[0].astype(np.uint8)
        text_horizontal = text_horizontal.data.cpu().numpy()[0].astype(np.uint8)
        text_two_lines = text_two_lines.data.cpu().numpy()[0].astype(np.uint8)

        score = score_vertical + score_horizontal + score_two_lines
        outputs.update(dict(score_maps=score))
        # np.savez('/data/ys/PycharmProjects2/pan_pp.pytorch/outputs/three_maps_pic/save_maps.npz', score_vertical, score_horizontal, score)
        # np.savez('/data/ys/PycharmProjects2/save_maps.npz', score_vertical, score_horizontal, score)
        text = text_vertical + text_horizontal + text_two_lines
        label_num, label = cv2.connectedComponents(text, connectivity=4)  # label_num包含了背景，实际要-1

        # image size
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]

        label_num = np.max(label) + 1
        label = cv2.resize(label, (img_size[1], img_size[0]),
                           interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (img_size[1], img_size[0]),
                           interpolation=cv2.INTER_NEAREST)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_post_time=time.time() - start))

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        with_rec = hasattr(cfg.model, 'recognition_head')

        if with_rec:
            bboxes_h = np.zeros((1, label_num, 4), dtype=np.int32)
            instances = [[]]

        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))

            if points.shape[0] < cfg.test_cfg.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue

            if with_rec:
                tl = np.min(points, axis=0)
                br = np.max(points, axis=0) + 1
                bboxes_h[0, i] = (tl[0], tl[1], br[0], br[1])
                instances[0].append(i)

            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(bboxes=bboxes, scores=scores))
        if with_rec:
            outputs.update(
                dict(label=label, bboxes_h=bboxes_h, instances=instances))

        return outputs

    def loss(self, out, gt_vertical, gt_horizontal, gt_two_lines, training_masks):
        # output
        vertical = out[:, 0, :, :]
        horizontal = out[:, 1, :, :]
        two_lines = out[:, 2, :, :]

        # gt_vertical loss
        selected_masks1 = ohem_batch(vertical, gt_vertical, training_masks)
        loss_vertical = self.text_loss(vertical,
                                       gt_vertical,
                                       selected_masks1,
                                       reduce=False)
        iou_vertical = iou((vertical > 0).long(),
                           gt_vertical,
                           training_masks,
                           reduce=False)
        losses = dict(loss_vertical=loss_vertical, iou_vertical=iou_vertical)

        # gt_horizontal loss
        selected_masks2 = ohem_batch(horizontal, gt_horizontal, training_masks)
        loss_horizontal = self.text_loss(horizontal,
                                         gt_horizontal,
                                         selected_masks2,
                                         reduce=False)
        iou_horizontal = iou((horizontal > 0).long(),
                             gt_horizontal,
                             training_masks,
                             reduce=False)
        losses.update(dict(loss_horizontal=loss_horizontal, iou_horizontal=iou_horizontal))

        # gt_two_lines loss
        selected_masks3 = ohem_batch(two_lines, gt_two_lines, training_masks)
        loss_two_lines = self.text_loss(two_lines,
                                        gt_two_lines,
                                        selected_masks3,
                                        reduce=False)
        iou_two_lines = iou((two_lines > 0).long(),
                            gt_two_lines,
                            training_masks,
                            reduce=False)
        losses.update(dict(loss_two_lines=loss_two_lines, iou_two_lines=iou_two_lines))
        return losses
