import queue

import numpy as np
import cv2
from scipy import spatial
from sklearn import preprocessing


def _pa_copy(kernel, emb, label, cc, label_num, min_area=0):
    # kernel:text（预测，未处理）
    # embemb四维向量图（全图）
    # label:label（0，1，2，3，···m）
    # cc:经过连通域的text图（0，1，2，3，···n）
    # label_num:经过连通域的kernel图的总数
    # min_area:过滤的最小面积
    # cv2.imshow('kernel', kernel[0]*255)
    # cv2.waitKey(0)
    pred = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    mean_emb = np.zeros((label_num, 4), dtype=np.float32)
    area = np.full((label_num,), -1, dtype=np.float32)
    flag = np.zeros((label_num,), dtype=np.int32)
    inds = np.zeros((label_num, label.shape[0], label.shape[1]), dtype=np.uint8)
    p = np.zeros((label_num, 2), dtype=np.int32)

    max_rate = 1024
    for i in range(1, label_num):
        ind = label == i
        inds[i] = ind

        area[i] = np.sum(ind)  # 614.0

        if area[i] < min_area:  # 0
            label[ind] = 0
            continue

        px, py = np.where(ind)  # (614,),(614,)
        p[i] = (px[0], py[0])  # px[0]==min(px), py[0]==min(py)

        for j in range(1, i):
            if area[j] < min_area:
                continue
            if cc[p[i, 0], p[i, 1]] != cc[p[j, 0], p[j, 1]]:  # 完整的text预测图中没有把两个kernel合并成一个
                continue
            rate = area[i] / area[j]
            if rate < 1 / max_rate or rate > max_rate:
                flag[i] = 1
                mean_emb[i] = np.mean(emb[:, ind], axis=1)

                if flag[j] == 0:
                    flag[j] = 1
                    mean_emb[j] = np.mean(emb[:, inds[j].astype(np.bool)], axis=1)

    que = queue.Queue(maxsize=0)
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    points = np.array(np.where(label > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        que.put((x, y, l))
        pred[x, y] = l

    while not que.empty():
        (x, y, l) = que.get()

        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]
            if tmpx < 0 or tmpx >= label.shape[0] or tmpy < 0 or tmpy >= label.shape[1]:
                continue
            if kernel[0, tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:  # 完整text预测图中这个点值为0或者已经扩充过了
                continue
            if flag[l] == 1 and np.linalg.norm(emb[:, tmpx, tmpy] - mean_emb[l]) > 3:  # 论文里是6
                continue

            # que.put((tmpx, tmpy))
            pred[tmpx, tmpy] = l

    return pred


def _pa(kernel, emb, label, cc, label_num, min_area=0):
    # kernel:text（预测，未处理）
    # embemb四维向量图（全图）
    # label:label（0，1，2，3，···m）
    # cc:经过连通域的text图（0，1，2，3，···n）
    # label_num:经过连通域的kernel图的总数
    # min_area:过滤的最小面积
    # cv2.imshow('kernel', kernel[0]*255)
    # cv2.waitKey(0)
    pred = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    mean_emb = np.zeros((label_num, 4), dtype=np.float32)
    area = np.full((label_num,), -1, dtype=np.float32)
    flag = np.zeros((label_num,), dtype=np.int32)
    inds = np.zeros((label_num, label.shape[0], label.shape[1]), dtype=np.uint8)
    p = np.zeros((label_num, 2), dtype=np.int32)
    kernel_i_in_text = np.zeros((label_num,), dtype=np.int32)

    max_rate = 1024
    for i in range(1, label_num):
        ind = label == i
        inds[i] = ind

        area[i] = np.sum(ind)  # 614.0

        if area[i] < min_area:  # 0
            label[ind] = 0
            continue

        px, py = np.where(ind)  # (614,),(614,)
        p[i] = (px[0], py[0])  # px[0]==min(px), py[0]==min(py)

        for j in range(1, i):
            if area[j] < min_area:
                continue
            if cc[p[i, 0], p[i, 1]] != cc[p[j, 0], p[j, 1]]:  # 完整的text预测图中没有把两个kernel合并成一个
                continue
            rate = area[i] / area[j]
            # if rate < 1 / max_rate or rate > max_rate:
            if 1 / max_rate < rate < max_rate:
                flag[i] = 1
                mean_emb[i] = np.mean(emb[:, ind], axis=1)
                kernel_i_in_text[i] = cc[p[i, 0], p[i, 1]]
                kernel_i_in_text[j] = cc[p[j, 0], p[j, 1]]
                if flag[j] == 0:
                    flag[j] = 1
                    mean_emb[j] = np.mean(emb[:, inds[j].astype(np.bool)], axis=1)

    # 把没有重叠部分的text的值改成kernel的
    predict = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    for i in range(1, label_num):
        if kernel_i_in_text[i] == 0:
            ind_k = label == i
            px_k, py_k = np.where(ind_k)  # (614,),(614,)
            x_k, y_k = (px_k[0], py_k[0])

            text_i_before = cc[x_k, y_k]
            ind_t = cc == text_i_before
            predict[ind_t] = i
        else:
            ind_k = label == i
            predict[ind_k] = i
    # cv2.imshow('kernel_copy', predict*15)
    # cv2.waitKey(0)

    kernel_i_in_text_copy = kernel_i_in_text.copy()
    for i in range(label_num):
        if kernel_i_in_text_copy[i] != 0:
            kernel_i_in_text_copy[i] = 0
            ind_tmp = inds[i]
            for j in range(i, label_num):
                if kernel_i_in_text_copy[j] == kernel_i_in_text[i]:
                    ind_tmp += inds[j]
                    kernel_i_in_text_copy[j] = 0
            tmp_label = kernel_i_in_text[i]
            ind_tmp_text = cc == tmp_label
            tmp_map = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
            tmp_map[ind_tmp_text] = 1
            ind_tmp_beside = tmp_map - ind_tmp
            ind_tmp_beside1 = ind_tmp_beside == 1
            # predict2 = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
            # predict2[ind_tmp_beside1] = 255
            # cv2.imshow('kernel_copy', predict2)
            # cv2.waitKey(0)
            # print(ind_tmp_beside)
            px_f, py_f = np.where(ind_tmp_beside1)  # (614,),(614,)

            for k in range(len(px_f)):
                vct_k = emb[:, px_f[k], py_f[k]]
                final_distance = 100
                final_point = i
                for m in range(1, label_num):
                    if kernel_i_in_text[m] == kernel_i_in_text[i]:
                        tmp_distance = np.linalg.norm(vct_k - mean_emb[m])
                        if tmp_distance < final_distance:
                            final_distance = tmp_distance
                            final_point = m
                predict[px_f[k], py_f[k]] = final_point



    #
    #
    # que = queue.Queue(maxsize=0)
    # dx = [-1, 1, 0, 0]
    # dy = [0, 0, -1, 1]
    #
    # points = np.array(np.where(label > 0)).transpose((1, 0))
    # for point_idx in range(points.shape[0]):
    #     x, y = points[point_idx, 0], points[point_idx, 1]
    #     l = label[x, y]
    #     que.put((x, y, l))
    #     pred[x, y] = l
    #
    # while not que.empty():
    #     (x, y, l) = que.get()
    #
    #     for j in range(4):
    #         tmpx = x + dx[j]
    #         tmpy = y + dy[j]
    #         if tmpx < 0 or tmpx >= label.shape[0] or tmpy < 0 or tmpy >= label.shape[1]:
    #             continue
    #         if kernel[0, tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:  # 完整text预测图中这个点值为0或者已经扩充过了
    #             continue
    #         if flag[l] == 1 and np.linalg.norm(emb[:, tmpx, tmpy] - mean_emb[l]) > 3:  # 论文里是6
    #             continue
    #
    #         # que.put((tmpx, tmpy))
    #         pred[tmpx, tmpy] = l

    return predict

def _pa_contrast(kernel, emb, label, cc, label_num, min_area=0):
    # kernel:text（预测，未处理）
    # embemb四维向量图（全图）
    # label:label（0，1，2，3，···m）
    # cc:经过连通域的text图（0，1，2，3，···n）
    # label_num:经过连通域的kernel图的总数
    # min_area:过滤的最小面积
    # cv2.imshow('kernel', kernel[0]*255)
    # cv2.waitKey(0)
    pred = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    mean_emb = np.zeros((label_num, 20), dtype=np.float32)
    area = np.full((label_num,), -1, dtype=np.float32)
    flag = np.zeros((label_num,), dtype=np.int32)
    inds = np.zeros((label_num, label.shape[0], label.shape[1]), dtype=np.uint8)
    p = np.zeros((label_num, 2), dtype=np.int32)
    kernel_i_in_text = np.zeros((label_num,), dtype=np.int32)

    max_rate = 1024
    for i in range(1, label_num):
        ind = label == i
        inds[i] = ind

        area[i] = np.sum(ind)  # 614.0

        if area[i] < min_area:  # 0
            label[ind] = 0
            continue

        px, py = np.where(ind)  # (614,),(614,)
        p[i] = (px[0], py[0])  # px[0]==min(px), py[0]==min(py)

        for j in range(1, i):
            if area[j] < min_area:
                continue
            if cc[p[i, 0], p[i, 1]] != cc[p[j, 0], p[j, 1]]:  # 完整的text预测图中没有把两个kernel合并成一个
                continue
            rate = area[i] / area[j]
            # if rate < 1 / max_rate or rate > max_rate:
            if 1 / max_rate < rate < max_rate:
                flag[i] = 1
                mean_emb[i] = np.mean(emb[:, ind], axis=1)
                kernel_i_in_text[i] = cc[p[i, 0], p[i, 1]]
                kernel_i_in_text[j] = cc[p[j, 0], p[j, 1]]
                if flag[j] == 0:
                    flag[j] = 1
                    mean_emb[j] = np.mean(emb[:, inds[j].astype(np.bool)], axis=1)

    # 把没有重叠部分的text的值改成kernel的
    predict = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    for i in range(1, label_num):
        if kernel_i_in_text[i] == 0:
            ind_k = label == i
            px_k, py_k = np.where(ind_k)  # (614,),(614,)
            if len(px_k) == 0:
                continue
            x_k, y_k = (px_k[0], py_k[0])

            text_i_before = cc[x_k, y_k]
            ind_t = cc == text_i_before
            predict[ind_t] = i
        else:
            ind_k = label == i
            predict[ind_k] = i
    # cv2.imshow('kernel_copy', predict*15)
    # cv2.waitKey(0)

    # 把重叠部分，通过相似度聚类算法，将外围的像素分给对应的文本kernel
    mean_emb = preprocessing.normalize(mean_emb, norm='l2')
    kernel_i_in_text_copy = kernel_i_in_text.copy()
    for i in range(label_num):
        if kernel_i_in_text_copy[i] != 0:
            kernel_i_in_text_copy[i] = 0
            ind_tmp = inds[i]
            for j in range(i, label_num):
                if kernel_i_in_text_copy[j] == kernel_i_in_text[i]:
                    ind_tmp += inds[j]
                    kernel_i_in_text_copy[j] = 0
            tmp_label = kernel_i_in_text[i]
            ind_tmp_text = cc == tmp_label
            tmp_map = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
            tmp_map[ind_tmp_text] = 1
            ind_tmp_beside = tmp_map - ind_tmp
            ind_tmp_beside1 = ind_tmp_beside == 1
            # predict2 = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
            # predict2[ind_tmp_beside1] = 255
            # cv2.imshow('kernel_copy', predict2)
            # cv2.waitKey(0)
            # print(ind_tmp_beside)
            px_f, py_f = np.where(ind_tmp_beside1)  # (614,),(614,)

            for k in range(len(px_f)):
                vct_k = emb[:, px_f[k], py_f[k]]
                vct_k = preprocessing.normalize(vct_k.reshape(1, -1), norm='l2').reshape(-1)
                final_sim = 0
                final_point = i
                for m in range(1, label_num):
                    if kernel_i_in_text[m] == kernel_i_in_text[i]:
                        # tmp_distance = 1 - spatial.distance.cosine(vct_k, mean_emb[m])
                        tmp_sim = np.matmul(vct_k, mean_emb[m])
                        if tmp_sim > final_sim:
                            final_sim = tmp_sim
                            final_point = m
                predict[px_f[k], py_f[k]] = final_point

    # 针对同一个文本，同一个kernel分裂为多个，将它们合并在一起
    kernel_i_in_text_copy = kernel_i_in_text.copy()
    for i in range(label_num):
        if kernel_i_in_text_copy[i] != 0:
            kernel_i_in_text_copy[i] = 0
            for j in range(i, label_num):
                if kernel_i_in_text_copy[j] == kernel_i_in_text[i]:
                    tmp_sim = np.matmul(mean_emb[i], mean_emb[j])
                    if tmp_sim > 0.97:
                        kernel_i_in_text_copy[j] = 0
                        predict[predict == j] = i

    return predict


def re_pa(kernels, emb, min_area=2):  # (2, 184, 328)，(4, 184, 328)，0
    # kernels[0]是预测的text完整图，kernels[1]是预测的以0.5比例shrink的kernel图
    # kernel_copy = kernels.copy()
    # text01 = kernel_copy[0]*255
    # kernel01 = kernel_copy[1]*255
    # cv2.imshow('text_copy', text01)
    # cv2.waitKey(0)
    # cv2.imshow('kernel_copy', kernel01)
    # cv2.waitKey(0)
    _, cc = cv2.connectedComponents(kernels[0], connectivity=4)
    label_num, label = cv2.connectedComponents(kernels[1], connectivity=4)  # label_num包含了背景，实际要-1

    return _pa_contrast(kernels[:-1], emb, label, cc, label_num, min_area)


# def re_pa1(kernels, emb, min_area=2):  # (2, 184, 328)，(4, 184, 328)，0
#     # kernels[0]是预测的text完整图，kernels[1]是预测的以0.5比例shrink的kernel图
#     kernel_copy = kernels.copy()
#     emb_copy = emb.copy()
#     # cv2.imshow('text_copy', kernel_copy[0])
#     # cv2.waitKey(0)
#     cv2.imshow('kernel_copy', kernel_copy[1])
#     cv2.waitKey(0)
#
#     _, cc = cv2.connectedComponents(kernel_copy[0], connectivity=4)
#     label_num, label = cv2.connectedComponents(kernel_copy[1], connectivity=4)  # label_num包含了背景，实际要-1
#
#     return _pa(kernels[:-1], emb_copy, label, cc, label_num, min_area)
#     # (1, 184, 328)，(4, 184, 328)，(184, 328)，(184, 328)，2,3,0
#     # kernels[0].shape=(184, 328), kernels[:-1].shape=(1, 184, 328)


# kernel = np.load('/data/YOUSUO/PycharmProjects/pan_pp.pytorch/models/post_processing/kernels.npy')
# emb = np.load('/data/YOUSUO/PycharmProjects/pan_pp.pytorch/models/post_processing/emb.npy')
# output2 = re_pa(kernel, emb)
# print(output2)

# kernels = cv2.imread('/data/YOUSUO/PycharmProjects/pan_pp.pytorch/data/total_text/Images/Test/img10.jpg',
#                      cv2.IMREAD_GRAYSCALE)
# ret, binary = cv2.threshold(kernels, 180, 1, cv2.THRESH_BINARY)
# _, cc = cv2.connectedComponents(binary, connectivity=4)
# print(cc)
