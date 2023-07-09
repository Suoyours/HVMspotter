import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from .multipos import MultiPosCrossEntropyLoss


class ContrastLoss(nn.Module):
    def __init__(self, device='cuda'):
        # input_f  (B, 27, 768)
        # target  (B, 27)
        super().__init__()
        self.device = device
        self.mul_pos_loss = MultiPosCrossEntropyLoss(device=self.device)

    def compute_loss(self, input1, input2):
        # num_text = len(labels)
        # global char_f
        # global char_f
        num_t1 = input1.shape[0]
        num_t2 = input2.shape[0]
        num_sum = num_t1 + num_t2
        char_f = torch.cat((input1, input2), dim=0)

        # char_f = F.normalize(char_f, p=2, dim=1)
        sim_f = torch.matmul(char_f, char_f.T)

        label_encode = torch.zeros((num_sum, 2), dtype=torch.float64).to(torch.device(self.device))
        for i in range(num_sum):
            if i < num_t1:
                label_encode[i][0] = 1
            else:
                label_encode[i][1] = 1

        label_sim = torch.matmul(label_encode, label_encode.T)

        return self.mul_pos_loss(sim_f, label_sim)


class EmbLoss_contrast(nn.Module):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_contrast, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = (1.5, 1)
        self.contrastLoss = ContrastLoss().to(self.device)
        self.hidden_size = feature_dim
        self.layer_norm_eps = 1e-5
        self.layernorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps).cuda()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.2).cuda()
        self.temperature = 0.07
        self.loss_contrast = torch.nn.CrossEntropyLoss().cuda()

    def get_present_emb(delf, inputs):
        input_num = inputs.shape[1]
        get_num = (input_num // 100) + 1
        row_rand_aa = np.arange(input_num)
        np.random.shuffle(row_rand_aa)
        get_samples = inputs[:, row_rand_aa[0:get_num]]
        return get_samples

    def get_cosine_loss(self, input1, input2):

        input_dim = input1.shape[1]
        len1 = input1.shape[0]
        len2 = input2.shape[0]
        input_all = torch.cat((input1, input2), 0)

        input_all_expan1 = input_all.repeat(1, input_all.shape[0])
        input_all_expan1 = input_all_expan1.reshape(-1, input_dim)

        input_all_expan2 = input_all.repeat(input_all.shape[0], 1)

        # cosine_target = -input_all.new_ones((len1 + len2) ** 2, dtype=torch.int32)
        cosine_target = input_all.new_zeros((len1 + len2) ** 2, dtype=torch.int32)

        for i in range(len1 + len2):
            for j in range(len1 + len2):
                if i < len1:
                    if j < len1:
                        cosine_target[i * (len1 + len2) + j] = 1
                    else:
                        cosine_target[i * (len1 + len2) + j] = -1
                else:
                    # if j < len1:
                    cosine_target[i * (len1 + len2) + j] = -1
                    # else:
                    #     cosine_target[i * (len1 + len2) + j] = 1
        return self.cosine_loss(input_all_expan1, input_all_expan2, cosine_target)

    def get_cosine_loss_one(self, input1, input_kernel):
        len1 = input1.shape[0]

        input_kernel_expan = input_kernel.repeat(len1, 1)
        cosine_target = input1.new_ones(len1, dtype=torch.int32)

        return self.cosine_loss(input1, input_kernel_expan, cosine_target)

    def get_contrast_loss(self, input_a, input_b):
        input_a = F.normalize(input_a, dim=1)
        input_b = F.normalize(input_b, dim=1)
        len1 = input_a.shape[0]
        len2 = input_b.shape[0]

        similarity_matrix_aa = torch.matmul(input_a, input_a.T)
        similarity_matrix_ab = torch.matmul(input_a, input_b.T)

        similarity_matrix_aa1 = similarity_matrix_aa.reshape(-1, 1)
        similarity_matrix_ab1 = similarity_matrix_ab.repeat((1, len1))
        similarity_matrix_ab1 = similarity_matrix_ab1.reshape(-1, len2)
        logits = torch.cat((similarity_matrix_aa1, similarity_matrix_ab1), dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        loss = self.loss_contrast(logits, labels)
        # print(similarity_matrix_aa1)
        # print(similarity_matrix_ab1)
        # print(logits)
        # print(loss)
        return loss

    def forward_single(self, emb, instance, kernel, training_mask, bboxes):
        emb = emb.permute(1, 2, 0)
        # emb_input = emb_input.reshape(736, 736, 4)

        emb = self.layernorm(emb)
        emb = emb.permute(2, 0, 1)

        training_mask = (training_mask > 0.5).long()
        kernel = (kernel > 0.5).long()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(self.feature_dim, -1)

        unique_labels, unique_ids = torch.unique(instance_kernel,
                                                 sorted=True,
                                                 return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0

        emb_mean = emb.new_zeros((self.feature_dim, num_instance),
                                 dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)

        l_pix = emb.new_zeros(num_instance, dtype=torch.float32)  # bug
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            # ind_k = instance_kernel == lb
            # emb_k = emb[:, ind_k]

            # sim_value = torch.matmul(emb_.T, emb_mean[:, i])
            # l_pix[i] = torch.mean(1/sim_value)
            # l_pix[i] = torch.mean(self.feature_dim-sim_value)
            l_pix[i] = self.get_cosine_loss_one(emb_.T, emb_mean[:, i].T)
            # l_pix[i] = torch.mean(torch.exp(-sim_value))
        l_pix = torch.sum(l_pix[1:])

        if num_instance > 2:
            emb_zero = emb.new_zeros(self.feature_dim,
                                     dtype=torch.float32)
            emb_t_mean = emb.new_zeros((self.feature_dim, num_instance), dtype=torch.float32)
            l_contrast = emb.new_zeros(num_instance, dtype=torch.float32)  # bug
            for i, lb in enumerate(unique_labels):
                if lb == 0:
                    continue
                ind_t = instance == lb
                # ins_i = emb[:, ind_t]
                # mean_a = torch.mean(ins_i, dim=1)
                emb_t_mean[:, i] = torch.mean(emb[:, ind_t], dim=1)
            for i, lb_i in enumerate(unique_labels):
                if lb_i == 0:
                    continue
                ind_i = instance == lb_i
                emb_i = emb[:, ind_i]
                emb_i_sample = self.get_present_emb(emb_i)
                emb_i_sample[:, 0] = emb_t_mean[:, i]  # 第i个文本实例的平均特征向量赋值给第一列
                emb_mean_i = emb_t_mean.clone()
                emb_mean_i[:, i] = emb_zero  # 第i个平均特征向量赋值赋值0向量

                emb_i_sample = emb_i_sample.permute(1, 0)  # 对比正例
                emb_mean_i = emb_mean_i.permute(1, 0)  # 对比负例

                # l_contrast[i] = self.contrastLoss.compute_loss(emb_i_sample, emb_mean_i)
                # loss_c = self.get_cosine_loss(emb_i_sample, emb_mean_i)
                loss_c = self.get_contrast_loss(emb_i_sample, emb_mean_i)
                 # = loss_c
                l_contrast[i] = loss_c
            l_contrast_avg = torch.sum(l_contrast)
        else:
            l_contrast_avg = 0
        # print(l_contrast_avg)

        # l_contrast_avg = 0
        l_pix = self.weights[0] * l_pix
        l_dis = self.weights[1] * l_contrast_avg
        # l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        print('l_pix:{}  l_dis:{}'.format(l_pix, l_dis))
        loss = l_pix + l_dis
        return loss

    def forward(self,
                emb,
                instance,
                kernel,
                training_mask,
                bboxes,
                reduce=True):
        loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

        for i in range(loss_batch.size(0)):
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i],
                                                training_mask[i], bboxes[i])

        loss_batch = self.loss_weight * loss_batch

        if reduce:
            loss_batch = torch.mean(loss_batch)

        return loss_batch
