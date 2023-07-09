import torch
import torch.nn as nn
from .multipos import MultiPosCrossEntropyLoss
import torch.nn.functional as F


dict96 = {'[GO]': 0, '[s]': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11, 'a': 12, 'b': 13, 'c': 14, 'd': 15, 'e': 16, 'f': 17, 'g': 18, 'h': 19, 'i': 20, 'j': 21, 'k': 22, 'l': 23, 'm': 24, 'n': 25, 'o': 26, 'p': 27, 'q': 28, 'r': 29, 's': 30, 't': 31, 'u': 32, 'v': 33, 'w': 34, 'x': 35, 'y': 36, 'z': 37, 'A': 38, 'B': 39, 'C': 40, 'D': 41, 'E': 42, 'F': 43, 'G': 44, 'H': 45, 'I': 46, 'J': 47, 'K': 48, 'L': 49, 'M': 50, 'N': 51, 'O': 52, 'P': 53, 'Q': 54, 'R': 55, 'S': 56, 'T': 57, 'U': 58, 'V': 59, 'W': 60, 'X': 61, 'Y': 62, 'Z': 63, '!': 64, '"': 65, '#': 66, '$': 67, '%': 68, '&': 69, "'": 70, '(': 71, ')': 72, '*': 73, '+': 74, ',': 75, '-': 76, '.': 77, '/': 78, ':': 79, ';': 80, '<': 81, '=': 82, '>': 83, '?': 84, '@': 85, '[': 86, '\\': 87, ']': 88, '^': 89, '_': 90, '`': 91, '{': 92, '|': 93, '}': 94, '~': 95}

target_n = torch.tensor([[0, 69, 20, 25,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0, 31, 19, 16, 11,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0, 32, 23, 31,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0, 55, 16, 79,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0]])
input_n = torch.rand(4, 27, 768)
# from torch.autograd import Variable


class ContrastLoss(nn.Module):
    def __init__(self, device='cuda'):
        # input_f  (B, 27, 768)
        # target  (B, 27)
        self.device=device
        super().__init__()
        self.char_dic = torch.zeros(96, 768).to(torch.device(self.device))
        self.layer_norm1 = nn.LayerNorm([768])
        self.layer_norm2 = nn.LayerNorm([768])
        # self.char_dic = Variable(self.char_dic, requires_grad=False)

    def compute_loss(self, input_f, target):
        # num_text = len(labels)
        input_f = self.layer_norm1(input_f)
        total_positive_loss = torch.FloatTensor([0]).to(torch.device(self.device))
        char_tem = []
        char_tem_dic = torch.zeros(96, 768).to(torch.device(self.device))
        for i in range(self.char_dic.shape[0]):
            chardic11 = self.char_dic[i].clone()
            char_tem.append(chardic11.reshape(1, -1))
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                index = target[i][j]
                # a = char_tem[index]
                # b = input_f[i][j].reshape(1, -1)
                # print(input_f.shape)
                # print(a.shape)
                # print(b.shape)
                char_tem[index] = torch.cat((char_tem[index], input_f[i][j].reshape(1, -1)), 0)
        # compute positive loss/data/YOUSUO/pycharm _projects/deep-text-recognition-benchmark-master
        for i in range(len(char_tem)):
            posi_dis = torch.matmul(char_tem[i], char_tem[i].transpose(0, 1))/768
            # a = torch.sum(posi_dis)
            total_positive_loss = total_positive_loss.clone() + torch.sum(posi_dis)
            # print(posi_dis)
        # compute negative loss
        for i in range(1, len(char_tem)):
            # u0 = torch.sum(char_tem[i], dim=0)
            llen = char_tem[i].shape[0]
            char_tem_dic[i] = torch.sum(char_tem[i], dim=0) / llen
            with torch.no_grad():
                self.char_dic[i] = self.char_dic[i].clone() + (char_tem_dic[i] * 0.1)
        with torch.no_grad():
            self.char_dic = self.layer_norm1(self.char_dic)
        total_negative_sim = torch.matmul(self.char_dic[1:], self.char_dic[1:].transpose(0, 1)) / 768
        total_negative_loss = torch.sum(total_negative_sim)

        return total_negative_loss - total_positive_loss


class ContrastLoss1(nn.Module):
    def __init__(self):
        # input_f  (B, 27, 768)
        # target  (B, 27)
        super().__init__()
        self.char_dic = torch.zeros(96, 768).to(torch.device('cuda'))
        self.layer_norm1 = nn.LayerNorm([768])
        self.layer_norm2 = nn.LayerNorm([768])
        # self.char_dic = Variable(self.char_dic, requires_grad=False)

    def compute_loss(self, input_f, target):
        # num_text = len(labels)
        input_f = self.layer_norm1(input_f)
        total_positive_loss = torch.FloatTensor([0]).to(torch.device('cuda'))
        char_tem = []
        char_tem_dic = torch.zeros(96, 768).to(torch.device('cuda'))
        for i in range(self.char_dic.shape[0]):
            chardic11 = self.char_dic[i].clone()
            char_tem.append(chardic11.reshape(1, -1))
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                index = target[i][j]
                # a = char_tem[index]
                # b = input_f[i][j].reshape(1, -1)
                # print(input_f.shape)
                # print(a.shape)
                # print(b.shape)
                char_tem[index] = torch.cat((char_tem[index], input_f[i][j].reshape(1, -1)), 0)
        # compute positive loss
        for i in range(len(char_tem)):
            posi_dis = torch.exp(torch.matmul(char_tem[i], char_tem[i].transpose(0, 1))/768)
            # a = torch.sum(posi_dis)
            total_positive_loss = total_positive_loss.clone() + torch.sum(posi_dis)
            # print(posi_dis)
        # compute negative loss
        for i in range(1, len(char_tem)):
            # u0 = torch.sum(char_tem[i], dim=0)
            llen = char_tem[i].shape[0]
            char_tem_dic[i] = torch.sum(char_tem[i], dim=0) / llen
            with torch.no_grad():
                self.char_dic[i] = self.char_dic[i].clone() + (char_tem_dic[i] * 0.1)
        with torch.no_grad():
            self.char_dic = self.layer_norm1(self.char_dic)
        total_negative_sim = torch.exp(torch.matmul(self.char_dic[1:], self.char_dic[1:].transpose(0, 1)) / 768)
        total_negative_loss = torch.sum(total_negative_sim)

        return -torch.log(total_positive_loss/total_negative_loss)


class ContrastLoss3(nn.Module):
    def __init__(self, device='cuda'):
        # input_f  (B, 27, 768)
        # target  (B, 27)
        super().__init__()
        self.device = device
        self.mul_pos_loss = MultiPosCrossEntropyLoss(device=self.device)

    def compute_loss(self, input_f, target):
        # num_text = len(labels)
        # global char_f
        # global char_f
        char_f = input_f[0, 0, :].reshape(-1, input_f.shape[2])
        char_target = target[0][0].reshape(1, -1)
        for i in range(1, 27):
            if target[0][i] == 0:
                char_f = torch.cat((char_f, input_f[0, 1:i, :].reshape(i-1, -1)), 0)
                char_target = torch.cat((char_target, target[0, 1:i].reshape(1, -1)), 1)
                break
        for i in range(1, input_f.shape[0]):
            for j in range(1, 27):
                if target[i][j] == 0:
                    char_f = torch.cat((char_f, input_f[i, :j, :].reshape(j, -1)), 0)
                    char_target = torch.cat((char_target, target[i, :j].reshape(1, -1)), 1)
                    break

        char_f = F.normalize(char_f, p=2, dim=1)
        sim_f = torch.matmul(char_f, char_f.T)
        char_target = char_target.view(-1)

        label_encode = torch.zeros((char_target.shape[0], 96), dtype=torch.float64).to(torch.device(self.device))
        for i in range(char_target.shape[0]):
            label_encode[i][char_target[i]] = 1

        label_sim = torch.matmul(label_encode, label_encode.T)

        return self.mul_pos_loss(sim_f, label_sim)


loss1 = ContrastLoss3(device='cpu')
sim_loss = loss1.compute_loss(input_n, target_n)
print(sim_loss)









#
#
# #特征向量a
# a = torch.rand(4, 512)
#
# ##特征向量b
# b = torch.rand(6, 512)
#
# # ##特征向量进行归一化
# # a, b = normalize(a), normalize(b)
#
# ##矩阵乘法求余弦相似度
# cos = torch.cdist(a, a)
# print(cos)
