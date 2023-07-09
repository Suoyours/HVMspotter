import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
# Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r,
# GnBu(绿到蓝), GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd(橘色到红色), OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired,
# Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd,
# PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds,
# Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia(蓝绿黄), Wistia_r, YlGn,YlGnBu,
# YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r,YlOrRd(红橙黄), YlOrRd_r, afmhot, afmhot_r,autumn, autumn_r, binary, binary_r,
# bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm(蓝到红), coolwarm_r, copper(铜色),
# copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat,
# gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg,
# gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r(红黄), hsv, hsv_r, icefire,
# icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean,
# ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r,rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r,
# spring, spring_r, summer (黄到绿), summer_r(绿到黄), tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r,
# terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r,viridis,viridis_r, vlag, vlag_r,
# winter, winter_r

data_path = '/data/ys/PycharmProjects2/pan_pp.pytorch/outputs/three_maps_pic'
save_path = '/data/ys/PycharmProjects2/pan_pp.pytorch/outputs/ployheatmap/two/'
img_name_list = os.listdir(data_path)

img = '2022-02-1022.npz'
img_data_path = data_path + '/' + img
img_save_path = save_path + img.split('.')[0] + '.jpg'
map1 = np.load(img_data_path)
map1 = cv2.resize(map1['arr_0'], (0, 0), fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
for i in range(map1.shape[0]):
    for j in range(map1.shape[1]):
        if map1[i][j] > 1:
            map1[i][j] = 1
        elif map1[i][j] < 0.4:
            map1[i][j] = map1[i][j]/ 1.5
# map1 = map1*2
# values = np.random.rand(3, 3)
x_ticks = ['x']
y_ticks = ['y']  # 自定义横纵轴
ax = sns.heatmap(map1, xticklabels=x_ticks, yticklabels=y_ticks, cmap='jet')
# ax = sns.heatmap(data=map1,linewidths=0.3,cmap="RdBu_r")
# ax.set_title('Heatmap for test')  # 图标题
# ax.set_xlabel('x label')  # x轴标题
# ax.set_ylabel('y label')

# plt.show()
figure = ax.get_figure()
figure.savefig(img_save_path)  # 保存图片
plt.close()
print(img_save_path)
# heatmap0 = np.uint8(255-255 * map1)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
# heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
# # superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
# plt.imshow(heatmap)  # ,cmap='gray' ，这里展示下可视化的像素值
# # plt.imshow(superimposed_img)  # ,cmap='gray'
# plt.show()