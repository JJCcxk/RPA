# # import cv2
# # import numpy as np
# # import os

# # def create_heatmap(image_path):
# #     """创建热力图（高亮度=深色，低亮度=浅色）"""
# #     # 读取图像并验证
# #     if not os.path.exists(image_path):
# #         raise FileNotFoundError(f"图像文件 {image_path} 不存在")
    
# #     img = cv2.imread(image_path)
# #     if img is None:
# #         raise ValueError("无法读取图像，请检查文件格式")
    
# #     # 转换为灰度图
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# #     # 归一化到0-255范围
# #     normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
# #     # 创建自定义颜色映射（浅色到深色）
# #     colormap = np.zeros((256, 1, 3), dtype=np.uint8)
# #     for i in range(256):
# #         # 创建从浅黄(255,255,100)到深红(100,0,0)的渐变
# #         colormap[i][0][0] = 100 + int(155 * (i/255))      # 蓝色通道（深色增强）
# #         colormap[i][0][1] = 255 - int(255 * (i/255))      # 绿色通道（渐减）
# #         colormap[i][0][2] = 255 - int(155 * (i/255))      # 红色通道（保持高亮）
    
# #     # 应用颜色映射
# #     heatmap = cv2.applyColorMap(normalized, colormap)
    
# #     # 显示结果
# #     cv2.imshow('Original', img)
# #     cv2.imshow('Heatmap', heatmap)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
    
# #     # 保存结果（可选）
# #     # cv2.imwrite('heatmap_result.jpg', heatmap)

# # if __name__ == "__main__":
# #     # 使用示例（替换为你的图片路径）
# #     image_path = r'C:\Users\27603\Desktop\test\r077507cat.png'
    
# #     try:
# #         create_heatmap(image_path)
# #     except Exception as e:
# #         print(f"错误发生：{str(e)}")
        
# #"C:\Users\27603\Desktop\test\r077507cat.png"



# # import cv2
# # import matplotlib.pyplot as plt
# # import numpy as np

# # # 读取彩色图像
# # img = cv2.imread('C:/Users/27603/Desktop/as/LOL-v2real/2.jpg')
# # if img is None:
# #     print("无法读取图像，请检查路径是否正确")
# # else:
# #     # 转换为RGB顺序（matplotlib使用RGB，而OpenCV使用BGR）
# #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# #     # 设置画布大小
# #     plt.figure(figsize=(10, 6))

# #     # 组合绘制RGB三个通道的直方图
# #     colors = ['r', 'g', 'b']
# #     channel_names = ['Red', 'Green', 'Blue']
    
# #     for i, (color, name) in enumerate(zip(colors, channel_names)):
# #         hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
# #         plt.plot(hist, color=color, label=name)
    
# #     plt.title('RGB')
# #     plt.xlabel('Pixel Value')
# #     plt.ylabel('Number of Pixels')
# #     plt.xlim([0, 256])
# #     plt.legend(loc='upper right')
# #     plt.grid(True, linestyle='--', alpha=0.7)
# #     plt.tight_layout()

# #     plt.show()    

# import matplotlib.pyplot as plt
# import numpy as np

# # 模型名称
# models = [  'URetinex', 'Restormer', 'Retinexformer', 'SNR-Net', 'Ours']
# # PSNR 值
# psnr_values = [  21.16, 19.94, 21.61, 21.48, 22.36]
# # SSIM 值
# ssim_values = [  84, 82.7, 83.2, 84.9, 84.2]

# # 创建画布和双Y轴
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()

# # 柱子宽度和位置
# bar_width = 0.35
# x = np.arange(len(models))

# # 绘制PSNR柱子（蓝色）在左侧Y轴
# bars1 = ax1.bar(x - bar_width/2, psnr_values, width=bar_width, color='#3b82f6', label='PSNR (dB)')
# ax1.bar_label(bars1, padding=3, fmt='%.2f')  # 添加数值标签

# # 绘制SSIM柱子（橙色）在右侧Y轴
# bars2 = ax2.bar(x + bar_width/2, ssim_values, width=bar_width, color='#f97316', label='SSIM (%)')
# ax2.bar_label(bars2, padding=3, fmt='%.1f')  # 添加数值标签

# # 设置坐标轴标签和标题
# ax1.set_xlabel('', fontsize=20)
# ax1.set_ylabel('PSNR (dB)', fontsize=20, color='#3b82f6')
# ax2.set_ylabel('SSIM (%)', fontsize=20, color='#f97316')
# plt.title('LOL-v2real', fontsize=20)

# # 设置x轴刻度和标签
# ax1.set_xticks(x)
# ax1.set_xticklabels(models, rotation=0, ha='center')

# # 设置Y轴范围（不强制从0开始）
# ax1.set_ylim(min(psnr_values) - 1, max(psnr_values) + 1)
# ax2.set_ylim(min(ssim_values) - 2, max(ssim_values) + 2)  # 调整SSIM轴范围，不强制从0开始

# # 添加图例
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# # 美化图表
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()

# # 显示图形
# plt.show()    

# #LOL-v1
# psnr_values = [  21.33, 22.43, 23.61, 24.61, 24.88]
# # SSIM 值
# ssim_values = [  83.5, 82.3, 83.4, 84.2, 83.8]

# # LOL-v2real
# # PSNR 值
# psnr_values = [  21.16, 19.94, 21.61, 21.48, 22.36]
# # SSIM 值
# ssim_values = [  84, 82.7, 83.2, 84.9, 84.2]

# # LOL-v2syn
# # PSNR 值
# psnr_values = [  23.36, 21.71, 24.07, 24.14, 25.31]
# # SSIM 值
# ssim_values = [  84.8, 83, 92.2, 92.8, 92.3]

# # SMID
# psnr_values = [  27.68, 26.97, 28.29, 28.49, 29.17]
# # SSIM 值
# ssim_values = [  78.7, 75.8, 80.3, 80.5, 81.5]

# # SID
# psnr_values = [  22.13, 22.27, 23.89, 22.84, 24.45]
# # SSIM 值
# ssim_values = [  64.2, 64.9, 66.9, 62.5, 67.7]



import matplotlib.pyplot as plt
import numpy as np

# LOL-v1 数据
methods = ['URetinex', 'Restormer', 'Retinexformer', 'SNR-Net', 'Ours']
psnr_values = [  23.36, 21.71, 24.07, 24.14, 25.31]
# SSIM 值
ssim_values = [  84.8, 83, 92.2, 92.8, 92.3]

# 配色与标记
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', 'red']
markers = ['o', 'o', 'o', 'o', '*']

# 创建图形
plt.figure(figsize=(6, 5), dpi=300)

for i in range(5):
    plt.scatter(psnr_values[i], ssim_values[i],
                c=colors[i],
                marker=markers[i],
                s=100 if i < 4 else 150,  # ours 用大一点的五角星
                edgecolors='black',
                linewidths=0.6,
                label=methods[i])

# 坐标轴设置：根据数据动态微调
plt.xlim(min(psnr_values) - 0.5, max(psnr_values) + 0.5)
plt.ylim(min(ssim_values) - 0.7, max(ssim_values) + 0.7)

# 标注
plt.xlabel('PSNR (dB)', fontsize=12)
plt.ylabel('SSIM (%)', fontsize=12)
plt.title('LOL-v2syn', fontsize=14)

# 网格 & 图例
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=9, loc='lower right')

plt.tight_layout()
#plt.savefig("lolv1_scatter.png", bbox_inches='tight')
plt.show()
