import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib


def save_prediction_visualization(prediction, save_path, image_name=None):
    """
    将模型预测结果可视化并保存为图片。

    参数:
        prediction: 模型预测的输出 (通常是经过softmax的结果)
        save_path: 保存可视化图像的路径
        image_name: 原始图像名称 (用于标题)
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将预测转换为分割图 (BxCxHxWxD或CxHxWxD张量)
    if isinstance(prediction, np.ndarray):
        if prediction.ndim == 4:  # CxHxWxD
            segmentation = np.argmax(prediction, axis=0)
        else:  # 其他维度，可能需要额外处理
            segmentation = np.argmax(prediction, axis=0)
    else:  # torch tensor
        segmentation = prediction.argmax(dim=0).cpu().numpy()

    # 获取维度
    num_dims = len(segmentation.shape)

    # 对于3D体积数据，创建多切片视图
    if num_dims == 3:
        # 获取每个维度的中间切片
        h, w, d = segmentation.shape
        slice_h = segmentation[h // 2, :, :]
        slice_w = segmentation[:, w // 2, :]
        slice_d = segmentation[:, :, d // 2]

        # 创建颜色映射用于可视化
        # 调整颜色数量以匹配类别数
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white']
        max_class = max(segmentation.max() + 1, len(colors))
        cmap = ListedColormap(colors[:max_class])

        # 绘制切片
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(slice_h, cmap=cmap, vmin=0, vmax=max_class - 1)
        axes[0].set_title('轴状面 (Axial)')
        axes[0].axis('off')

        axes[1].imshow(slice_w, cmap=cmap, vmin=0, vmax=max_class - 1)
        axes[1].set_title('冠状面 (Coronal)')
        axes[1].axis('off')

        axes[2].imshow(slice_d, cmap=cmap, vmin=0, vmax=max_class - 1)
        axes[2].set_title('矢状面 (Sagittal)')
        axes[2].axis('off')

        if image_name:
            plt.suptitle(f'预测结果: {image_name}')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        # 尝试生成多个轴状面切片的可视化
        if h > 10:  # 确保有足够的切片
            multi_slice_dir = os.path.join(os.path.dirname(save_path), 'multi_slice')
            os.makedirs(multi_slice_dir, exist_ok=True)
            base_name = os.path.basename(save_path).split('.')[0]

            for slice_idx in range(0, h, h // 10):  # 每隔h/10选取一个切片
                plt.figure(figsize=(8, 8))
                plt.imshow(segmentation[slice_idx, :, :], cmap=cmap, vmin=0, vmax=max_class - 1)
                plt.title(f'轴状面 - 切片 {slice_idx}/{h}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(multi_slice_dir, f'{base_name}_slice_{slice_idx}.png'))
                plt.close()

    # 对于2D图像
    elif num_dims == 2:
        plt.figure(figsize=(8, 8))

        # 创建颜色映射用于可视化
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white']
        max_class = max(segmentation.max() + 1, len(colors))
        cmap = ListedColormap(colors[:max_class])

        plt.imshow(segmentation, cmap=cmap, vmin=0, vmax=max_class - 1)
        plt.axis('off')
        if image_name:
            plt.title(f'预测结果: {image_name}')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # 可以选择额外保存为NIfTI格式，便于医学影像查看软件使用
    if save_path.endswith('.png'):
        nii_save_path = save_path.replace('.png', '.nii.gz')
        try:
            # 创建NIfTI对象 (使用单位仿射矩阵)
            if num_dims == 3:
                nii_img = nib.Nifti1Image(segmentation.astype(np.int16), np.eye(4))
                nib.save(nii_img, nii_save_path)
        except Exception as e:
            print(f"保存NIfTI文件时出错: {e}")


def save_comparison_visualization(prediction, ground_truth, save_path, image_name=None):
    """
    将模型预测结果和真实标签进行对比可视化并保存为图片。

    参数:
        prediction: 模型预测结果
        ground_truth: 真实标签
        save_path: 保存路径
        image_name: 图像名称
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将预测和标签转换为分割图
    if isinstance(prediction, np.ndarray):
        if prediction.ndim == 4:  # CxHxWxD
            segmentation = np.argmax(prediction, axis=0)
        else:
            segmentation = np.argmax(prediction, axis=0)
    else:  # torch tensor
        segmentation = prediction.argmax(dim=0).cpu().numpy()

    # 确保ground truth是numpy数组
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.cpu().numpy()

    # 处理ground truth可能的维度问题
    if ground_truth.ndim == 4:
        ground_truth = ground_truth[0]  # 获取第一个batch的标签

    # 获取维度
    num_dims = len(segmentation.shape)

    # 对于3D体积数据
    if num_dims == 3:
        # 获取每个维度的中间切片
        h, w, d = segmentation.shape

        # 预测结果切片
        pred_slice_h = segmentation[h // 2, :, :]
        pred_slice_w = segmentation[:, w // 2, :]
        pred_slice_d = segmentation[:, :, d // 2]

        # 真实标签切片
        gt_slice_h = ground_truth[h // 2, :, :]
        gt_slice_w = ground_truth[:, w // 2, :]
        gt_slice_d = ground_truth[:, :, d // 2]

        # 创建颜色映射
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white']
        max_class = max(max(segmentation.max(), ground_truth.max()) + 1, len(colors))
        cmap = ListedColormap(colors[:max_class])

        # 创建2x3网格：上排是预测，下排是真实标签
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 预测结果行
        axes[0, 0].imshow(pred_slice_h, cmap=cmap, vmin=0, vmax=max_class - 1)
        axes[0, 0].set_title('预测 - 轴状面')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(pred_slice_w, cmap=cmap, vmin=0, vmax=max_class - 1)
        axes[0, 1].set_title('预测 - 冠状面')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(pred_slice_d, cmap=cmap, vmin=0, vmax=max_class - 1)
        axes[0, 2].set_title('预测 - 矢状面')
        axes[0, 2].axis('off')

        # 真实标签行
        axes[1, 0].imshow(gt_slice_h, cmap=cmap, vmin=0, vmax=max_class - 1)
        axes[1, 0].set_title('真实标签 - 轴状面')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(gt_slice_w, cmap=cmap, vmin=0, vmax=max_class - 1)
        axes[1, 1].set_title('真实标签 - 冠状面')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(gt_slice_d, cmap=cmap, vmin=0, vmax=max_class - 1)
        axes[1, 2].set_title('真实标签 - 矢状面')
        axes[1, 2].axis('off')

        if image_name:
            plt.suptitle(f'{image_name} - 预测 vs 真实标签')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        # 也保存差异图（预测与真实标签的不同之处）
        diff_dir = os.path.join(os.path.dirname(save_path), 'difference')
        os.makedirs(diff_dir, exist_ok=True)
        diff_path = os.path.join(diff_dir, os.path.basename(save_path))

        # 计算三个平面的差异
        diff_h = (pred_slice_h != gt_slice_h).astype(np.float32)
        diff_w = (pred_slice_w != gt_slice_w).astype(np.float32)
        diff_d = (pred_slice_d != gt_slice_d).astype(np.float32)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(diff_h, cmap='hot')
        axes[0].set_title('差异 - 轴状面')
        axes[0].axis('off')

        axes[1].imshow(diff_w, cmap='hot')
        axes[1].set_title('差异 - 冠状面')
        axes[1].axis('off')

        axes[2].imshow(diff_d, cmap='hot')
        axes[2].set_title('差异 - 矢状面')
        axes[2].axis('off')

        plt.suptitle(f'{image_name} - 预测与真实标签的差异')
        plt.tight_layout()
        plt.savefig(diff_path)
        plt.close(fig)