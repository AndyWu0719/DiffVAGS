import numpy as np
from plyfile import PlyData, PlyElement
import torch
import os
import matplotlib.pyplot as plt

def convert(data, path):
    """增强版转换函数，包含详细诊断"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    print(f"\n🔍 详细数据分析 - {os.path.basename(path)}")
    print(f"="*50)
    print(f"输入数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"总体范围: [{data.min():.6f}, {data.max():.6f}]")
    
    # 提取各个组件
    xyz = data[:, :3]
    normals = np.zeros_like(xyz)  
    f_dc = data[:, 3:6]           # DC颜色分量 (最重要!)
    f_rest = data[:, 6:51]        # 球谐函数其他系数
    opacities = data[:, 51:52]    # 透明度 (原始logit值)
    scale = data[:, 52:55]        # 缩放 (原始log值)
    rotation = data[:, 55:59]     # 旋转四元数
    
    print(f"\n📊 各组件详细分析:")
    print(f"位置 (XYZ):")
    print(f"  形状: {xyz.shape}")
    print(f"  X范围: [{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}], 均值: {xyz[:,0].mean():.3f}")
    print(f"  Y范围: [{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}], 均值: {xyz[:,1].mean():.3f}")
    print(f"  Z范围: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}], 均值: {xyz[:,2].mean():.3f}")
    
    print(f"\n🎨 颜色DC分量 (关键!):")
    print(f"  形状: {f_dc.shape}")
    print(f"  R(DC0)范围: [{f_dc[:,0].min():.6f}, {f_dc[:,0].max():.6f}], 均值: {f_dc[:,0].mean():.6f}")
    print(f"  G(DC1)范围: [{f_dc[:,1].min():.6f}, {f_dc[:,1].max():.6f}], 均值: {f_dc[:,1].mean():.6f}")
    print(f"  B(DC2)范围: [{f_dc[:,2].min():.6f}, {f_dc[:,2].max():.6f}], 均值: {f_dc[:,2].mean():.6f}")
    print(f"  绝对值最大: {np.abs(f_dc).max():.6f}")
    
    print(f"\n💡 透明度 (Opacity Logits):")
    print(f"  形状: {opacities.shape}")
    print(f"  原始范围: [{opacities.min():.3f}, {opacities.max():.3f}], 均值: {opacities.mean():.3f}")
    # 应用sigmoid激活
    opacity_activated = 1.0 / (1.0 + np.exp(-opacities.flatten()))
    print(f"  激活后范围: [{opacity_activated.min():.3f}, {opacity_activated.max():.3f}], 均值: {opacity_activated.mean():.3f}")
    
    print(f"\n📏 缩放 (Scale Logs):")
    print(f"  形状: {scale.shape}")
    print(f"  原始范围: [{scale.min():.3f}, {scale.max():.3f}], 均值: {scale.mean():.3f}")
    # 应用指数激活
    scale_activated = np.exp(scale)
    print(f"  激活后范围: [{scale_activated.min():.6f}, {scale_activated.max():.6f}], 均值: {scale_activated.mean():.6f}")
    
    print(f"\n🔄 旋转四元数:")
    print(f"  形状: {rotation.shape}")
    print(f"  范围: [{rotation.min():.3f}, {rotation.max():.3f}]")
    rotation_norms = np.linalg.norm(rotation, axis=1)
    print(f"  四元数长度: [{rotation_norms.min():.3f}, {rotation_norms.max():.3f}], 均值: {rotation_norms.mean():.3f}")
    
    print(f"\n🚨 潜在问题检测:")
    issues = []
    
    # 检查颜色DC分量 - 这是最常见的问题!
    if np.abs(f_dc).max() < 1e-6:
        issues.append("❌ 致命问题: 颜色DC分量几乎为0 - 这会导致完全黑色!")
    elif np.abs(f_dc).max() < 0.01:
        issues.append("⚠️  警告: 颜色DC分量很小 - 可能导致颜色很暗")
    
    # 检查透明度
    if opacity_activated.mean() < 0.01:
        issues.append("❌ 致命问题: 透明度过低 - 高斯点几乎不可见!")
    elif opacity_activated.mean() < 0.1:
        issues.append("⚠️  警告: 透明度较低 - 可能影响可见性")
    
    # 检查缩放
    if scale_activated.max() < 1e-8:
        issues.append("❌ 致命问题: 缩放过小 - 高斯点可能不可见!")
    elif scale_activated.max() < 1e-4:
        issues.append("⚠️  警告: 缩放较小 - 可能影响渲染")
    
    # 检查位置
    if np.abs(xyz).max() > 50:
        issues.append("⚠️  警告: 位置坐标很大 - 可能超出查看器默认视野")
    
    # 检查旋转四元数
    if np.abs(rotation_norms - 1.0).max() > 0.1:
        issues.append("⚠️  警告: 四元数未正确归一化")
    
    if issues:
        print(f"\n🚨 发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✅ 数据检查通过，没有发现明显问题")
    
    # 🔧 自动修复严重问题
    fixed = False
    
    # 修复颜色DC分量为0的问题
    if np.abs(f_dc).max() < 1e-6:
        print(f"\n🔧 自动修复: 设置合理的颜色DC值...")
        # 3DGS中的DC分量对应球谐函数的第0阶系数
        # 对于RGB，合理的值范围通常在[-1, 1]
        f_dc = np.random.uniform(-0.5, 0.5, f_dc.shape).astype(np.float32)
        print(f"  修复后颜色范围: [{f_dc.min():.3f}, {f_dc.max():.3f}]")
        fixed = True
    
    # 修复透明度过低的问题
    if opacity_activated.mean() < 0.01:
        print(f"\n🔧 自动修复: 提高透明度...")
        # 设置透明度为可见的值 (logit space)
        opacities = np.ones_like(opacities) * 2.0  # sigmoid(2.0) ≈ 0.88
        opacity_activated = 1.0 / (1.0 + np.exp(-opacities.flatten()))
        print(f"  修复后透明度均值: {opacity_activated.mean():.3f}")
        fixed = True
    
    # 修复缩放过小的问题
    if scale_activated.max() < 1e-8:
        print(f"\n🔧 自动修复: 设置合理的缩放值...")
        # 设置合理的缩放值 (log space)
        scale = np.ones_like(scale) * np.log(0.01)  # exp(log(0.01)) = 0.01
        scale_activated = np.exp(scale)
        print(f"  修复后缩放范围: [{scale_activated.min():.6f}, {scale_activated.max():.6f}]")
        fixed = True
    
    # 归一化旋转四元数
    if np.abs(rotation_norms - 1.0).max() > 0.1:
        print(f"\n🔧 自动修复: 归一化旋转四元数...")
        rotation = rotation / (rotation_norms[:, np.newaxis] + 1e-8)
        rotation_norms = np.linalg.norm(rotation, axis=1)
        print(f"  修复后四元数长度范围: [{rotation_norms.min():.6f}, {rotation_norms.max():.6f}]")
        fixed = True
    
    if fixed:
        print(f"\n✅ 自动修复完成!")
    
    # 构建PLY文件
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

    write_path = path
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(write_path)
    
    file_size_mb = os.path.getsize(write_path) / (1024 * 1024)
    print(f"\n💾 文件保存完成:")
    print(f"  路径: {write_path}")
    print(f"  点数量: {len(elements):,}")
    print(f"  文件大小: {file_size_mb:.2f} MB")
    
    # 生成可视化
    try:
        create_debug_visualization(xyz, f_dc, opacity_activated, scale_activated, path)
    except Exception as e:
        print(f"⚠️  可视化生成失败: {e}")

def create_debug_visualization(xyz, f_dc, opacity, scale, save_path):
    """创建调试可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 3D散点图 (XY投影)
    ax = axes[0, 0]
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('点云XY投影 (颜色=Z)')
    plt.colorbar(scatter, ax=ax)
    
    # 颜色DC分量可视化
    ax = axes[0, 1]
    # 将DC分量转换为近似的RGB颜色进行显示
    rgb_approx = f_dc * 0.28209479177387814 + 0.5  # 球谐函数转换
    rgb_approx = np.clip(rgb_approx, 0, 1)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb_approx, s=1, alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('颜色可视化 (近似RGB)')
    
    # 透明度分布
    ax = axes[0, 2]
    ax.hist(opacity, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Opacity (激活后)')
    ax.set_ylabel('Count')
    ax.set_title(f'透明度分布 (均值: {opacity.mean():.3f})')
    ax.grid(True, alpha=0.3)
    
    # RGB分量分布
    ax = axes[1, 0]
    ax.hist(f_dc[:, 0], bins=50, alpha=0.6, color='red', label='R (DC0)', edgecolor='black')
    ax.hist(f_dc[:, 1], bins=50, alpha=0.6, color='green', label='G (DC1)', edgecolor='black')
    ax.hist(f_dc[:, 2], bins=50, alpha=0.6, color='blue', label='B (DC2)', edgecolor='black')
    ax.set_xlabel('DC Component Value')
    ax.set_ylabel('Count')
    ax.set_title('颜色DC分量分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 缩放分布
    ax = axes[1, 1]
    ax.hist(scale.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Scale (激活后)')
    ax.set_ylabel('Count')
    ax.set_title(f'缩放分布 (均值: {scale.mean():.6f})')
    ax.set_yscale('log')  # 使用对数刻度
    ax.grid(True, alpha=0.3)
    
    # 位置分布
    ax = axes[1, 2]
    ax.hist(xyz[:, 0], bins=50, alpha=0.5, color='red', label='X')
    ax.hist(xyz[:, 1], bins=50, alpha=0.5, color='green', label='Y')
    ax.hist(xyz[:, 2], bins=50, alpha=0.5, color='blue', label='Z')
    ax.set_xlabel('Position')
    ax.set_ylabel('Count')
    ax.set_title('位置分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    debug_path = save_path.replace('.ply', '_debug_analysis.png')
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🎨 调试可视化保存: {debug_path}")