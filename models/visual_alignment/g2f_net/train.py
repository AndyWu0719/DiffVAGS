#!/usr/bin/env python3
import argparse
from g2fnet import train_model

def main():
    parser = argparse.ArgumentParser(description='训练多视角特征预测模型')
    parser.add_argument('--gaussian_path', type=str, required=False, help='高斯数据路径', default='/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/convert_data')
    parser.add_argument('--image_path', type=str, required=False, help='图像数据路径', default='/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/training_data')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    
    args = parser.parse_args()
    
    # 设置全局参数（在实际应用中应传递到训练函数）
    global GAUSSIAN_PATH, IMAGE_PATH
    GAUSSIAN_PATH = args.gaussian_path
    IMAGE_PATH = args.image_path
    
    print(f"开始训练: 高斯数据={args.gaussian_path}, 图像数据={args.image_path}")
    print(f"参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    
    train_model()

if __name__ == "__main__":
    main()

