import argparse  # 可以移除或注释掉 import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.dataset_flickr import FlickrImageDataset  # 确保导入正确
# from utils.utils import train_transform # 如果下面不用可以移除
import os
import pickle
from model.CLIP.clip import clip
import tqdm


# --- get_features 函数保持不变 ---
@torch.no_grad()
def get_features(Dataset, encoder_model, batch_size=32):
    """使用指定的编码器模型从数据集中提取特征"""
    loader = DataLoader(Dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_feats = []
    device = next(encoder_model.parameters()).device
    print(f"Extracting features using device: {device}")
    for i, (images, _, _) in enumerate(tqdm.tqdm(loader, desc="Extracting Features")):
        images = images.to(device)
        features = encoder_model(images)
        # features /= features.norm(dim=-1, keepdim=True) # 可选：归一化
        all_feats.append(features.cpu().numpy())

    if not all_feats:
        print("Warning: No features extracted.")
        return None
    feats_array = np.concatenate(all_feats, axis=0)
    print(f"Concatenated features shape: {feats_array.shape}")
    return feats_array.astype(np.float32)


# --- 主程序 ---
if __name__ == '__main__':



    # --- 2. 直接定义变量 (硬编码) ---

    data_name = 'flickr'
    flickr_pkl_path = "./data/flickr30k/flickr30k_text_to_image_retrieval_from_json.pkl"  # <--- 修改这里
    flickr_image_dir = "./data/flickr30k/flickr30k-images"  #
    output_npy_path = "flickr30k_imlist_clip_features.npy"
    clip_model_name = 'ViT-B/16'
    batch_size = 256

    # --- 数据集加载 ---
    Dataset = None
    if data_name == 'flickr':
        print("Loading Flickr30k Image Dataset...")
        # 使用上面定义的硬编码变量
        Dataset = FlickrImageDataset(flickr_pkl_path,
                                     flickr_image_dir,
                                     transform=None)  # 后面会用 clip_preprocess 覆盖 transform
    else:
        print(f"Error: Dataset '{data_name}' is not supported by this hardcoded script.")
        exit()

    if Dataset is None or len(Dataset) == 0:
        print("Error: Failed to load dataset or dataset is empty.")
        exit()

    print(f"Dataset object created with {len(Dataset)} potential images.")

    # --- 加载 CLIP 模型 ---
    print(f"Loading CLIP model: {clip_model_name}...")
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    # 加载模型和预处理器
    mm, clip_preprocess = clip.load(clip_model_name, device=device)
    visual_encoder = mm.visual.eval()

    # --- 重新设置/加载数据集，确保使用 CLIP 的预处理器 ---
    print("Setting dataset transform using CLIP preprocess...")
    if data_name == 'flickr':
        # 直接设置 Dataset 对象的 transform 属性
        Dataset._transform = clip_preprocess  # <--- 使用 CLIP 的预处理器
        # 或者重新初始化 Dataset:
        # Dataset = FlickrImageDataset(flickr_pkl_path, flickr_image_dir, transform=clip_preprocess)
    # ... (处理其他数据集，如果需要)

    # 再次检查 Dataset 是否有效
    if Dataset is None or len(Dataset) == 0:
        print("Error: Failed to assign transform or re-load dataset.")
        exit()
    print(f"Dataset ready with {len(Dataset)} images and CLIP preprocess.")

    # --- 提取特征 ---
    print("Extracting features...")
    start_time = time.time()
    extracted_features = get_features(Dataset, visual_encoder, batch_size=batch_size)
    end_time = time.time()
    print(f"Feature extraction took {end_time - start_time:.2f} seconds.")

    # --- 保存特征 ---
    if extracted_features is not None:
        print(f"Saving extracted features to {output_npy_path}...")
        # 确保输出目录存在
        output_dir = os.path.dirname(output_npy_path)
        if output_dir:  # 只有在路径包含目录时才创建
            os.makedirs(output_dir, exist_ok=True)
        np.save(output_npy_path, extracted_features)
        print("Features saved successfully.")
    else:
        print("Feature extraction failed, nothing to save.")

    print("Script finished.")