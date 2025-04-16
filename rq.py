import argparse
import faiss  # 确保 faiss 已安装 (pip install faiss-gpu or faiss-cpu)
import numpy as np
import os
import pickle
from model.DictTree import TreeNode  # 确保 DictTree.py 在你的项目中

# --- 参数解析 ---
parser = argparse.ArgumentParser(description='Residual Quantization for Image Features')
parser.add_argument('--data_name', default='flickr', type=str,  # 默认改为 'flickr'
                    help='Dataset name (e.g., flickr, cub, isc)')
parser.add_argument('--features', required=True, type=str,  # 改为必需参数
                    help='Path to the .npy file containing image features (N, D)')
parser.add_argument('--output_file', required=True, type=str,  # 改为必需参数，明确指定输出文件名
                    help='Output .pkl file name (e.g., flickr30k_rq_codes.pkl)')
parser.add_argument('--output_dir', default='./', type=str,  # 输出目录
                    help='Directory to save the output .pkl file')
parser.add_argument('--m', type=int, default=4,  # 将 m 作为参数，默认为 4 (可调整)
                    help='Number of residual quantization stages (code length)')
parser.add_argument('--k', type=int, default=8,  # 将 k 作为参数，默认为 8 (可调整)
                    help='Log2 of the number of centroids per quantizer (k=8 means 256 centroids)')

opt = parser.parse_args()

# --- 获取参数 ---
data_name = opt.data_name
features_path = opt.features
output_filename = opt.output_file
output_dir = opt.output_dir
m = opt.m  # 量化阶段数/代码长度
k_bits = opt.k  # 每个阶段的比特数 (码本大小 = 2^k_bits)
num_centroids = 2 ** k_bits  # 每个PQ量化器的中心点数量

print(f"Dataset: {data_name}")
print(f"Loading features from: {features_path}")
print(f"Output directory: {output_dir}")
print(f"Output file: {output_filename}")
print(f"RQ Parameters: m={m}, k={k_bits} ({num_centroids} centroids per stage)")

# --- 加载特征数据 ---
try:
    data = np.load(features_path).astype('float32')  # 确保是 float32
    print(f"Loaded features shape: {data.shape}")
except FileNotFoundError:
    print(f"Error: Features file not found at {features_path}")
    exit()

dim = data.shape[1]  # 特征维度
num_data = data.shape[0]  # 数据点数量

# --- 执行残差量化 (RQ) ---
print("Starting Residual Quantization...")
original_data = data.copy()  # 保留原始数据用于构建树（如果需要）或调试
rq_codes = np.zeros((num_data, m), dtype=np.uint8 if k_bits <= 8 else np.uint16)  # 初始化代码存储，类型取决于k
codebooks = []  # 存储每个阶段的码本

for i in range(m):
    print(f"--- RQ Stage {i + 1}/{m} ---")

    # 注意：Faiss ProductQuantizer 输入维度必须能被 M 整除
    # 这里我们假设 M=1 (每个量化器处理整个维度或其子维度)
    # 更常见的用法是 M > 1，将维度 d 划分为 M 个子维度 dsub = d / M
    # 如果你的特征维度很大，可以考虑使用 M > 1
    # 为简单起见，这里保持 M=1，每个量化器处理整个当前残差维度 d
    M_pq = 1  # Number of subspaces for PQ - keeping it simple at 1
    # 检查维度是否可以被 M_pq 整除 (当 M_pq=1 时总是可以)
    # if dim % M_pq != 0:
    #     print(f"Error: Feature dimension {dim} is not divisible by M_pq {M_pq}")
    #     exit()

    # nbits = k_bits # 每个子向量分配的比特数
    pq = faiss.ProductQuantizer(dim, M_pq, k_bits)

    print(f"Training PQ {i + 1} on current residuals (shape: {data.shape})...")
    # Faiss 期望训练数据是 C-contiguous float32
    pq.train(data)

    print(f"Computing codes for stage {i + 1}...")
    codes_i = pq.compute_codes(data)  # (num_data, M_pq)
    # print(f"Stage {i+1} codes shape: {codes_i.shape}")

    # 将当前阶段的代码存储到最终的 rq_codes 矩阵中
    # 因为 M_pq=1，codes_i 的形状是 (num_data, 1)，我们取第一列
    rq_codes[:, i] = codes_i[:, 0]

    print(f"Decoding data for stage {i + 1} to compute residuals...")
    data_reconstructed_i = pq.decode(codes_i)

    print(f"Storing codebook for stage {i + 1}...")
    # 获取并存储码本 (centroids)
    # .centroids 是一个扁平数组，大小为 (M * ksub * dsub)
    # 当 M=1 时, ksub = 2^k_bits, dsub = dim
    codebook_i = faiss.vector_to_array(pq.centroids).reshape(M_pq, num_centroids, dim)
    codebooks.append(codebook_i)

    print(f"Updating residuals for next stage...")
    data -= data_reconstructed_i  # 计算残差

    print(f"Residual norm after stage {i + 1}: {np.linalg.norm(data) / np.sqrt(num_data)}")

print("RQ encoding finished.")
print(f"Final RQ codes shape: {rq_codes.shape}")  # (num_data, m)

# 合并所有阶段的码本
# codebooks 是一个 list of arrays, shape [(1, 256, dim), (1, 256, dim), ...]
# 我们希望得到一个 (m, 256, dim) 的数组 (假设 M_pq=1)
combined_codebook = np.array(codebooks).squeeze(axis=1)  # Squeeze the M_pq=1 dimension
print(f"Combined codebook shape: {combined_codebook.shape}")  # (m, num_centroids, dim)

# --- 构建字典树 ---
print("Building dictionary tree (TreeNode)...")
# 移除原代码中针对 'isc' 的特殊处理，为所有数据构建树
kmeans_tree = TreeNode()
# rq_codes 的每一行是一个 m 维的整数代码 (uint8 或 uint16)
# TreeNode 的 insert 方法期望一个可迭代对象 (如 list 或 string)
# 我们需要将每行的 numpy 数组转为 list 或 tuple
codes_for_tree = [tuple(code) for code in rq_codes]
kmeans_tree.insert_many(codes_for_tree)
print(f"Dictionary tree built with {kmeans_tree.count} codes.")  # count 应该是 num_data

# --- 准备输出文件 ---
output_data = {
    'mapping': rq_codes,  # (N, m) 形状的 RQ 代码矩阵
    'codebook': combined_codebook,  # (m, 2^k, D) 形状的码本
    'dict_tree': kmeans_tree  # 用于快速搜索的字典树
}

# --- 保存到 .pkl 文件 ---
output_path = os.path.join(output_dir, output_filename)
print(f"Saving processed data to {output_path}...")
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(output_data, f)

print("Done.")