import argparse  # 可以移除
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# --- 项目内导入 ---
from model.IRGen import IRGenText2Image  # <--- 导入修改后的 IRGen
from model.DictTree import TreeNode # 字典树从 pkl 加载
from dataset.dataset_flickr_caption_test import FlickrCaptionTestDataset  # <--- 导入新的测试 Dataset
# from dataset.config import config_gnd # 不再需要这个
from utils.evaluate import compute_map  # 假设这个函数能用
from utils.logger import get_logger
# from utils.utils import test_transform # 不再需要图像变换
# from utils.utils import train_transform
from model.CLIP.clip import clip, tokenize  # 需要 CLIP
import pickle
from tqdm import tqdm  # 加入 tqdm

# --- 主程序 ---
if __name__ == '__main__':

    # --- 1. 硬编码参数 ---
    # !! 修改为你自己的路径和参数 !!
    config = {
        "data_name": 'flickr_t2i',  # 与训练时一致
        "metadata_pkl_path": "/home/iiserver31/Workbench/likaipeng/IRgen/data/flick30k/flickr30k_text_to_image_retrieval_from_json.pkl", # <--- 元数据文件路径
        # !! 修改这里: 指向只包含测试图像的 RQ 文件 !!
        "rq_codes_pkl_path": "./output_rq/flickr30k_rq_codes_m4_k8_test_gallery.pkl", # <--- 测试集库 RQ 文件路径
        # !! 确认这里: 模型检查点路径 !! (应该是由在完整数据上训练的模型得到)
        "model_checkpoint_path": "./output_models_t2i/author_params_modified/best_model.pkl", # <--- 训练好的模型路径
        "clip_model_name": 'ViT-B/16',
        "beam_size": 30,  # Beam search 的宽度 (k 值)
        "ks": [1, 5, 10],  # 评估 Recall@k 和 mAP@k 的 k 值列表
        # --- 模型参数 (需要与训练时完全一致) ---
        "rq_m": 4,
        "rq_k_bits": 8,
        "decoder_depth": 8,
        # "embed_dim": 512, # 下面会自动确定
        # "num_heads": 8,   # 下面会自动确定
    }

    # --- 确定 embed_dim 和 num_heads (同训练脚本逻辑) ---
    if config["clip_model_name"] == 'ViT-B/16':
        config["embed_dim"] = 512
    elif config["clip_model_name"] == 'ViT-L/14':
        config["embed_dim"] = 768
    else:
        config["embed_dim"] = 512  # 默认

    if config["embed_dim"] % 8 == 0:
        config["num_heads"] = 8
    elif config["embed_dim"] % 12 == 0:
        config["num_heads"] = 12
    else:
        config["num_heads"] = 8  # 默认

    print("Using Configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")

    # --- 设置设备 ---
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 加载数据和模型 ---
    print("Loading test dataset (Query Captions)...")
    try:
        test_dataset = FlickrCaptionTestDataset(config["metadata_pkl_path"], max_caption_len=77)
        # 使用 batch_size=1 进行推理
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    except Exception as e:
        print(f"Error initializing Test Dataset/Loader: {e}")
        exit()

    print("Loading TEST GALLERY RQ codes, mapping, and tree...")
    try:
        # !! 加载测试集库的 RQ 文件 !!
        with open(config["rq_codes_pkl_path"], 'rb') as f:
            rq_data = pickle.load(f)
        num_classes = 2 ** config["rq_k_bits"]

        k_tree = rq_data['dict_tree']  # 测试集图像的字典树
        k_tree.clip_keys(max_key=num_classes - 1) # 使用 num_classes - 1 作为 max_key
        mapping = rq_data['mapping']  # 测试集图像的 RQ codes mapping (N_test, M)
        ids = torch.tensor(mapping).to(device)  # 将测试集 RQ codes 转为 Tensor
        id_length = config["rq_m"]

        # !! 获取从完整索引到测试库索引的映射 (如果生成了的话) !!
        full_idx_to_test_idx = rq_data.get('full_idx_to_test_idx')
        if full_idx_to_test_idx is None:
             print("Warning: 'full_idx_to_test_idx' not found in test gallery RQ file. Ground truth generation might fail.")
             # 可以选择退出或尝试其他方式构建 gnd

        num_db_images = ids.shape[0] # 这现在是测试集库的大小 (e.g., 1000)
        print(f"Test Gallery RQ data loaded. Num DB images: {num_db_images}, Code length: {id_length}")
    except FileNotFoundError:
         print(f"Error: Test Gallery RQ file not found at {config['rq_codes_pkl_path']}")
         exit()
    except Exception as e:
        print(f"Error loading Test Gallery RQ codes from {config['rq_codes_pkl_path']}: {e}")
        exit()

    print("Loading FULL metadata for ground truth lookup...")
    try:
        with open(config["metadata_pkl_path"], 'rb') as f:
            metadata = pickle.load(f)
        # imlist = metadata['imlist'] # 不再直接需要完整的 imlist 进行搜索，但映射关系需要
        q_caps = metadata['q_caps'] # 需要查询标题列表
        query_caption_idx_to_target_img_filename = metadata['query_caption_idx_to_target_img_filename']
        img_filename_to_imlist_idx = metadata['img_filename_to_imlist_idx'] # 需要从文件名映射到 *完整* 索引
        print(f"Full Metadata loaded. Num query captions: {len(q_caps)}")
    except FileNotFoundError:
         print(f"Error: Metadata file not found at {config['metadata_pkl_path']}")
         exit()
    except Exception as e:
        print(f"Error loading metadata from {config['metadata_pkl_path']}: {e}")
        exit()

    print("Loading CLIP model...")
    try:
        clip_model, _ = clip.load(config["clip_model_name"], device=device)
        clip_model.eval()
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        exit()

    print("Initializing and loading trained IRGen model...")
    try:
        # 使用与训练时相同的参数初始化模型
        model = IRGenText2Image(dec_depth=config["decoder_depth"],
                      num_classes=num_classes,
                      id_len=id_length,
                      embed_dim=config["embed_dim"],
                      num_heads=config["num_heads"])


        print(f"Loading state dict from: {config['model_checkpoint_path']}")

        state_dict = torch.load(config["model_checkpoint_path"], map_location='cpu')

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # print(f"Loaded model weights from epoch {checkpoint.get('epoch', 'N/A')}") # state_dict 里没有 epoch 信息
        print(f"Successfully loaded model state_dict.")

        model.eval()
        model = model.to(device)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {config['model_checkpoint_path']}")
        exit()
    except Exception as e:
        print(f"Error loading trained model from {config['model_checkpoint_path']}: {e}")
        exit()

    # --- 准备 Ground Truth for compute_map (适配测试集库) ---
    print("Preparing ground truth (gnd) for evaluation using TEST GALLERY indices...")
    # gnd 应该是一个列表，长度等于查询数量 (len(q_caps))
    # 每个元素 gnd[i] 是一个列表，包含第 i 个查询对应的正确图像在 *测试集库* 中的索引 (0 to N_test-1)
    gnd = []
    query_order_indices = [] # 记录 test_loader 输出的 query_caption_index 的顺序
    query_captions_map = {cap['caption_index']: cap for cap in q_caps} # 方便通过 index 查找

    # --- 确保 full_idx_to_test_idx 可用 ---
    if full_idx_to_test_idx is None:
        print("Error: Cannot proceed without 'full_idx_to_test_idx' mapping for ground truth generation.")
        exit()

    processed_queries = 0
    for _, query_caption_index_tensor in test_loader:
        query_caption_index = query_caption_index_tensor.item()
        query_order_indices.append(query_caption_index) # 记录顺序

        # 1. 通过查询标题索引找到对应的目标图像文件名
        target_img_filename = query_caption_idx_to_target_img_filename.get(query_caption_index)

        if target_img_filename:
            # 2. 通过文件名找到目标图像在 *完整 imlist* 中的索引
            target_full_imlist_index = img_filename_to_imlist_idx.get(target_img_filename)

            if target_full_imlist_index is not None:
                # 3. 使用 full_idx_to_test_idx 将完整索引映射到测试集库索引
                target_test_gallery_index = full_idx_to_test_idx.get(target_full_imlist_index)

                if target_test_gallery_index is not None:
                    # 找到了对应的测试库索引，这就是 ground truth
                    gnd.append([target_test_gallery_index])
                    processed_queries += 1
                else:
                    # 目标图像（来自完整索引）不在测试库索引映射中，这理论上不应发生，因为测试库就是从测试图像构建的
                    print(f"Warning: Target image {target_img_filename} (full index {target_full_imlist_index}) not found in test gallery mapping for query caption index {query_caption_index}.")
                    gnd.append([]) # 标记为无有效 GT
            else:
                 # 目标图像文件名不在完整 imlist 映射中
                 print(f"Warning: Target image {target_img_filename} (from query caption index {query_caption_index}) not found in img_filename_to_imlist_idx.")
                 gnd.append([]) # 标记为无有效 GT
        else:
            # 无法通过查询标题索引找到目标文件名
            print(f"Warning: Could not find target image filename for query caption index {query_caption_index}.")
            gnd.append([]) # 标记为无有效 GT

    print(f"Ground truth list generated with {len(gnd)} entries.")
    if processed_queries != len(test_loader):
         print(f"Warning: Processed {processed_queries} ground truth entries, but expected {len(test_loader)}. Some queries might lack valid ground truth in the test gallery.")


    # --- 执行测试循环 ---
    logger = get_logger('test_t2i_standard.log') # 使用新的日志文件名
    print("Starting inference using TEST GALLERY...")
    all_ranks = [] # 存储每个查询的排名结果 (测试库索引)

    with torch.no_grad():
        # 重新迭代 test_loader 以获取与 gnd 顺序对应的查询
        # 注意：这里假设 test_loader 每次迭代顺序固定 (shuffle=False)
        for caption_tokens, query_caption_index_tensor in tqdm(test_loader, desc="Testing"):
            caption_tokens = caption_tokens.to(device)
            query_idx_item = query_caption_index_tensor.item()

            # 获取文本特征
            text_features = clip_model.encode_text(caption_tokens).float()

            # --- 调用 beam_search 或 rerank (在测试集库上) ---
            # ids 和 k_tree 现在只包含测试集的数据
            try:
                # beam_search 返回的是测试集库中的 top-k 索引 (0 to N_test-1)
                out_indices = model.beam_search(text_feat=text_features,
                                                k=config["beam_size"],
                                                k_tree=k_tree, # 测试集树
                                                ids=ids)      # 测试集 mapping
                if isinstance(out_indices, np.ndarray):
                    out_indices = out_indices.flatten().tolist()
                elif isinstance(out_indices, torch.Tensor):
                    out_indices = out_indices.flatten().cpu().tolist()

                # 截取评估所需的 top-k
                rank_list = out_indices[:max(config["ks"])]
                all_ranks.append(rank_list)
                # 日志记录的是测试库索引
                # logger.info('QueryCapIdx:{} \t Top-{} TestDB Indices:{}'.format(query_idx_item, config["beam_size"], rank_list)) # 可选记录详细日志

            except AttributeError:
                print("Error: model.beam_search not found or not adapted. Trying rerank...")
                try:
                    # rerank 返回测试集库索引的排序 (0 to N_test-1)
                    ranked_db_indices = model.rerank(text_feat=text_features, ids=ids) # ids 是测试集 mapping
                    top_k_indices = ranked_db_indices[0, :max(config["ks"])]
                    rank_list = top_k_indices.tolist()
                    all_ranks.append(rank_list)
                    # logger.info('QueryCapIdx:{} \t Top-{} TestDB Indices (Rerank):{}'.format(query_idx_item, max(config["ks"]), rank_list)) # 可选记录详细日志
                except Exception as e_rerank:
                    print(f"Error during rerank for query {query_idx_item}: {e_rerank}")
                    all_ranks.append([])
                    # logger.error(f"Rerank failed for query {query_idx_item}")

            except Exception as e_beam:
                print(f"Error during beam_search for query {query_idx_item}: {e_beam}")
                all_ranks.append([])
                # logger.error(f"Beam search failed for query {query_idx_item}")

    # --- 评估结果 ---
    print("Inference finished. Evaluating results...")
    if not all_ranks or len(all_ranks) != len(gnd):
        print(f"Error: Mismatch between number of ranking results ({len(all_ranks)}) and ground truth entries ({len(gnd)}).")
        # 可能需要检查测试循环是否有提前退出的情况
        exit()

    max_k = max(config["ks"])
    num_queries_total = len(gnd) # 总查询数
    ranks_array = np.full((max_k, num_queries_total), -1, dtype=int) # 先按总数初始化

    valid_query_indices_for_eval = [] # 存储有效查询的原始索引 (在 all_ranks 和 gnd 中的位置)
    final_gnd = []                    # 只包含有效查询的 gnd 列表

    for i in range(num_queries_total):
        # 检查 gnd[i] 是否非空 (即找到了有效的测试库 GT 索引) 并且 all_ranks[i] 也非空 (即检索成功)
        if gnd[i] and all_ranks[i]:
            rank_list = all_ranks[i]
            padded_rank_list = (rank_list + [-1] * max_k)[:max_k]
            ranks_array[:, len(final_gnd)] = padded_rank_list # 填充到有效查询的位置
            final_gnd.append(gnd[i])
            valid_query_indices_for_eval.append(i) # 记录这个查询有效
        # else:
        #     query_idx_item = query_order_indices[i] # 获取对应的原始 caption index
        #     print(f"Skipping query {i} (Caption Index: {query_idx_item}) due to missing GND or failed retrieval.")

    num_valid_queries = len(final_gnd)
    if num_valid_queries == 0:
        print("Error: No valid queries with ground truth found for evaluation.")
        exit()

    # 只取有效查询的 ranks 列
    ranks_final = ranks_array[:, :num_valid_queries]

    print(f"Evaluating on {num_valid_queries} valid queries (out of {num_queries_total} total).")
    print(f"Ranks array shape for compute_map: {ranks_final.shape}")
    print(f"Ground truth list length for compute_map: {len(final_gnd)}")


    # --- 调用评估函数 ---
    try:
        kappas = config["ks"] if isinstance(config["ks"], list) else [config["ks"]]
        # !! ranks_final 的值应该是测试库的索引 (0 to N_test-1) !!
        # !! final_gnd 的值也应该是测试库的索引 (0 to N_test-1) !!
        map_val, aps, mpr, prs = compute_map(ranks_final, final_gnd, kappas=kappas)

        print(f"Evaluation Results (k={kappas}):")
        print(f"  mAP: {map_val:.4f}")
        # 假设 mpr 是 Recall@k
        print(f"  Mean Recall@k: {mpr}")

        logger.info(f"Evaluation on {num_valid_queries} queries.")
        logger.info(f"mAP: {map_val:.4f}")
        logger.info(f"Mean Recall@{kappas}: {mpr}")

    except NameError:
        print("Error: 'compute_map' function not found. Make sure it's imported correctly from utils.evaluate.")
    except Exception as e_eval:
        print(f"Error during evaluation with compute_map: {e_eval}")
        print("Please check the format expected by compute_map for 'ranks' and 'gnd'.")
    # --- [新增] 手动检查部分 ---
    print("-" * 30)
    print("Manual Check of Sample Queries:")
    print("-" * 30)

    num_samples_to_show = 10  # 你想看多少个样本
    if num_valid_queries >= num_samples_to_show:
        # 从有效查询的索引中随机抽取
        # valid_query_indices_for_eval 包含了 all_ranks/final_gnd 中有效条目的原始位置(0 到 N_query-1)
        sample_indices_in_valid_list = np.random.choice(
            num_valid_queries, num_samples_to_show, replace=False
        )

        # 需要重新加载元数据或确保 query_captions_map 可用
        try:
            with open(config["metadata_pkl_path"], 'rb') as f:
                metadata_check = pickle.load(f)
            q_caps_check = metadata_check['q_caps']
            query_captions_map_check = {cap['caption_index']: cap for cap in q_caps_check}
            # 如果测试库的 full_idx_to_test_idx 和 test_idx_to_full_idx 可用，也加载它们
            with open(config["rq_codes_pkl_path"], 'rb') as f:
                rq_data_check = pickle.load(f)
            # 注意：如果你在 rq.py 中没有保存 test_idx_to_full_idx，这里会报错或需要修改
            test_idx_to_full_idx_check = rq_data_check.get('test_idx_to_full_idx')

        except Exception as e:
            print(f"Error loading data for manual check: {e}")
            query_captions_map_check = {}  # 防止下面代码出错
            test_idx_to_full_idx_check = None

        for i, sample_idx in enumerate(sample_indices_in_valid_list):
            # sample_idx 是在 valid_query_indices_for_eval / final_gnd / ranks_final 中的索引 (0 to num_valid_queries-1)
            # original_query_pos 是该查询在原始 test_loader / gnd / all_ranks 列表中的位置
            original_query_pos = valid_query_indices_for_eval[sample_idx]
            original_caption_index = query_order_indices[original_query_pos]  # 获取原始标题ID

            caption_text = "Caption not found"
            if query_captions_map_check:
                caption_info = query_captions_map_check.get(original_caption_index)
                if caption_info:
                    caption_text = caption_info['caption']

            # ground_truth_list 是包含一个或多个正确 测试库索引 的列表
            ground_truth_list = final_gnd[sample_idx]
            # predicted_ranks 是预测的 Top-K 测试库索引 列表
            predicted_ranks = ranks_final[:, sample_idx].tolist()  # 取出对应列并转为 list

            print(
                f"\n--- Sample {i + 1} (Original Query Pos: {original_query_pos}, Caption Index: {original_caption_index}) ---")
            print(f"Caption: {caption_text}")
            print(f"Ground Truth (Test Gallery Index): {ground_truth_list}")
            print(f"Predicted Top-{max_k} (Test Gallery Indices): {predicted_ranks}")

            # 可选：将测试库索引映射回完整索引或文件名
            gt_full_indices = []
            if test_idx_to_full_idx_check and ground_truth_list:
                gt_full_indices = [test_idx_to_full_idx_check.get(gt_idx, "N/A") for gt_idx in ground_truth_list]
                print(f"Ground Truth (Full ImList Index): {gt_full_indices}")

            pred_full_indices = []
            if test_idx_to_full_idx_check:
                pred_full_indices = [test_idx_to_full_idx_check.get(pred_idx, "N/A") for pred_idx in predicted_ranks
                                     if pred_idx != -1]  # 排除填充的-1
                print(f"Predicted Top-{max_k} (Full ImList Indices): {pred_full_indices}")


    else:
        print(f"Not enough valid queries ({num_valid_queries}) to show {num_samples_to_show} samples.")
    print("-" * 30)

# --- 在这之后才是 print("Testing finished.") ---


    print("Testing finished.")