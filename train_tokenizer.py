# File: train_flickr_tokenizer.py

# --- Standard Imports ---
import argparse
import datetime
import json
import os
import pickle
import random
import time
from pathlib import Path

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader

# --- Third-party Imports ---
import numpy as np
from PIL import Image # Needed if Dataset class uses it implicitly
import tqdm

# --- Local Imports (Ensure these paths are correct) ---
try:
    # Assumes dataset definition is in dataset/dataset_flickr_instance.py
    from dataset.dataset_flickr_instance import FlickrInstanceDataset
except ImportError:
    print("错误：无法导入 FlickrInstanceDataset。")
    print("请确保 dataset/dataset_flickr_instance.py 文件存在且包含 FlickrInstanceDataset 类。")
    exit()
try:
    # Assumes model definition is in model/tokenizer.py
    from model.tokenizer import vitrqfc
except ImportError:
    print("错误：无法导入 'vitrqfc' 模型。请确保 model/tokenizer.py 文件存在且包含 vitrqfc 类。")
    exit()
try:
    # Assumes utility functions are in utils/utils.py
    from utils.utils import get_trainable_params, residualquantizer
except ImportError:
    print("错误：无法导入 utils.utils 中的函数。请确保 utils/utils.py 文件存在。")
    exit()
try:
    # Assumes logger is in utils/logger.py
    from utils.logger import get_logger
except ImportError:
    print("错误：无法导入 utils.logger 中的 get_logger。请确保 utils/logger.py 文件存在。")
    exit()
try:
    # Assumes CLIP model is accessible via model.CLIP.clip
    from model.CLIP.clip import clip
except ImportError:
     print("错误: 无法导入 CLIP 模型。请确保 CLIP 安装正确或路径设置正确。")
     exit()


# --- Helper Functions ---

@torch.no_grad()
def get_features(dataset_instance, encoder_model, batch_size=32, device="cuda", num_workers=4):
    """Extracts features using the specified encoder model."""
    encoder_model.eval() # Set to evaluation mode
    loader = DataLoader(dataset_instance, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    all_feats = []
    print(f"Extracting features using device: {device}")
    for i, batch_data in enumerate(tqdm.tqdm(loader, desc="Extracting Features")):
        # Assuming dataset returns (image, class_label, index)
        images = batch_data[0].to(device)
        try:
            features = encoder_model(images)
            if isinstance(features, tuple):
                 features = features[0]
            # Use class token (assuming ViT architecture)
            features = features[:, 0, :]
            all_feats.append(features.cpu().numpy())
        except Exception as e:
             print(f"\n错误: 特征提取在 batch {i} 失败: {e}")
             # Optionally skip batch or handle error
             # raise e # Re-raise if critical

    encoder_model.train() # Set back to training mode

    if not all_feats:
        print("警告: 未提取到任何特征。")
        return None

    feats_array = np.concatenate(all_feats, axis=0)
    if feats_array.shape[0] != len(dataset_instance):
         print(f"\n警告: 提取的特征数量 ({feats_array.shape[0]}) 与数据集大小 ({len(dataset_instance)}) 不匹配！\n")
         # This might happen if errors occurred during extraction for some batches.
         # Depending on severity, you might want to stop or continue with partial features.
         # Returning None for safety for now.
         return None

    print(f"提取到的特征形状: {feats_array.shape}")
    return feats_array.astype(np.float32)

# --- Main Execution Block ---
if __name__ == '__main__':

    # === 1. Configuration ===
    # --- Dataset ---
    DATA_NAME = 'flickr'
    # *** Modify paths as needed ***
    FLICKR_GND_PKL_PATH = "./data/flick30k/flickr30k_instance_retrieval_with_gnd_v2.pkl"
    FLICKR_IMAGE_DIR = "/home/iiserver31/Workbench/likaipeng/IRgen/data/flick30k/flickr30k-images"
    OUTPUT_DIR = './output_flickr_tokenizer_trained/'

    # --- Model ---
    CLIP_MODEL_NAME = 'ViT-B/16'
    TOKENIZER_DEC_DEPTH = 12 # Or your desired depth

    # --- Training Hyperparameters ---
    NUM_EPOCHS = 200
    BATCH_SIZE = 128
    LR_FC = 5e-4
    LR_ENCODER_RATIO = 0.01
    WEIGHT_DECAY = 0.05
    BETAS = (0.9, 0.96)
    EPS = 1e-8
    RQ_LOSS_WEIGHT = 1e-7 # Weight for quantization + RQ classification loss

    # --- Scheduler ---
    WARMUP_EPOCHS = 20
    COSINE_LR_MIN = 1e-6
    WARMUP_LR_INIT = 1e-7

    # --- RQ ---
    RQ_M = 4 # Number of codebooks
    RQ_K_BITS = 8 # Bits per codebook (L = 2**8 = 256)

    # --- System ---
    NUM_WORKERS = 8
    SAVE_INTERVAL = 50 # Save checkpoint every N epochs
    INITIAL_FEATS_NPY = None # Optional: Path to pre-extracted initial CLIP features .npy file

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 2. Setup Device ===
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # === 3. Load Preprocessor and Dataset ===
    print(f"Loading CLIP model ({CLIP_MODEL_NAME}) for preprocessing...")
    # Load on CPU first to get preprocess function without occupying GPU memory
    _, clip_preprocess = clip.load(CLIP_MODEL_NAME, device='cpu')

    print(f"Loading Flickr30k Instance Dataset using: {FLICKR_GND_PKL_PATH}")
    try:
        dataset = FlickrInstanceDataset(
            gnd_pkl_path=FLICKR_GND_PKL_PATH,
            image_root_dir=FLICKR_IMAGE_DIR,
            split='db', # Load only the training data
            transform=clip_preprocess
        )
        num_classes = dataset.get_num_classes()
        print(f"Dataset loaded. Number of training images/classes: {len(dataset)}/{num_classes}")
        if num_classes == 0 or num_classes != len(dataset):
             print(f"错误或警告: 数据集类别数 ({num_classes}) 与图像数 ({len(dataset)}) 不匹配或为0。")
             print("对于实例判别，类别数应等于图像数。将强制设置为图像数。")
             num_classes = len(dataset)
             if num_classes == 0:
                  print("错误：数据集为空！")
                  exit()

    except Exception as e:
        print(f"加载或处理数据集时出错: {e}")
        exit()

    # === 4. Initialize Model ===
    print("Initializing tokenizer model (vitrqfc)...")
    model = vitrqfc(dec_depth=TOKENIZER_DEC_DEPTH, num_classes=num_classes)

    print(f"Loading and initializing encoder from CLIP model ({CLIP_MODEL_NAME})...")
    clip_model_loaded, _ = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model_loaded = clip_model_loaded.float() # Ensure correct dtype
    model.encoder = clip_model_loaded.visual
    del clip_model_loaded # Free memory
    model = model.to(DEVICE)
    print("Model initialized and moved to device.")

    # === 5. Get or Load Initial Features ===
    current_features_np = None # Variable to hold features for RQ in the next epoch
    if INITIAL_FEATS_NPY and os.path.exists(INITIAL_FEATS_NPY):
        print(f"Loading initial features from {INITIAL_FEATS_NPY}...")
        try:
            current_features_np = np.load(INITIAL_FEATS_NPY)
            if current_features_np.shape[0] != len(dataset):
                 print(f"警告: 加载的初始特征数量 ({current_features_np.shape[0]}) 与数据集大小 ({len(dataset)}) 不匹配！将重新提取。")
                 current_features_np = None
            else:
                 print(f"Initial features loaded, shape: {current_features_np.shape}")
        except Exception as e:
            print(f"加载初始特征时出错: {e}. 将重新提取。")
            current_features_np = None

    if current_features_np is None:
        print("Extracting initial features (using pre-trained CLIP)...")
        current_features_np = get_features(dataset, model.encoder, BATCH_SIZE, DEVICE, NUM_WORKERS)
        if current_features_np is None:
            print("错误：无法提取初始特征。")
            exit()
        initial_save_path = os.path.join(OUTPUT_DIR, f'{DATA_NAME}_initial_clip_feats.npy')
        try:
            np.save(initial_save_path, current_features_np)
            print(f"Initial features saved to {initial_save_path}")
        except Exception as e:
            print(f"保存初始特征时出错: {e}")


    # === 6. Setup Optimizer and Scheduler ===
    print("Setting up optimizer and learning rate scheduler...")
    try:
         optimizer = AdamW([{'params': get_trainable_params(model.fc), 'lr': LR_FC},
                            {'params': get_trainable_params(model.fc_rq), 'lr': LR_FC},
                            {'params': get_trainable_params(model.encoder), 'lr': LR_FC * LR_ENCODER_RATIO}
                           ], lr=LR_FC, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
    except AttributeError as e:
         print(f"模型定义错误: 无法找到 fc, fc_rq 或 encoder. 请检查 'vitrqfc' 类定义. Error: {e}")
         exit()

    # Epoch-based Cosine Annealing Scheduler
    cosine_t_max_epochs = NUM_EPOCHS - WARMUP_EPOCHS
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max(1, cosine_t_max_epochs), eta_min=COSINE_LR_MIN) # T_max must be > 0

    # Apply initial warmup learning rate manually
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
        group['lr'] = WARMUP_LR_INIT

    print("Optimizer and scheduler ready.")

    # === 7. Setup Loss and Logger ===
    criterion = nn.CrossEntropyLoss()
    log_file = os.path.join(OUTPUT_DIR, f'{DATA_NAME}_instance_tokenizer_train.log')
    logger = get_logger(log_file)
    print(f"Logging to {log_file}")

    # === 8. Training Loop ===
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        logger.info(f"--- Starting Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()

        # --- Learning Rate Warmup Logic ---
        current_lrs_str = []
        if epoch < WARMUP_EPOCHS:
             # Linear warmup from warmup_lr_init to initial_lr
             alpha = (epoch + 1) / float(WARMUP_EPOCHS) # alpha goes from 1/W to 1.0
             for group in optimizer.param_groups:
                 # Correct linear interpolation: start + (end - start) * alpha
                 new_lr = WARMUP_LR_INIT + (group['initial_lr'] - WARMUP_LR_INIT) * alpha
                 group['lr'] = new_lr
                 current_lrs_str.append(f"{new_lr:.2e}")
             lr_log_prefix = f"[Warmup {epoch+1}/{WARMUP_EPOCHS}]"
        elif epoch == WARMUP_EPOCHS:
             # Set to base LR at the end of warmup, before cosine starts
             for group in optimizer.param_groups:
                 group['lr'] = group['initial_lr']
                 current_lrs_str.append(f"{group['lr']:.2e}")
             lr_log_prefix = "[Warmup End]"
             # Cosine scheduler will start stepping at the end of this epoch
        else:
             # LR is controlled by scheduler.step() called at the end of the epoch
             current_lrs_str = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
             lr_log_prefix = "[Cosine Anneal]"
        logger.info(f"Epoch {epoch+1} LRs: {lr_log_prefix} {', '.join(current_lrs_str)}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} LRs: {lr_log_prefix} {', '.join(current_lrs_str)}")


        # --- Residual Quantization ---
        print(f"Epoch {epoch+1}: Performing Residual Quantization...")
        try:
             if not isinstance(current_features_np, np.ndarray):
                  raise TypeError("Features for RQ are not a NumPy array.")
             # Ensure residualquantizer function is available and works
             z_q_list, _ = residualquantizer(current_features_np, RQ_M, RQ_K_BITS)
             print(f"Residual Quantization complete. z_q_list length: {len(z_q_list)}")
             if not z_q_list or len(z_q_list) != RQ_M:
                  raise ValueError(f"residualquantizer did not return expected list of length {RQ_M}")
        except NameError:
             print("错误: 'residualquantizer' 函数未定义或无法导入。")
             exit()
        except Exception as e:
             logger.error(f"Error during residual quantization: {e}")
             print(f"Error during residual quantization: {e}")
             break # Stop training if RQ fails

        # --- Batch Training ---
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        epoch_loss_total = 0.0
        epoch_loss_ce_total = 0.0
        epoch_loss_ce_rq_total = 0.0
        epoch_loss_quant_total = 0.0
        processed_batches = 0

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for j, batch_data in pbar:
            try:
                # Ensure batch data is correct format
                if len(batch_data) != 3:
                     raise ValueError(f"DataLoader expected tuple of 3 elements, got {len(batch_data)}")
                img, target, idx = batch_data
                img = img.to(DEVICE)
                # Target should be LongTensor for CrossEntropyLoss
                target = target.long().to(DEVICE)
                if target.min() < 0:
                     print(f"\n警告: Batch {j} 包含无效的 target 标签 (< 0): {target[target<0]}")
                     continue # Skip batch with invalid labels (e.g., from image loading errors)


                # Prepare z_q for the current batch using indices idx
                z_q_batch = [torch.tensor(z_q_list[l][idx.numpy()], dtype=torch.float32).to(DEVICE) for l in range(RQ_M)]

                # Forward pass
                z, loss_quant, output, output_rq = model(img, z_q_batch)

                # Calculate loss
                loss_ce = criterion(output, target)

                loss_ce_rq_list = [criterion(output_rq[k], target) for k in range(len(output_rq))]
                loss_ce_rq = sum(loss_ce_rq_list)

                loss_quant_sum = sum(loss_quant) if isinstance(loss_quant, list) else loss_quant

                loss = RQ_LOSS_WEIGHT * (loss_ce_rq + loss_quant_sum) + loss_ce

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate losses for logging
                epoch_loss_total += loss.item()
                epoch_loss_ce_total += loss_ce.item()
                epoch_loss_ce_rq_total += loss_ce_rq.item()
                epoch_loss_quant_total += loss_quant_sum.item()
                processed_batches += 1

                # Update progress bar description
                lr_enc = optimizer.param_groups[2]['lr']
                pbar.set_description(f"E{epoch+1} Loss:{loss.item():.4f} CE:{loss_ce.item():.2f}")
                pbar.set_postfix_str(f"LR_Enc:{lr_enc:.2e}")

            except Exception as e:
                logger.error(f"训练 Batch {j} 时出错: {e}")
                print(f"\n错误: 训练 Batch {j} 时出错: {e}")
                # Optionally add more debugging or skip the batch
                # import traceback; traceback.print_exc() # Print detailed traceback
                continue # Skip to next batch

        # --- End of Epoch ---
        avg_loss = epoch_loss_total / processed_batches if processed_batches > 0 else 0
        avg_loss_ce = epoch_loss_ce_total / processed_batches if processed_batches > 0 else 0
        avg_loss_ce_rq = epoch_loss_ce_rq_total / processed_batches if processed_batches > 0 else 0
        avg_loss_quant = epoch_loss_quant_total / processed_batches if processed_batches > 0 else 0
        epoch_duration = time.time() - epoch_start_time

        logger.info(f"--- Epoch {epoch+1} Summary ---")
        logger.info(f"Duration: {epoch_duration:.2f}s")
        logger.info(f"Avg Loss: {avg_loss:.4f}")
        logger.info(f"Avg Loss CE: {avg_loss_ce:.4f}")
        logger.info(f"Avg Loss CE_RQ: {avg_loss_ce_rq:.4f}")
        logger.info(f"Avg Loss Quant: {avg_loss_quant:.4f}")
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s")


        # Step the Cosine Annealing scheduler (after warmup phase)
        if epoch >= WARMUP_EPOCHS:
            scheduler_cosine.step()

        # --- Re-extract Features for Next Epoch ---
        if epoch < NUM_EPOCHS - 1: # No need to extract after the last epoch
            logger.info(f"Epoch {epoch+1}: Re-extracting features with updated encoder...")
            print(f"Epoch {epoch+1}: Re-extracting features...")
            new_features_np = get_features(dataset, model.encoder, BATCH_SIZE, DEVICE, NUM_WORKERS)
            if new_features_np is None:
                 logger.error("特征提取失败，训练终止。")
                 print("错误: 特征提取失败，训练终止。")
                 break # Stop training
            else:
                 current_features_np = new_features_np # Update features for the next RQ step
                 logger.info(f"Features re-extracted, shape: {current_features_np.shape}")
                 print("Features re-extracted.")


        # --- Save Checkpoint ---
        if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            logger.info(f"Saving checkpoint at epoch {epoch+1}...")
            print(f"Saving checkpoint...")
            state = {'net': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler_cosine.state_dict(),
                     'epoch': epoch + 1,
                     'loss': avg_loss,
                     'num_classes': num_classes}
            ckpt_path = os.path.join(OUTPUT_DIR, f'{DATA_NAME}_instance_tokenizer_epoch_{epoch+1}.pth')
            try:
                 torch.save(state, ckpt_path)
                 logger.info(f"Checkpoint saved to {ckpt_path}")
                 print(f"Checkpoint saved to {ckpt_path}")

                 # Save the features extracted *at the end* of this epoch
                 feats_path = os.path.join(OUTPUT_DIR, f'{DATA_NAME}_finetuned_feats_epoch_{epoch+1}.npy')
                 np.save(feats_path, current_features_np)
                 logger.info(f"Fine-tuned features saved to {feats_path}")
                 print(f"Fine-tuned features saved to {feats_path}")
            except Exception as e:
                 logger.error(f"保存 checkpoint 或特征时出错: {e}")
                 print(f"错误: 保存 checkpoint 或特征时出错: {e}")


    # === End of Training Loop ===
    total_time = time.time() - start_time
    logger.info(f"--- Tokenizer Training Finished --- Total Time: {total_time:.2f}s")
    print(f"--- Tokenizer Training Finished --- Total Time: {total_time:.2f}s")