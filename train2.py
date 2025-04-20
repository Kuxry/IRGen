import os
import time
import pickle  # <--- 添加导入
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb

# --- 项目内导入 ---
# 假设这些路径是正确的
from dataset.dataset_flickr_caption_rq import FlickrCaptionRQDataset
from dataset.dataset_flickr_caption_val import FlickrCaptionRQValDataset
from model.CLIP.clip import clip
from model.IRGen import IRGenText2Image
from utils.utils import get_trainable_params
from utils.logger import get_logger

if __name__ == '__main__':
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- 配置修改 ---
    config = {
        "metadata_pkl_path": "/home/iiserver31/Workbench/likaipeng/IRgen/data/flick30k/flickr30k_text_to_image_retrieval_from_json.pkl",
        'rq_codes_pkl_path': './output_rq/flickr30k_rq_codes_m4_k8_full.pkl',
        'output_dir': './output_models_t2i/author_params_modified', # 建议修改输出目录以区分实验
        'clip_model_name': 'ViT-B/16',
        'rq_m': 4,
        'rq_k_bits': 8,
        'lr': 3e-6,  # 使用较低的学习率
        'batch_size': 64,
        'num_epochs': 100,
        'smoothing': 0.1,
        'decoder_depth': 8,      # 降低解码器深度
        'weight_decay': 0.01,    # 降低权重衰减
        'betas': (0.9, 0.96),
        'eps': 1e-8,
        'dropout': 0.5,          # 增加 dropout
        'patience': 3           # 减小早停耐心
    }
    # --- 结束配置修改 ---

    # 确保新的输出目录存在
    os.makedirs(config['output_dir'], exist_ok=True)

    # WandB 初始化
    wandb.init(project='IRGen-Flickr30k-T2I-improved', name='decoder8_dropout0.5_wd0.01_lr3e-6_patience5', config=config) # 更新 run name

    # 加载 RQ codes
    try:
        with open(config['rq_codes_pkl_path'], 'rb') as f:
            rq_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: RQ codes file not found at {config['rq_codes_pkl_path']}")
        exit()
    id_len = config['rq_m']
    num_cls = 2 ** config['rq_k_bits']

    # 数据集
    try:
        train_ds = FlickrCaptionRQDataset(config['metadata_pkl_path'], config['rq_codes_pkl_path'], max_caption_len=77)
        val_ds = FlickrCaptionRQValDataset(config['metadata_pkl_path'], config['rq_codes_pkl_path'], max_caption_len=77)
    except FileNotFoundError:
         print(f"Error: Metadata file not found at {config['metadata_pkl_path']}")
         exit()

    if len(train_ds) == 0:
        print("Error: Training dataset is empty. Check metadata and RQ code paths.")
        exit()

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], sampler=RandomSampler(train_ds), num_workers=8, pin_memory=True)
    run_val = len(val_ds) > 0
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'] * 2, shuffle=False, num_workers=4, pin_memory=True) if run_val else None

    # 模型
    model = IRGenText2Image(dec_depth=config['decoder_depth'], num_classes=num_cls, id_len=id_len,
                            embed_dim=512, num_heads=8, dropout=config['dropout'])
    try:
        clip_model, _ = clip.load(config['clip_model_name'], device=device)
    except Exception as e:
        print(f"Error loading CLIP model '{config['clip_model_name']}': {e}")
        exit()
    clip_model = clip_model.float()
    clip_model.eval()

    model.to(device)

    # 优化器 & Scheduler
    optimizer = AdamW(
        get_trainable_params(model.decoder),
        lr=config['lr'], betas=config['betas'], eps=config['eps'], weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3,
        patience=config['patience'] // 2,
        min_lr=1e-7,
        verbose=True
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config['smoothing'])
    logger = get_logger(os.path.join(config['output_dir'], 'train.log'))

    best_val_loss = float('inf')
    early_stopping_wait = 0
    early_stopping_patience = config['patience']

    # 训练循环
    for epoch in range(config['num_epochs']):
        # --- 训练 ---
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Train]")

        for step, (caps, tgt) in enumerate(train_pbar):
            caps, tgt = caps.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            with torch.no_grad():
                text_feat = clip_model.encode_text(caps).float()
            optimizer.zero_grad()
            out = model(text_feat=text_feat, tgt=tgt)[:, :id_len, :]
            loss = criterion(out.reshape(-1, num_cls), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model.decoder), max_norm=1.0)
            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({'loss': f"{batch_loss:.4f}", 'lr': f"{current_lr:.2e}"})

            global_step = epoch * len(train_loader) + step
            wandb.log({
                'train_loss_step': batch_loss,
                'learning_rate': current_lr
            }, step=global_step)

        avg_train_loss = total_train_loss / len(train_loader)

        # --- 验证 ---
        current_val_loss = None
        if run_val and val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Val]")
            with torch.no_grad():
                for caps, tgt in val_pbar:
                    caps, tgt = caps.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
                    text_feat = clip_model.encode_text(caps).float()
                    o = model(text_feat=text_feat, tgt=tgt)[:, :id_len, :]
                    l = criterion(o.reshape(-1, num_cls), tgt.reshape(-1))
                    total_val_loss += l.item()
                    val_pbar.set_postfix({'val_loss': f"{l.item():.4f}"})
            current_val_loss = total_val_loss / len(val_loader)
            scheduler.step(current_val_loss)

        # --- 日志 & WandB (Epoch End) ---
        epoch_metrics = {'epoch': epoch + 1, 'avg_train_loss': avg_train_loss}
        log_msg = f"Epoch {epoch + 1}/{config['num_epochs']} Train Loss={avg_train_loss:.4f}"
        if current_val_loss is not None:
            epoch_metrics['avg_val_loss'] = current_val_loss
            log_msg += f", Val Loss={current_val_loss:.4f}"
        else:
             log_msg += ", Val Loss=N/A"
        wandb.log(epoch_metrics)
        print(log_msg)
        logger.info(log_msg)

        # --- Early Stopping & Checkpointing ---
        if current_val_loss is not None:
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                early_stopping_wait = 0
                # --- 修改：保存最佳模型为 .pkl ---
                best_model_path = os.path.join(config['output_dir'], 'best_model.pkl') # 改扩展名
                try:
                    with open(best_model_path, 'wb') as f:
                        pickle.dump(model.state_dict(), f) # 使用 pickle.dump 保存 state_dict
                    print(f"Validation loss improved. Saved best model state_dict to {best_model_path}")
                    logger.info(f"Validation loss improved to {best_val_loss:.4f}. Saved best model state_dict to {best_model_path}")
                except Exception as e:
                    print(f"Error saving best model state_dict with pickle: {e}")
                    logger.error(f"Error saving best model state_dict with pickle: {e}")
                # --- 结束修改 ---
            else:
                early_stopping_wait += 1
                print(f"Validation loss did not improve for {early_stopping_wait} epoch(s).")
                if early_stopping_wait >= early_stopping_patience:
                    print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
                    logger.warning(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

        # --- 修改：定期保存 checkpoint 为 .pkl ---
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch + 1}.pkl') # 改扩展名
            ckpt_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(), # Checkpoint 包含 state_dict
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            try:
                with open(ckpt_path, 'wb') as f:
                    pickle.dump(ckpt_data, f) # 使用 pickle.dump 保存整个 checkpoint 字典
                print(f"Saved checkpoint to {ckpt_path}")
                logger.info(f"Saved checkpoint at epoch {epoch + 1} to {ckpt_path}")
            except Exception as e:
                print(f"Error saving checkpoint with pickle: {e}")
                logger.error(f"Error saving checkpoint with pickle: {e}")
        # --- 结束修改 ---


    # --- 训练结束 ---
    # --- 修改：保存最终模型为 .pkl ---
    final_model_path = os.path.join(config['output_dir'], 'final_model.pkl') # 改扩展名
    try:
        with open(final_model_path, 'wb') as f:
            pickle.dump(model.state_dict(), f) # 使用 pickle.dump 保存 state_dict
        print(f"Saved final model state_dict to {final_model_path}")
        logger.info(f"Training finished. Saved final model state_dict to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model state_dict with pickle: {e}")
        logger.error(f"Error saving final model state_dict with pickle: {e}")
    # --- 结束修改 ---

    wandb.finish()