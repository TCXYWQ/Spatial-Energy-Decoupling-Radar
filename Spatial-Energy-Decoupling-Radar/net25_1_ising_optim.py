import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pywt
import torch.amp as amp
import warnings
import sys
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 强制后台渲染
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ================= 0. 炼丹超参数控制台 =================
SEQ_LEN = 32           
BATCH_SIZE = 12         
NUM_WORKERS = 0        
LR = 3e-4              
T_0_EPOCHS = 10 
WINDOW = 400           
STRIDE = 10            
MAX_EPOCHS = 150
# =======================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "Mamba_Dataset_Complex")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_v25_1_Ising_Optim") # 🌟 V25.1专属输出文件夹

WEIGHT_DIR = os.path.join(OUTPUT_DIR, "weights")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(WEIGHT_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)
BPM_MEAN, BPM_STD = 75.0, 20.0

# ================= 1. 核心网络架构 (ICEF优化版 + DSEG + Mamba) =================

class PurePyTorchMamba(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dropout=0.2):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        seq_len = x.size(1)
        x_proj = self.in_proj(x)
        x_ssm, x_gate = x_proj.chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_ssm.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        ssm_out = x_conv * torch.sigmoid(self.x_proj(x_conv)[..., :1]) 
        return self.dropout(self.out_proj(ssm_out * F.silu(x_gate)))

# 🌟 V25.1 优化点 1: 可学习温度的 伊辛耦合能量滤波器
class IsingCoupledEnergyFilter(nn.Module):
    def __init__(self, dim=128, init_beta=1.0, num_steps=3): # 步数增加到 3
        super().__init__()
        self.dim = dim
        # 变成可学习参数，网络自适应调节远近场的“退火温度”
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))  
        self.num_steps = num_steps  
        
        self.J_matrix = nn.Linear(dim, dim, bias=False)
        self.h_field = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim)
        )
        nn.init.orthogonal_(self.J_matrix.weight)

    def forward(self, x):
        local_energy = torch.mean(x, dim=1)  
        h = self.h_field(local_energy).unsqueeze(1) 
        spin_state = torch.tanh(x) 
        
        for _ in range(self.num_steps):
            interaction = self.J_matrix(spin_state)
            spin_state = torch.tanh(self.beta * (interaction + h))
            
        energy_mask = (spin_state + 1.0) / 2.0
        return x * energy_mask

# 🌟 动态门控注意力 (DSEG) 
class DynamicCrossTargetAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
        self.dynamic_gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()  
        )

    def forward(self, target_feat, interference_feat):
        attn_output, _ = self.multihead_attn(target_feat, interference_feat, interference_feat)
        
        t_pool = torch.mean(target_feat, dim=1)         
        i_pool = torch.mean(interference_feat, dim=1)   
        gate_input = torch.cat([t_pool, i_pool], dim=-1) 
        
        dynamic_alpha = self.dynamic_gate(gate_input).unsqueeze(1) 
        return self.norm(target_feat - dynamic_alpha * attn_output)

class MambaV9_Ising_Optimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)), 
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)), 
            nn.Conv2d(64, 128, kernel_size=(8, 1)), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        
        self.ising_filter_1 = IsingCoupledEnergyFilter(dim=128, num_steps=3)
        self.ising_filter_2 = IsingCoupledEnergyFilter(dim=128, num_steps=3)
        
        self.ctac_1to2 = DynamicCrossTargetAttention(128)
        self.ctac_2to1 = DynamicCrossTargetAttention(128)
        
        self.mamba_blocks = nn.ModuleList([PurePyTorchMamba(d_model=128, dropout=0.2) for _ in range(4)])
        self.norm = nn.LayerNorm(128)
        
        self.head1 = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.head2 = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x1, x2, B, S):
        f1, f2 = self.backbone(x1).squeeze(2).permute(0, 2, 1), self.backbone(x2).squeeze(2).permute(0, 2, 1) 
        
        f1_ising, f2_ising = self.ising_filter_1(f1), self.ising_filter_2(f2)
        f1_clean, f2_clean = self.ctac_1to2(f1_ising, f2_ising), self.ctac_2to1(f2_ising, f1_ising)
        
        f1_seq, f2_seq = torch.mean(f1_clean, 1).view(B, S, 128), torch.mean(f2_clean, 1).view(B, S, 128)
        
        for mamba in self.mamba_blocks:
            f1_seq, f2_seq = self.norm(f1_seq + mamba(f1_seq)), self.norm(f2_seq + mamba(f2_seq))
            
        return self.head1(f1_seq).squeeze(-1), self.head2(f2_seq).squeeze(-1)

# ================= 2. 带有精准 P_ID 映射的 Dataset =================
def compute_full_cwt(radar_complex_waveform, fs=20.0):
    freqs_hz = np.linspace(0.7, 3.0, 64) 
    scales = pywt.frequency2scale('cmor1.5-1.0', freqs_hz / fs)
    coef, _ = pywt.cwt(radar_complex_waveform, scales, 'cmor1.5-1.0', sampling_period=1.0/fs)
    mag, real, imag = np.abs(coef), np.real(coef), np.imag(coef)
    def norm(x): return (x - np.mean(x)) / (np.std(x) + 1e-8)
    return np.stack([norm(mag), norm(real), norm(imag)], axis=0).astype(np.float32)

class ContinuousMambaDataset(Dataset):
    def __init__(self, X_full, Y_full, P_full, pair_indices, is_train=False):
        self.sequences = []
        self.is_train = is_train
        self.max_shift = 15  
        self.cwt_cache = {}
        
        for pair_idx in pair_indices:
            idx1, idx2 = 2 * pair_idx, 2 * pair_idx + 1
            if idx2 >= len(X_full): continue
            
            p_id = P_full[pair_idx] 
            
            if idx1 not in self.cwt_cache: self.cwt_cache[idx1] = compute_full_cwt(X_full[idx1])
            if idx2 not in self.cwt_cache: self.cwt_cache[idx2] = compute_full_cwt(X_full[idx2])
                
            valid_windows = []
            for start in range(self.max_shift, X_full.shape[1] - WINDOW - self.max_shift + 1, STRIDE):
                bpm1, bpm2 = np.mean(Y_full[idx1, start:start+WINDOW]), np.mean(Y_full[idx2, start:start+WINDOW])
                if 40.0 <= bpm1 <= 180.0 and 40.0 <= bpm2 <= 180.0:
                    valid_windows.append((start, bpm1, bpm2, True))
                else:
                    valid_windows.append((start, 0, 0, False))
            
            seq = []
            for start, b1, b2, valid in valid_windows:
                if valid:
                    seq.append((start, b1, b2))
                    if len(seq) == SEQ_LEN:
                        self.sequences.append((idx1, idx2, p_id, list(seq)))
                        seq.pop(0) 
                else: seq = [] 

    def __len__(self): return len(self.sequences)
    
    def __getitem__(self, idx):
        idx1, idx2, p_id, seq = self.sequences[idx]
        f1_tensor, f2_tensor = torch.empty((SEQ_LEN, 3, 64, WINDOW)), torch.empty((SEQ_LEN, 3, 64, WINDOW))
        ny1_tensor, ny2_tensor = torch.empty(SEQ_LEN), torch.empty(SEQ_LEN)
        
        shift1 = np.random.randint(-self.max_shift, self.max_shift + 1) if self.is_train else 0
        shift2 = np.random.randint(-self.max_shift, self.max_shift + 1) if self.is_train else 0
        
        for i, (start, bpm1, bpm2) in enumerate(seq):
            s1, e1 = start + shift1, start + WINDOW + shift1
            s2, e2 = start + shift2, start + WINDOW + shift2
            f1_tensor[i] = torch.from_numpy(self.cwt_cache[idx1][:, :, s1:e1])
            f2_tensor[i] = torch.from_numpy(self.cwt_cache[idx2][:, :, s2:e2])
            ny1_tensor[i] = float(bpm1 - BPM_MEAN) / BPM_STD
            ny2_tensor[i] = float(bpm2 - BPM_MEAN) / BPM_STD
            
        return f1_tensor, f2_tensor, ny1_tensor, ny2_tensor, p_id

# ================= 3. 训练主引擎 =================
def run_ising_optim_train():
    try:
        X_full = np.load(os.path.join(DATA_DIR, "radar_X_features.npy"))
        Y_full = np.load(os.path.join(DATA_DIR, "ecg_Y_labels.npy"))
        P_full = np.load(os.path.join(DATA_DIR, "radar_P_positions.npy"))
    except FileNotFoundError:
        print("❌ 找不到数据集文件！")
        sys.exit(1)
    
    file_indices = np.arange(len(P_full))
    train_indices, val_indices = train_test_split(
        file_indices, test_size=0.15, random_state=42, stratify=P_full 
    )
    
    train_dataset = ContinuousMambaDataset(X_full, Y_full, P_full, train_indices, is_train=True)
    val_dataset = ContinuousMambaDataset(X_full, Y_full, P_full, val_indices, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaV9_Ising_Optimized().to(device)
    
    if hasattr(torch, 'compile') and sys.platform != 'win32':
        try:
            model = torch.compile(model)
            with torch.no_grad(), amp.autocast(device_type='cuda'):
                dummy = torch.randn(BATCH_SIZE * SEQ_LEN, 3, 64, WINDOW).to(device)
                _ = model(dummy, dummy, BATCH_SIZE, SEQ_LEN)
        except:
            model = MambaV9_Ising_Optimized().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0_EPOCHS)
    scaler = amp.GradScaler()
    
    print(f"🚀 V25.1 [伊辛调优版] 启动 | 动态自适应温度 + 困难样本挖掘")
    best_mae = 100.0

    for epoch in range(MAX_EPOCHS):
        epoch_start_time = time.time()  
        
        model.train()
        total_loss, total_loss_t1, total_loss_t2 = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{MAX_EPOCHS} [Train]", leave=False, dynamic_ncols=True)
        for x1, x2, ny1, ny2, _ in pbar:
            B, S, C, H, W = x1.shape 
            x1, x2 = x1.view(B*S, C, H, W).to(device, non_blocking=True), x2.view(B*S, C, H, W).to(device, non_blocking=True)
            ny1, ny2 = ny1.to(device, non_blocking=True), ny2.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):
                p1, p2 = model(x1, x2, B, S)
                
                # 🌟 V25.1 优化点 2: 困难样本挖掘回归 Loss (Hard Example Mining)
                loss_t1_raw = F.smooth_l1_loss(p1, ny1, reduction='none')
                loss_t2_raw = F.smooth_l1_loss(p2, ny2, reduction='none')
                
                # 计算动态惩罚权重，截断最大惩罚倍数防止初期崩盘
                w1 = torch.exp(torch.clamp(loss_t1_raw.detach() * 2.0, max=1.5))
                w2 = torch.exp(torch.clamp(loss_t2_raw.detach() * 2.0, max=1.5))
                
                # 加权 Loss
                loss_t1 = torch.mean(loss_t1_raw * w1)
                loss_t2 = torch.mean(loss_t2_raw * w2)
                
                loss = loss_t1 + 2.0 * loss_t2 + 0.5 * torch.abs(loss_t1 - loss_t2)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            
            # 用于显示的依然是不带权重的真实误差
            total_loss += loss.item()
            total_loss_t1 += torch.mean(loss_t1_raw).item() 
            total_loss_t2 += torch.mean(loss_t2_raw).item()
            pbar.set_postfix({'Loss': f"{loss.item():.3f}"}) 

        # ---------- 分层验证与全量监控 ----------
        model.eval()
        p1_all, p2_all, y1_all, y2_all, pids_all = [], [], [], [], []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{MAX_EPOCHS} [Valid]", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for x1, x2, ny1, ny2, batch_pids in val_pbar:
                B, S, C, H, W = x1.shape 
                x1, x2 = x1.view(B*S, C, H, W).to(device, non_blocking=True), x2.view(B*S, C, H, W).to(device, non_blocking=True)
                p1, p2 = model(x1, x2, B, S)
                
                p1_all.extend(p1.cpu().numpy()[:, -1] * BPM_STD + BPM_MEAN)
                p2_all.extend(p2.cpu().numpy()[:, -1] * BPM_STD + BPM_MEAN)
                y1_all.extend(ny1.numpy()[:, -1] * BPM_STD + BPM_MEAN)
                y2_all.extend(ny2.numpy()[:, -1] * BPM_STD + BPM_MEAN)
                pids_all.extend(batch_pids.numpy())

        p1_all, p2_all = np.array(p1_all), np.array(p2_all)
        y1_all, y2_all = np.array(y1_all), np.array(y2_all)
        pids_all = np.array(pids_all)

        diff1, diff2 = p1_all - y1_all, p2_all - y2_all
        mae_t1, mae_t2 = np.mean(np.abs(diff1)), np.mean(np.abs(diff2))
        diff_all = np.concatenate([diff1, diff2])
        overall_mae = np.mean(np.abs(diff_all))
        overall_rmse = np.sqrt(np.mean(diff_all**2))
        
        def calc_loa(diff):
            if len(diff) == 0: return 0, 0, 0, 0
            bias = np.mean(diff)
            sd = np.std(diff)
            return bias - 1.96*sd, bias + 1.96*sd, bias, 3.92*sd
            
        loa_l_all, loa_u_all, bias_all, width_all = calc_loa(diff_all)
        loa_l1, loa_u1, bias1, width1 = calc_loa(diff1)
        loa_l2, loa_u2, bias2, width2 = calc_loa(diff2)

        print(f"\n" + "="*80)
        print(f"🎯 Epoch {epoch+1:03d} | 耗时: {time.time() - epoch_start_time:.1f}s | V25.1 困难样本自适应模式")
        print(f"📉 [Loss] 总:{total_loss/len(train_loader):.3f} | T1:{total_loss_t1/len(train_loader):.3f} | T2:{total_loss_t2/len(train_loader):.3f}")
        print("-" * 80)
        print(f"🏆 [核心指标] Overall MAE: {overall_mae:.2f} | RMSE: {overall_rmse:.2f}")
        print(f"   -> Target 1 MAE: {mae_t1:.2f} BPM (LoA宽: {width1:.2f} | Bias: {bias1:.2f})")
        print(f"   -> Target 2 MAE: {mae_t2:.2f} BPM (LoA宽: {width2:.2f} | Bias: {bias2:.2f})")
        
        # 🌟 每轮无死角打印九宫格探伤
        print("\n" + "🌟"*18 + f" 伊辛微雕·九宫格深潜报告 (每轮刷新) " + "🌟"*18)
        print(f"{'Position ID':<12} | {'Sample Count':<12} | {'Target 1 MAE':<14} | {'Target 2 MAE':<14}")
        print("-" * 65)
        for pid in range(1, 10):
            idx = np.where(pids_all == pid)[0]
            if len(idx) > 0:
                mae1_pos = np.mean(np.abs(p1_all[idx] - y1_all[idx]))
                mae2_pos = np.mean(np.abs(p2_all[idx] - y2_all[idx]))
                print(f"  Pos {pid:<6} |  {len(idx):<11} |  {mae1_pos:.2f} BPM        |  {mae2_pos:.2f} BPM")
        print("🌟"*56 + "\n")
        print("="*80)
        
        weight_path = os.path.join(WEIGHT_DIR, f"mamba_v25_1_epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), weight_path)
        
        if overall_mae < best_mae:
            best_mae = overall_mae
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_mamba_v25_1.pth"))
            print(f"🔥 极限微雕爆发! 新 MAE 记录: {overall_mae:.2f}！")

        # ================= 🎨 每轮全量绘图 =================
        plot_len = min(120, len(y1_all)) 
        
        # 【图 1】追踪图
        fig_track, axs_track = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        axs_track[0].plot(y1_all[:plot_len], label='GT', color='black', linewidth=1.5); axs_track[0].plot(p1_all[:plot_len], label='Ours', color='green', linewidth=1.5, linestyle='--')
        axs_track[0].set_title(f"T1 Tracking (Epoch {epoch+1:03d})", fontweight='bold'); axs_track[0].set_ylabel("HR"); axs_track[0].legend(); axs_track[0].grid(True)
        axs_track[1].plot(y2_all[:plot_len], label='GT', color='black', linewidth=1.5); axs_track[1].plot(p2_all[:plot_len], label='Ours', color='green', linewidth=1.5, linestyle='--')
        axs_track[1].set_title(f"T2 Tracking (Epoch {epoch+1:03d})", fontweight='bold'); axs_track[1].set_ylabel("HR"); axs_track[1].legend(); axs_track[1].grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, f"01_tracking_epoch_{epoch+1:03d}.png"), dpi=200); plt.close(fig_track)

        # 【图 2】CDF
        fig_cdf, ax_cdf = plt.subplots(figsize=(8, 6))
        ax_cdf.plot(np.sort(np.abs(diff1)), np.linspace(0, 1, len(diff1)), label=f'T1 (MAE: {mae_t1:.2f})', color='blue', linewidth=2)
        ax_cdf.plot(np.sort(np.abs(diff2)), np.linspace(0, 1, len(diff2)), label=f'T2 (MAE: {mae_t2:.2f})', color='red', linewidth=2)
        ax_cdf.set_title(f"CDF Error (Epoch {epoch+1:03d})", fontweight='bold'); ax_cdf.set_xlabel("Error"); ax_cdf.set_ylabel("CDF"); ax_cdf.grid(True); ax_cdf.set_xlim(0, 20); ax_cdf.legend()
        plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, f"02_cdf_epoch_{epoch+1:03d}.png"), dpi=200); plt.close(fig_cdf)

        # 【图 3】Bland-Altman
        fig_ba, axs_ba = plt.subplots(1, 2, figsize=(16, 6))
        axs_ba[0].scatter((p1_all + y1_all)/2, diff1, alpha=0.5, color='blue', s=10)
        axs_ba[0].axhline(bias1, color='black', label=f'Bias: {bias1:.2f}'); axs_ba[0].axhline(loa_u1, color='red', linestyle='--'); axs_ba[0].axhline(loa_l1, color='red', linestyle='--')
        axs_ba[0].set_title(f"Bland-Altman: T1", fontweight='bold'); axs_ba[0].grid(True); axs_ba[0].legend()
        axs_ba[1].scatter((p2_all + y2_all)/2, diff2, alpha=0.5, color='red', s=10)
        axs_ba[1].axhline(bias2, color='black', label=f'Bias: {bias2:.2f}'); axs_ba[1].axhline(loa_u2, color='blue', linestyle='--'); axs_ba[1].axhline(loa_l2, color='blue', linestyle='--')
        axs_ba[1].set_title(f"Bland-Altman: T2", fontweight='bold'); axs_ba[1].grid(True); axs_ba[1].legend()
        plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, f"03_ba_epoch_{epoch+1:03d}.png"), dpi=200); plt.close(fig_ba)

        scheduler.step()

if __name__ == "__main__":
    run_ising_optim_train()