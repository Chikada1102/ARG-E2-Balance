# ARG E2 Balance

基于 VM-UNet Small 的 Synapse 二维多器官分割工程。E2 将解剖关系模块 ARG 接入编码器第 3 阶段最后一个 VSSBlock，利用器官软原型及其空间关系向 SS2D 的 `B`、`C` 输入写入有界残差。训练使用 Balance 在线区域混合。推理返回单个 `[B, 9, H, W]` 分割 logits 张量。

## 环境

使用支持 VM-UNet 原生 selective scan 的 Linux CUDA 环境。可复用已有的 `vmunet` 环境，或按 [VM-UNet 上游安装说明](https://github.com/JCruan519/VM-UNet/blob/b87827eb5a5faff00bd4b7b6505e56d3ca813ca2/README.md) 安装与 GPU、PyTorch 和 Python 版本匹配的 `torch`、`torchvision`、`causal-conv1d` 和 `mamba-ssm`。然后在仓库根目录安装其余依赖。

```bash
python -m pip install -r requirements.txt
```

## 本地文件

源码仓库包含数据读取器、配置、训练与评估代码。训练前把自己的 Synapse 数据、原病例名单和 VMamba Small 预训练文件放在以下路径。

| 路径 | 内容 |
|---|---|
| `data/Synapse/train_npz/` | 训练用二维切片，NPZ 包含 `image`、`label` |
| `data/Synapse/test_vol_h5/` | 测试病例体积，文件名为 `<case>.npy.h5` |
| `data/Synapse/lists/lists_Synapse/train.txt` | 原 18 个训练病例的切片名单 |
| `data/Synapse/lists/lists_Synapse/test_vol.txt` | 原 12 个测试病例的体积名单 |
| `pre_trained_weights/vmamba_small_e238_ema.pth` | VMamba Small 原始预训练权重 |

`.gitignore` 会排除本地数据、权重、日志和训练输出。复现实验请沿用相同的病例名单与预训练文件。训练入口校验 18/12 划分和病例无重叠，并在审计中记录文件哈希。

## 自检与训练

```bash
# CPU 自检使用合成数据与可微参考扫描
OMP_NUM_THREADS=1 python self_check_arg.py

# 在 CUDA 环境训练 E2，启动脚本会先检查原生扫描核
bash run_E2.sh
```

默认输入尺寸为 224×224，batch size 为 32，种子为 42，共训练 300 轮。优化器为 AdamW，初始学习率为 0.001，使用 CosineAnnealingLR（`T_max=100`，`eta_min=1e-5`）。训练目标为主分割损失 `CeDiceLoss(9, [1, 1])` 加 0.1 倍语义损失和 0.05 倍存在性损失。

结果默认写入 `results/arg_e2_balance/`。其中 `best.pth` 按最低主分割训练损失选择，`final.pth` 对应第 300 轮。病例级、器官级和总体 Dice/HD95 位于 `audit/`，ARG、在线增强和训练设置也会写入该目录。

## 源码入口

| 文件 | 作用 |
|---|---|
| `models/vmunet/vmunet.py` | VM-UNet 模型构造与 E2 模块接入 |
| `models/vmunet/vmamba.py` | VSS 编码器、解码器与 SS2D |
| `models/vmunet/arg_vss.py` | 器官原型、解剖关系和 B/C 反馈 |
| `configs/config_setting_synapse.py` | Synapse 路径与训练配置 |
| `train_synapse.py`、`engine_synapse.py` | 训练、检查点、推理与评估 |
| `datasets/region_mix.py`、`datasets/region_mix_refinement.py` | 在线区域混合与 Balance 反馈 |

## 来源与许可

基础实现来自 [VM-UNet](https://github.com/JCruan519/VM-UNet/tree/b87827eb5a5faff00bd4b7b6505e56d3ca813ca2)。原作者与修改文件见 [NOTICE](NOTICE)，许可见 [Apache-2.0](LICENSE)。
