# Prompt Conditional VAE Image Generator

基于 **条件变分自编码器 Conditional Variational Autoencoder, CVAE** 的轻量级图像生成项目。

本项目可以使用一组图片进行训练，并根据简单的文本提示词 prompt 生成对应类别或风格的图片。项目主要用于学习生成模型的基础流程，包括数据集构建、VAE 训练、隐空间采样、图像重建、模型保存、命令行推理、FastAPI 后端接口和 Gradio 可视化界面。

> 注意：本项目是一个纯 VAE / CVAE 教学项目，不是 Stable Diffusion、Midjourney、DALL·E 这类大型文生图模型。由于 VAE 本身的机制限制，生成图片可能比较模糊，尤其是在自然图像、复杂场景和高清图片上更明显。

---

## 1. 项目功能

* 支持自定义图片数据集训练
* 支持使用文件夹名作为提示词
* 支持使用同名 `.txt` 文件作为图片 caption
* 支持条件 VAE 图像生成
* 支持输入 prompt 生成图片
* 支持输入图片进行重建
* 支持训练过程保存 checkpoint
* 支持保存训练过程中的重建效果图
* 提供命令行训练脚本
* 提供命令行生成脚本
* 提供 FastAPI 后端接口
* 提供 Gradio 网页界面
* 支持 Windows CMD、PowerShell、Linux、WSL、macOS

---

## 2. 项目目录结构

```text
vae_prompt_project_v2/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── models/
│   │   ├── conditional_vae.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── checkpoint_service.py
│   │   ├── generation_service.py
│   │   └── training_service.py
│   └── utils/
│       ├── dataset.py
│       ├── image_io.py
│       ├── project_paths.py
│       └── text.py
├── configs/
│   └── default.yaml
├── datasets/
│   └── my_images/
├── outputs/
│   ├── generated/
│   └── vae/
├── scripts/
│   ├── train_vae.py
│   ├── generate.py
│   ├── reconstruct.py
│   └── make_default_captions.py
├── webui/
│   └── app_gradio.py
├── run_api.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. 项目原理简介

本项目的整体流程如下：

```text
图片数据集 + 文本标签
        ↓
文本分词与词表构建
        ↓
训练 Conditional VAE
        ↓
保存模型权重 checkpoint
        ↓
输入 prompt
        ↓
随机采样隐变量 z
        ↓
Decoder 生成图片
```

模型结构可以理解为：

```text
输入图像 x + 文本条件 c
        ↓
Encoder
        ↓
mu, logvar
        ↓
重参数化采样
        ↓
隐变量 z
        ↓
Decoder + 文本条件 c
        ↓
重建图像 / 生成图像
```

VAE 的训练损失为：

```text
Loss = Reconstruction Loss + beta × KL Loss
```

其中：

* `Reconstruction Loss`：让重建图像尽量接近原图
* `KL Loss`：让隐变量分布接近标准正态分布
* `beta`：控制 KL 约束强度

---

## 4. 环境安装

### 4.1 创建虚拟环境

#### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / WSL / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 4.2 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果需要使用 NVIDIA GPU 训练，需要安装支持 CUDA 的 PyTorch。

例如 CUDA 12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出：

```text
True
```

说明当前环境可以使用 GPU。

---

## 5. 数据集准备

项目支持两种数据格式。

---

### 5.1 使用文件夹名作为 prompt

这是最简单的方式。

```text
datasets/my_images/
├── cat/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
├── dog/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
└── flower/
    ├── 001.jpg
    ├── 002.jpg
    └── 003.jpg
```

在这种情况下，父文件夹名会自动作为提示词。

例如：

```text
datasets/my_images/cat/001.jpg
```

对应 prompt 为：

```text
cat
```

再例如：

```text
datasets/my_images/flower/001.jpg
```

对应 prompt 为：

```text
flower
```

---

### 5.2 使用 `.txt` 文件作为 prompt

也可以为每张图片提供一个同名 `.txt` 文件。

```text
datasets/my_images/
├── cat_001.jpg
├── cat_001.txt
├── flower_001.jpg
└── flower_001.txt
```

例如 `cat_001.txt` 内容为：

```text
a black cat sitting on grass
```

这种方式比单纯使用文件夹名更细致，但由于本项目不是 CLIP 文本编码器，复杂语义理解能力仍然有限。

---

## 6. 训练模型

训练前请确保当前命令行位于项目根目录。

### Windows CMD

```cmd
cd /d D:\path\to\vae_prompt_project_v2
.venv\Scripts\activate.bat
set PYTHONPATH=%cd%
```

### Linux / WSL / macOS

```bash
cd /path/to/vae_prompt_project_v2
source .venv/bin/activate
export PYTHONPATH=$(pwd)
```

---

### 6.1 基础训练命令

Windows CMD：

```cmd
python scripts\train_vae.py --data_dir datasets\my_images --out_dir outputs\vae\default_run --image_size 64 --epochs 50 --batch_size 32 --latent_dim 128 --beta 0.001
```

Linux / WSL / macOS：

```bash
python scripts/train_vae.py \
  --data_dir datasets/my_images \
  --out_dir outputs/vae/default_run \
  --image_size 64 \
  --epochs 50 \
  --batch_size 32 \
  --latent_dim 128 \
  --beta 0.001
```

---

### 6.2 快速测试训练

```cmd
python scripts\train_vae.py --data_dir datasets\my_images --out_dir outputs\vae\test_run --image_size 64 --epochs 20 --batch_size 16 --latent_dim 128 --beta 0.001
```

---

### 6.3 更高质量的训练参数

如果希望重建效果更清晰，可以尝试：

```cmd
python scripts\train_vae.py --data_dir datasets\my_images --out_dir outputs\vae\better_run --image_size 128 --epochs 100 --batch_size 16 --latent_dim 256 --beta 0.0001
```

参数说明：

| 参数             | 含义         |
| -------------- | ---------- |
| `--data_dir`   | 训练图片目录     |
| `--out_dir`    | 训练结果保存目录   |
| `--image_size` | 输入图片尺寸     |
| `--epochs`     | 训练轮数       |
| `--batch_size` | 批大小        |
| `--latent_dim` | 隐变量维度      |
| `--beta`       | KL loss 权重 |

---

## 7. 生成图片

训练完成后，可以使用保存的 checkpoint 进行生成。

```cmd
python scripts\generate.py --ckpt outputs\vae\default_run\checkpoints\best.pt --prompt "cat" --num_images 8 --out_dir outputs\generated\cat
```

例如生成花朵：

```cmd
python scripts\generate.py --ckpt outputs\vae\default_run\checkpoints\best.pt --prompt "flower" --num_images 8 --out_dir outputs\generated\flower
```

生成结果会保存在：

```text
outputs/generated/flower/
├── sample_000.png
├── sample_001.png
├── sample_002.png
└── grid.png
```

其中：

* `sample_xxx.png`：单张生成图片
* `grid.png`：多张生成图片拼接图

---

## 8. 图像重建

可以输入一张真实图片，让 VAE 进行重建，用来判断模型是否学到了有效特征。

```cmd
python scripts\reconstruct.py --ckpt outputs\vae\default_run\checkpoints\best.pt --image_path datasets\my_images\cat\001.jpg --prompt "cat" --out_dir outputs\generated\reconstruct
```

输出结果会保存在：

```text
outputs/generated/reconstruct/
```

---

## 9. 启动 Gradio 网页界面

```cmd
python webui\app_gradio.py --ckpt outputs\vae\default_run\checkpoints\best.pt
```

然后在浏览器打开：

```text
http://127.0.0.1:7860
```

网页中可以设置：

* prompt
* 生成图片数量
* 随机种子 seed
* temperature
* 输出目录

---

## 10. 启动 FastAPI 后端

```cmd
python run_api.py
```

然后在浏览器打开：

```text
http://127.0.0.1:8000/docs
```

可以看到自动生成的 API 文档。

主要接口包括：

```text
GET  /api/health
POST /api/upload_dataset
POST /api/train
POST /api/generate
POST /api/reconstruct
```

---

## 11. 主要代码说明

### `app/models/conditional_vae.py`

定义 Conditional VAE 模型。

主要包含：

* `TextConditionEncoder`
* `ConditionalVAE`
* `vae_loss`

该文件负责神经网络结构，包括图像 Encoder、文本条件编码器、重参数化采样和 Decoder。

---

### `app/utils/dataset.py`

负责数据集读取。

主要功能：

* 查找图片文件
* 读取图片
* 调整图片尺寸
* 读取 `.txt` caption
* 使用父文件夹名作为默认 caption
* 返回图像 tensor 和文本 token id

---

### `app/utils/text.py`

负责文本处理。

主要功能：

* prompt 分词
* 建立词表
* 将 prompt 编码成 token id
* 保存词表
* 加载词表

---

### `app/utils/image_io.py`

负责图像保存和随机种子设置。

主要功能：

* 创建目录
* 固定随机种子
* 保存图片网格

---

### `app/services/training_service.py`

负责完整训练流程。

主要功能：

* 收集 caption
* 构建词表
* 创建 Dataset
* 创建 DataLoader
* 初始化模型
* 计算 VAE loss
* 执行反向传播
* 保存 checkpoint
* 保存训练过程中的重建样例

---

### `app/services/generation_service.py`

负责图像生成和图像重建。

主要功能：

* 加载 checkpoint
* 编码 prompt
* 随机采样 latent vector
* 使用 Decoder 生成图片
* 保存生成结果

---

### `app/services/checkpoint_service.py`

负责模型权重加载。

主要功能：

* 读取 `.pt` 文件
* 恢复模型配置
* 恢复词表
* 恢复模型参数

---

### `app/api/routes.py`

定义 FastAPI 接口。

主要功能：

* 健康检查
* 上传数据
* 训练模型
* 生成图片
* 重建图片

---

### `scripts/train_vae.py`

命令行训练入口。

运行示例：

```cmd
python scripts\train_vae.py --data_dir datasets\my_images --out_dir outputs\vae\default_run --image_size 64 --epochs 50 --batch_size 32
```

---

### `scripts/generate.py`

命令行生成入口。

运行示例：

```cmd
python scripts\generate.py --ckpt outputs\vae\default_run\checkpoints\best.pt --prompt "cat" --num_images 8 --out_dir outputs\generated\cat
```

---

### `scripts/reconstruct.py`

命令行重建入口。

运行示例：

```cmd
python scripts\reconstruct.py --ckpt outputs\vae\default_run\checkpoints\best.pt --image_path datasets\my_images\cat\001.jpg --prompt "cat" --out_dir outputs\generated\reconstruct
```

---

### `scripts/make_default_captions.py`

根据图片父文件夹名自动生成 `.txt` caption 文件。

运行示例：

```cmd
python scripts\make_default_captions.py --data_dir datasets\my_images
```

如果需要覆盖已有 `.txt` 文件：

```cmd
python scripts\make_default_captions.py --data_dir datasets\my_images --overwrite
```

---

### `webui/app_gradio.py`

Gradio 网页界面入口。

---

### `run_api.py`

FastAPI 后端启动入口。

---

## 12. 训练输出说明

训练后输出目录类似：

```text
outputs/vae/default_run/
├── vocab.json
├── samples/
│   ├── recon_epoch_0001.png
│   ├── recon_epoch_0010.png
│   └── ...
└── checkpoints/
    ├── best.pt
    ├── latest.pt
    └── vae_epoch_0010.pt
```

说明：

| 文件                  | 作用               |
| ------------------- | ---------------- |
| `vocab.json`        | 训练时构建的 prompt 词表 |
| `samples/`          | 训练过程中保存的重建效果图    |
| `best.pt`           | 验证集效果最好的模型       |
| `latest.pt`         | 最新保存的模型          |
| `vae_epoch_xxxx.pt` | 指定 epoch 的模型快照   |

---

## 13. 为什么生成图片比较模糊？

这是 VAE 的常见现象，不是代码错误。

主要原因包括：

1. VAE 通常使用 MSE 重建损失。
2. MSE 会让模型倾向于预测平均像素值。
3. 隐变量会压缩图像信息，细节容易丢失。
4. KL loss 会让 latent space 更平滑，但也会牺牲清晰度。
5. 本项目的文本编码器只是简单词表 embedding，不具备 CLIP 那样的语义理解能力。
6. 本项目模型规模较小，没有 Diffusion、Attention、GAN loss 等高级模块。

如果想提升效果，可以尝试：

```cmd
python scripts\train_vae.py --data_dir datasets\my_images --out_dir outputs\vae\better_run --image_size 128 --epochs 100 --batch_size 16 --latent_dim 256 --beta 0.0001
```

常见改进方向：

| 方法              | 作用               |
| --------------- | ---------------- |
| 增大 `image_size` | 提高输出分辨率          |
| 增大 `latent_dim` | 保留更多图像信息         |
| 降低 `beta`       | 减弱 KL 约束，提高重建清晰度 |
| 增加训练轮数          | 提高收敛程度           |
| 清洗数据集           | 减少噪声数据           |
| 使用更简单的数据        | 降低建模难度           |

---

## 14. 推荐数据集

适合本项目的入门数据集：

* CIFAR-10
* Oxford Flowers 102
* Oxford-IIIT Pet
* Fashion-MNIST
* 自定义猫狗图片
* 自定义花朵图片
* 简单图标数据
* 简单动漫头像数据

更适合 VAE 的数据：

```text
花朵
图标
简单物体
居中目标
简单卡通头像
低复杂度纹理图片
```

不太适合纯 VAE 入门的复杂数据：

```text
复杂街景
高清人脸
多物体场景
复杂自然图像
大量背景变化的图片
```

---

## 15. 常见问题

### 15.1 报错：`ModuleNotFoundError: No module named 'app'`

说明 Python 没有找到项目根目录。

在项目根目录执行：

Windows CMD：

```cmd
set PYTHONPATH=%cd%
```

Linux / WSL / macOS：

```bash
export PYTHONPATH=$(pwd)
```

然后重新运行训练或生成命令。

---

### 15.2 Windows CMD 中 `ls` 或 `which python` 不能用

Windows CMD 使用下面这些命令：

| Linux / WSL    | Windows CMD    |
| -------------- | -------------- |
| `ls`           | `dir`          |
| `pwd`          | `cd`           |
| `which python` | `where python` |
| `rm file`      | `del file`     |
| `clear`        | `cls`          |

---

### 15.3 CUDA 不可用

检查：

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出：

```text
False
```

说明当前 PyTorch 不是 CUDA 版本，或者显卡驱动 / CUDA 环境没有配置好。

---

### 15.4 生成图像太模糊

可以尝试：

```cmd
python scripts\train_vae.py --data_dir datasets\my_images --out_dir outputs\vae\better_run --image_size 128 --epochs 100 --batch_size 16 --latent_dim 256 --beta 0.0001
```

也可以先查看重建图：

```text
outputs/vae/default_run/samples/
```

如果重建图也很模糊，说明模型训练不充分或数据太复杂。

如果重建图还可以，但随机生成很差，说明 latent space 学习还不稳定。

---

## 16. GitHub 上传建议

建议不要上传以下内容：

```text
.venv/
outputs/
datasets/raw/
__pycache__/
*.pt
*.pth
*.ckpt
```

推荐 `.gitignore`：

```text
.venv/
__pycache__/
*.pyc
outputs/
datasets/raw/
*.pt
*.pth
*.ckpt
.DS_Store
```

如果想保留空目录，可以添加 `.gitkeep`：

```text
datasets/my_images/.gitkeep
outputs/.gitkeep
```

---

## 17. 项目局限

本项目主要用于学习 VAE 生成模型的基本原理和工程结构。

它不能像 Stable Diffusion 那样理解复杂自然语言，例如：

```text
a cat wearing sunglasses on Mars, cinematic lighting, ultra realistic
```

本项目更适合使用简单 prompt，例如：

```text
cat
dog
flower
red flower
anime face
```

如果想实现真正高质量的文生图，可以进一步学习：

* VQ-VAE
* VQGAN
* DDPM
* Latent Diffusion Model
* Stable Diffusion
* LoRA
* DreamBooth
* ControlNet

---

## 18. 后续可扩展方向

可以在本项目基础上继续扩展：

* 使用更深的 ResNet Encoder / Decoder
* 加入 U-Net skip connection
* 加入 perceptual loss
* 加入 GAN loss
* 改造成 VQ-VAE
* 改造成 VAE-GAN
* 加入 CLIP 文本编码器
* 改造成 Diffusion 模型
* 加入训练任务队列
* 增加前端上传数据集功能
* 增加模型管理页面

---

## 19. License

本项目仅用于学习、课程实验和生成模型入门研究。

你可以自由修改、扩展和使用本项目。

---

## 20. Acknowledgement

本项目基于 Variational Autoencoder 和 Conditional Generative Model 的基本思想实现。

项目目标不是追求高清图像生成效果，而是帮助理解一个图像生成系统从数据准备、模型训练、checkpoint 保存、推理生成、API 服务到 Web UI 的完整工程流程。
