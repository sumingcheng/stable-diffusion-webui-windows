# SD 简介

## stable-diffusion

1. 潜空间扩散模型（LDM）latent diffusion model，其实就是一种模型,扩散模型。
2. Stable Diffusion是通过与Stability AI和Runway的合作实现的，它基于先前的工作，主要是关于高分辨率图像合成与潜在扩散模型

## 两个开源项目

1. https://github.com/CompVis/stable-diffusion
2. https://github.com/AUTOMATIC1111/stable-diffusion-webui

### 衍生GUI

1. stable-diffusion-WebUI v1.5
2. stable-diffusion-ComfyUI v1.5
3. stable-diffusion-Fooocus v1.0

## 大模型：核心

https://huggingface.co/runwayml/stable-diffusion-v1-5

1. ckpt、safetensors（安全）
2. 文生图：在模型中寻找输入的特征

## VAE：潜在扩散模型

VAE是一种深度学习模型，它能够学习如何有效地对数据进行压缩（编码）和恢复（解码）。它的特点是在编码过程中，不是直接压缩为一个确定的值，而是压缩为一个概率分布（通常是高斯分布）。这使得VAE在解码时可以产生多种可能的输出，这种多样性使其在生成任务上表现出色。

本质是作为滤镜或者是调色的作用，更改风格4

## LoRa：低秩适应模型

1. 使用loar可以生成特定的衣服特定的风格，根据需要生成不同于底模(大模型)的图片，体积小，效果好

2. 消耗小：2060:6GB即可

3. 步数：image*repeat*epoch/bathc_size = total steps

   | image      | 素材     | 原图质量越高，模型质量越好,但也不要过 |
   | ---------- | -------- | ------------------------------------- |
   | Repeat     | 学习次数 | 次数越多，效果越好,过多会过拟合       |
   | Epoch      | 循环     | 循环越多，效果越好,过多会过拟合       |
   | Batch size | 并行数量 | 越高，越快。过高收敛效果会差          |

4. 速率/质量

   | Unet_lr           | 学习率，使用时候覆盖lr | 学习率越高，速度越快 | 1e-4                  |
   | ----------------- | ---------------------- | -------------------- | --------------------- |
   | Learning rete     | 学习率                 | 同上                 | 1e-4                  |
   | Text_encoder_lr   | 学习率                 | 同上                 | 5e-5                  |
   | Network Dimension | 学习精度               | 精度越高，细节越好   | 128,输出文件大小140MB |
   | optimazer         | 优化器                 |                      | 1e-4                  |

## 做模型流程

1. 制作素材https://www.birme.net/
2. 打tag：提示词反推，针对提示词进行优化
3. 批素材生成模型
4. 使用模型，调整参数，再次训练
5. 满意的模型

| tag 类型 | 内容                       | 说明 |
| -------- | -------------------------- | ---- |
| 主题     | 1 boy 1girl                | 主体 |
| 动作     | sitting，looking at viewer |      |
| 人物特征 | short hair                 |      |
| 视角     | upper body                 |      |
| 光影     | night，absuerdres：1.2     |      |

## 训练设置

Lora训练总步数：1500~6000

1. 二次元：10~16 step
2. 写实：17~35 step
3. 场景：50起步 step

## 做图流程

1. 选择合适的大模型和其他模型
2. 填写正向、反向关键词
3. 使用随机种子抽卡，4图
4. 根据效果调整关键词，直到选中心怡图片，锁定种子
5. 根据选中图进行图生图，二次优化、三次优化、局部优化、放大或者高清修复等等
6. 完成图，将图进行放大或者拼接
7. 出图

## 功能

### midjourney

1. 4图选择抽卡
2. 上传云端进行图生图
3. 国外数据敏感

### SD

1. 功能强大有很多插件和配置项，且数据本地化

### demo

1. midjourney的交互方式生图功能：关键词->文生图->默认大部分参数->选图->图生图->高清优化->放大+tag
2. 针对建筑、室内、环境的模型：风格图和模型进行匹配，前端进行选择。也就是匹配不同的loar
3. 指定区域重绘：可以
4. 图库浏览：图存到了SD/outputs，要改存到服务端