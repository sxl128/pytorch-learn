
# 训练过程（更准确版本）
# 1）前向过程：加噪
# 从一张干净图片 x_0 开始，按照预设规则不断加入噪声，得到：
# x_0 -> x_1 -> x_2 -> ... -> x_t
# 2）学习逆过程：预测噪声
# 训练时，随机取某个噪声步 t，把：
# - 带噪图片 x_t
# - 当前时间步 t
# - 文本描述
# 一起输入模型。
# 模型的目标通常不是直接输出干净图，而是：
# 预测当前图片中包含的噪声

# 再通过损失函数让模型学会在不同噪声程度下都能正确预测噪声，从而学会逆转加噪过程。
# 生成过程（Text-to-Image）
# 1. 从纯噪声开始
# 2. 输入文本描述
# 3. 模型每一步预测噪声并去掉一点
# 4. 重复多步后，逐渐生成符合文本描述的图片
# Inpainting 过程（局部编辑）
# 1. 输入原图 + mask + 文本描述
# 2. mask 指定哪些区域允许修改
# 3. 模型在保留整体布局和上下文的基础上，主要对 mask 内区域进行重新生成
# 4. 最终输出修改后的图片，mask 外区域基本保持原图
# 比如：
# - 原图里是一匹马
# - mask 把马框出来
# - prompt 写 "a zebra"
# 那么模型会参考原图的姿态、位置、背景、光照，把马所在区域重新生成成一匹斑马。


from diffusers import StableDiffusionInpaintPipeline
import torch
import PIL.Image as Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# pipe, bundles all the parts needed for generation
# - tokenizer & text encoder for the text prompt
# - a UNet denoiser
# - a variational autoencoder (VAE)
# compresses the image into a compact internal representation and then 
# reconstruct it 

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16
).to(device)

# print(pipe)

img = Image.open('./data/horse.jpg')
# img.show()

mask_img = Image.open('./data/horse_mask.jpg')
# mask_img.show()

prompt = ("a zebra replacing the original horse, "
          "same pose, same lighting, background unchanged")
negative = "distorted background, blurry, text, watermark"

out = pipe(
    prompt=prompt,
    image=img,
    mask_image=mask_img,
    negative_prompt=negative,
    guidance_scale=7.5, # 提示词服从度 CFG。1=不听prompt，7.5=常用平衡值，>12 会过饱和、死板
    strength=0.8, # 重绘幅度 0~1。本质是给Mask内加多少噪再去噪。0=完全不改，1=完全不参考原图Mask内像素重画，0.8算改动很大，保留很少原结构
    generator=torch.Generator(device).manual_seed(42) # 固定42保证每次结果可复现，改数字就换结果
)

out.images[0].show()
