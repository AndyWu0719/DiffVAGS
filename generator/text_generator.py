import os
import argparse
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

def load_model(model_name="Salesforce/blip-image-captioning-large"):
    # 加载 BLIP 模型及对应处理器
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    processor = BlipProcessor.from_pretrained(model_name)
    return model, processor

def generate_caption(image, model, processor, device, temperature=0.8):
    # 图像读取与处理
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    # 直接只传入图像让模型生成描述
    inputs = processor(images=image, return_tensors="pt")
    # 将所有 tensor 移到指定 device 上
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成描述时采用采样策略
    outputs = model.generate(**inputs,
                             max_length=64,
                             do_sample=True,
                             temperature=temperature,
                             top_p=0.92,
                             num_return_sequences=1)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model()
    model.to(device)
    model.eval()

    # 获取指定目录下所有图像
    image_files = sorted([os.path.join(args.image_dir, f) 
                          for f in os.listdir(args.image_dir)
                          if os.path.isfile(os.path.join(args.image_dir, f))
                          and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    
    results = {}
    for image_path in image_files:
        try:
            caption = generate_caption(image_path, model, processor, device, temperature=args.temperature)
            results[os.path.basename(image_path)] = caption
            print(f"Image: {os.path.basename(image_path)}\nCaption: {caption}\n")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "captions.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for img_name, cap in results.items():
            f.write(f"{img_name}: {cap}\n")
    print(f"Captions saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate English image descriptions using BLIP model with slight diversity in output."
    )
    parser.add_argument("--image_dir", type=str, default="/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/images",
                        help="Directory containing images")
    parser.add_argument("--out_dir", type=str, default="/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/texts",
                        help="Directory to save generated captions")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature, lower values yield more consistent output.")
    args = parser.parse_args()
    main(args)