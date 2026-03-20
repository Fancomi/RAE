"""
从 ImageNet validation ImageFolder 目录生成 reference npz。
格式：{"arr_0": np.ndarray [N, H, W, C] uint8}

用法：
  python make_ref_npz.py \
      --val-dir /root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-1k/validation \
      --out     /root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-1k/val_256_ref.npz \
      --image-size 256
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy: cy + image_size, cx: cx + image_size])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=str,
                        default="/root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-1k/validation")
    parser.add_argument("--out", type=str,
                        default="/root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-1k/val_256_ref.npz")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=None, help="截取前 N 张，默认全部")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.Lambda(lambda img: img.convert("RGB")),
    ])
    ds = ImageFolder(args.val_dir, transform=transform)
    n = len(ds) if args.num_samples is None else min(args.num_samples, len(ds))
    print(f"处理 {n} 张图片 ({args.image_size}×{args.image_size}) → {args.out}")

    images = []
    for i in tqdm(range(n)):
        img, _ = ds[i]
        images.append(np.array(img, dtype=np.uint8))

    arr = np.stack(images)   # [N, H, W, C] uint8
    print(f"保存 shape={arr.shape}, dtype={arr.dtype}")
    np.savez(args.out, arr_0=arr)
    print("完成！")


if __name__ == "__main__":
    main()
