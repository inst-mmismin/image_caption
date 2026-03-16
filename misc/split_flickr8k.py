"""
Flickr8k 데이터를 9:1 비율로 train/val 분할.
이미지 이름이 적힌 txt 파일을 생성 (train.txt, val.txt, all.txt).
"""
import os
import random


def split_flickr8k(root, output_dir, ratio, seed):
    """
    Args:
        root: Flickr8k 데이터 루트 (Images/, captions.txt 포함)
        output_dir: txt 파일을 저장할 폴더 경로
        ratio: 학습 비율 (기본 0.9 = 9:1)
        seed: 랜덤 시드
    """
    images_dir = os.path.join(root, "Images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images 폴더를 찾을 수 없습니다: {images_dir}")

    images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not images:
        raise ValueError(f"이미지가 없습니다: {images_dir}")

    random.seed(seed)
    random.shuffle(images)

    n_train = int(len(images) * ratio)
    train_images = images[:n_train]
    val_images = images[n_train:]

    os.makedirs(output_dir, exist_ok=True)

    def write_txt(path: str, names: list):
        with open(path, "w") as f:
            for name in names:
                f.write(name + "\n")

    write_txt(os.path.join(output_dir, "train.txt"), train_images)
    write_txt(os.path.join(output_dir, "val.txt"), val_images)
    write_txt(os.path.join(output_dir, "all.txt"), images)

    print(f"분할 완료: {output_dir}")
    print(f"  전체: {len(images)}개")
    print(f"  train: {len(train_images)}개 ({ratio*100:.0f}%)")
    print(f"  val:   {len(val_images)}개 ({(1-ratio)*100:.0f}%)")


def main():
    split_flickr8k(
        root = "./dataset/flickr8k",
        output_dir = "./dataset/flickr8k/splits",
        ratio = 0.9,
        seed = 42
        )


if __name__ == "__main__":
    main()
