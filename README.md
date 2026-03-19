# Image Captioning with MobileCLIP + sLLM

MobileCLIP 이미지 인코더와 Small Language Model(SmolLM2)을 활용한 이미지 캡션 생성 모델입니다.  
2단계 학습(Embedding Alignment → Caption Fine-tuning) 방식으로 구성되어 있습니다.

## 환경 요구 사항

- Python: 3.12.x 권장
- 컴퓨팅 가속: 
  - 학습(Step 1, Step 2): GPU 권장 (CUDA, bf16 mixed precision 사용)
  - 추론(infer, eval): CPU에서도 동작하나, GPU 사용 시 속도 향상

## 설치

### 1. 코드 

```
git clone https://github.com/inst-mmismin/image_caption
cd image_caption
```

### 2. 가상환경 생성 

```
conda create -n my_env python=3.12
conda activate my_env
```

### 3. 의존성 설치

```
pip install -r requirements.txt
```

### 4. 환경 설정 스크립트 실행

`setup.sh`는 다음을 수행합니다:
- ml-mobileclip 클론 및 open_clip 설정
- MobileCLIP-B, SmolLM2-135M-Instruct 모델 다운로드
- Flickr8k, Flickr30k 데이터셋 다운로드 및 분할

```
 setup.sh
```

> 참고: `setup.sh` 실행 중 `mobileclip2` 관련 오류가 발생하면, [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip) 저장소 구조를 확인하세요.

## 실행 방법

### Step 1 학습: Embedding Alignment

CLIP 이미지 특징을 LLM 임베딩 공간에 정렬합니다.

```
python step1.py --epochs 50 \
                --learning_rate 0.01 \
                --min_lr 0.0001 \
                --batch_size 512 \
                --use_contrastive \
                --weight_contrastive 1 \
                --contra_temp 0.05 \
                --proj_type linear \
                --use_layer_norm 
```

- 결과 저장 : `runs/step1/YYYYMMDD_HHMMSS/checkpoints/projection_best.pt`

### Step 2 학습: Caption Fine-tuning

정렬된 projection과 LoRA로 캡션 생성 모델을 학습합니다.

```
python step2.py --projection_ckpt ./runs/step1/YYYYMMDD_HHMMSS/checkpoints/projection_best.pt \
                --batch_size 128 \
                --epochs 50 \
                --projection_lr 2e-3 \
                --lora_lr 1e-3 
```

- 결과 저장: `runs/step2/YYYYMMDD_HHMMSS/checkpoints/`

### 추론 (infer)

단일 이미지에 대한 캡션 생성:

```
python infer.py --ckpt_dir runs/step2/YYYYMMDD_HHMMSS \
                --image path/to/image.jpg
```

### 평가 (eval)

Flickr30k 데이터셋(Flickr8k train 제외)으로 CIDEr 평가:

```
python eval.py --ckpt_dir runs/step2/YYYYMMDD_HHMMSS
```

## 프로젝트 구조

```
├── step1.py              # Step 1 학습
├── step2.py               # Step 2 학습
├── infer.py               # 단일 이미지 추론
├── eval.py                # Flickr30k CIDEr 평가
├── setup.sh               # 환경 설정 스크립트
├── env.py                 # 경로 및 상수
├── module/
│   ├── projection.py     # Linear/MLP Projection
│   ├── loss.py           # Contrastive, LM Loss
│   ├── flickr.py         # Flickr8k Dataset
│   ├── flickr30k.py      # Flickr30k Dataset (Flickr8k train 제외 과정 포함)
│   └── EvalDataset.py    # 평가용 Dataset 래퍼
├── utils/
│   ├── load.py           # 모델/데이터 등의 로드 함수 
│   ├── evaluate.py       # CIDEr, clean_caption
│   ├── step1_tools.py    # Step 1 학습 유틸
│   └── step2_tools.py    # Step 2 학습 유틸
├── misc/
│   ├── split_flickr8k.py # Flickr8k train/val 분할
│   └── **_test.py        # 모듈 실행 테스트 파일 
├── checkpoints/          # CLIP, LLM 모델 (setup.sh로 다운로드)
├── dataset/              # Flickr8k, Flickr30k (setup.sh로 다운로드)
├── runs/                 # 학습 로그 및 체크포인트
└── figs/                 # 리포트에서 사용한 이미지 파일 
```
