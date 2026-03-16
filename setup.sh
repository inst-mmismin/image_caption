## mobileCLIP-B 

# mobileclip git clone 
if [ ! -d "ml-mobileclip" ]; then
    git clone https://github.com/apple/ml-mobileclip.git
fi

# open-clip 
cd ml-mobileclip
git clone https://github.com/mlfoundations/open_clip.git
pushd open_clip
git apply ../mobileclip2/open_clip_inference_only.patch
cp -r ../mobileclip2/* ./src/open_clip/
pip install -e .
popd

pip install git+https://github.com/huggingface/pytorch-image-models
cd .. 

# 만약 hf가 없으면 아래를 실행해서 설치
if ! command -v hf &> /dev/null; then
    pip install -U "huggingface_hub"
fi

# mobileCLIP-B 모델 다운로드
if [ ! -f "./checkpoints/clip/mobileclip_b.pt" ]; then
    hf download apple/MobileCLIP-B --local-dir ./checkpoints/clip
fi

# HuggingFaceTB/SmolLM2-135M-Instruct 모델 다운로드 
if [ ! -f "./checkpoints/llm/SmolLM2-135M-Instruct.pt" ]; then
    hf download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir ./checkpoints/llm
fi


# 만약 gdowndl 없으면 아래를 실행해서 설치
if ! command -v gdown &> /dev/null; then
    pip install -U "gdown"
fi

# 만약 dataset 폴더가 없으면 생성
if [ ! -d "./dataset" ]; then
    mkdir -p ./dataset
fi

# Flickr8k 데이터셋 다운로드 
if [ ! -f "./dataset/flickr8k" ]; then
    gdown 1a5j4v8S1y7yPoX9m7XOSSyTzrQzyTlg_ -O ./dataset/flickr8k.zip
    unzip -qq ./dataset/flickr8k.zip -d ./dataset/flickr8k
    rm ./dataset/flickr8k.zip
fi


pip install -r requirements.txt
python misc/split_flickr8k.py 