# wav-jepa-knn

WavJEPA 임베딩을 추출한 뒤, cosine kNN 분류로 성능(Accuracy/F1)을 측정하는 간단한 평가 스크립트입니다.

이 버전은 **실행 시 HuggingFace 리모트를 참조하지 않도록** 구성되어 있습니다.
즉, 모델 코드/설정/가중치를 미리 로컬에 복사해 둔 디렉토리만 사용합니다.

---

## 1) 데이터셋 포맷

`dataset_root` 아래에 다음 구조가 있어야 합니다.

```text
dataset_root/
  train/
    a.wav
    a.json
    b.wav
    b.json
    ...
  test/
    x.wav
    x.json
    ...
```

- 각 `.wav`와 같은 basename의 `.json` 라벨 파일이 있어야 합니다.
- 라벨 키는 `label`, `labels`, `class`, `target` 순으로 우선 탐색합니다.

---

## 2) 로컬 모델 디렉토리 준비

`--model`로 넘길 경로에 아래 파일들이 있어야 합니다(예시).

```text
/path/to/local_wavjepa_model/
  config.json
  model.py                 # HF remote code를 로컬로 복사한 파일
  feature_extractor.py     # HF remote code를 로컬로 복사한 파일
  preprocessor_config.json
  model.safetensors        # 또는 sharded safetensors
```

> 핵심: `model.safetensors`만 있는 것이 아니라, 해당 가중치를 읽을 모델 코드(`model.py` 등)도 로컬에 있어야 합니다.

---

## 3) 설치

```bash
pip install -r requirements.txt
```

---

## 4) 실행 (오프라인/로컬 전용)

```bash
python knn_eval.py \
  --dataset_root /path/to/dataset \
  --model /path/to/local_wavjepa_model \
  --k 20
```

- 스크립트 내부에서 `local_files_only=True`와 `TRANSFORMERS_OFFLINE=1`을 사용하므로 원격 HF 조회를 피합니다.

---

## 5) 출력 예시

```text
test/acc: 0.8123
test/f1_macro: 0.7988
```

---

## 6) 참고

- 임베딩은 마지막 hidden state를 mean pooling 해서 사용합니다.
- 분류기는 `KNeighborsClassifier(metric="cosine")`를 사용합니다.
