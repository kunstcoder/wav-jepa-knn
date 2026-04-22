# wav-jepa-knn

WavJEPA 임베딩을 추출한 뒤, cosine kNN 분류로 성능(Accuracy/F1)을 측정하는 간단한 평가 스크립트입니다.

현재 구현은 **HuggingFace 인터페이스 기준**으로 동작합니다.

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

## 2) 설치

```bash
pip install -r requirements.txt
```

---

## 3) 실행

```bash
python knn_eval.py \
  --dataset_root /path/to/dataset \
  --model labhamlet/wavjepa-base \
  --k 20
```

- `--model`: HuggingFace 모델 ID 또는 로컬 HF 모델 디렉토리
- `--device`: 기본값은 CUDA 사용 가능 시 `cuda`, 아니면 `cpu`

---

## 4) 출력 예시

```text
test/acc: 0.8123
test/f1_macro: 0.7988
```

---

## 5) 참고

- 임베딩은 마지막 hidden state를 mean pooling 해서 사용합니다.
- 분류기는 `KNeighborsClassifier(metric="cosine")`를 사용합니다.
