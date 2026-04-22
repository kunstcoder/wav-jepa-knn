# wav-jepa-knn

WavJEPA 임베딩을 추출한 뒤, cosine kNN 분류로 성능(Accuracy/F1)을 측정하는 간단한 평가 스크립트입니다.

이번 버전은 **HF remote code를 실행 시 내려받지 않고**, 저장소 내부의 클래스(`local_wavjepa.py`)로 모델을 구성해 로컬 `safetensors`를 로드합니다.

---

## 1) 데이터셋 포맷

```text
dataset_root/
  train/
    a.wav
    a.json
    ...
  test/
    x.wav
    x.json
    ...
```

---

## 2) 로컬 모델 디렉토리 준비

`--model` 경로에는 최소한 아래 파일이 필요합니다.

```text
/path/to/local_wavjepa_model/
  config.json
  model.safetensors   # 또는 *.safetensors
```

> 참고: 모델 구조는 `local_wavjepa.py`의 `WavJEPA`, `ConvFeatureExtractor`, `WavJEPAFeatureExtractor` 클래스로 내부 구현되어 있습니다.

---

## 3) 설치

```bash
pip install -r requirements.txt
```

---

## 4) 실행

```bash
python knn_eval.py \
  --dataset_root /path/to/dataset \
  --model /path/to/local_wavjepa_model \
  --k 20
```

실행 시 모델 로드 정보(`missing/unexpected`)가 먼저 출력됩니다.

---

## 5) 출력 예시

```text
model load info: missing=12, unexpected=0
test/acc: 0.8123
test/f1_macro: 0.7988
```

---

## 6) 주의사항

- 내부 클래스 구현은 HF custom code를 단순화해 옮긴 버전입니다.
- `config.json`/`safetensors`의 구조 차이가 큰 경우 로딩이 실패할 수 있습니다.


## 7) 자주 발생하는 오류

### `WavJEPA 형태의 가중치로 보이지 않습니다. state_dict key 예시: ...`

이 메시지는 보통 아래 경우입니다.

- `model.safetensors.index.json`이 있는데 shard 파일 일부가 없는 경우
- state_dict key에 `model.`/`module.` 같은 prefix가 붙은 체크포인트를 로더가 제대로 정규화하지 못한 경우
- `config.json`과 safetensors가 서로 다른 모델 버전인 경우

현재 로더는 shard index(`model.safetensors.index.json`)와 prefix 제거(`model.`, `module.`, `wavjepa.` 등)를 지원합니다.
그래도 실패하면, 출력된 key 예시를 기준으로 실제 prefix 패턴을 추가해야 합니다.
