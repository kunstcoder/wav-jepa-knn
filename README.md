# wav-jepa-knn

로컬 WavJEPA 체크포인트(`.ckpt`)를 사용해 임베딩을 추출하고, 간단한 cosine kNN 분류로 성능(Acc/F1)을 측정하는 스크립트입니다.

요청사항에 맞게 **HuggingFace 의존이 필수는 아니며**, 기본 동작은 `wavjepa` GitHub 코드(HEAR runtime 로딩 경로)를 참조합니다.

- 기본 백엔드: `wavjepa` (로컬 github repo + 로컬 ckpt)
- 호환 백엔드: `hf` (기존 방식 유지)

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

- 각 `.wav` 파일과 같은 basename의 `.json` 라벨 파일이 필요합니다.
- 라벨 키는 `label`, `labels`, `class`, `target` 등을 우선 탐색합니다.

---

## 2) 설치

```bash
pip install -r requirements.txt
```

---

## 3) 권장 실행: 로컬 ckpt + wavjepa github 코드

먼저 `wavjepa` 저장소를 로컬에 준비합니다(예: `/path/to/wavjepa`).

그 다음 아래처럼 실행합니다.

```bash
python knn_eval.py \
  --dataset_root /path/to/dataset \
  --backend wavjepa \
  --wavjepa_repo /path/to/wavjepa \
  --ckpt_path /path/to/checkpoint.ckpt \
  --k 20
```

### 동작 개요

`--backend wavjepa`일 때:

1. `--wavjepa_repo`를 `sys.path`에 추가
2. `hear_configs.WavJEPA.load_model(ckpt_path)` 호출
3. 로드된 runtime의 `get_scene_embeddings()`로 파일별 임베딩 추출
4. train/test 임베딩으로 cosine kNN 평가

이는 `wavjepa` GitHub의 HEAR runtime 사용 흐름을 기준으로 구성했습니다.

---

## 4) 선택 실행: HuggingFace 백엔드(호환)

```bash
python knn_eval.py \
  --dataset_root /path/to/dataset \
  --backend hf \
  --model labhamlet/wavjepa-base \
  --k 20
```

---

## 5) 출력 예시

```text
test/acc: 0.8123
test/f1_macro: 0.7988
```

---

## 6) 트러블슈팅

- `backend=wavjepa`인데 `--wavjepa_repo` 또는 `--ckpt_path`가 없으면 오류가 발생합니다.
- `wavjepa` 코드/ckpt의 구조가 크게 다른 경우, 해당 리포 버전에 맞춰 로딩 부분(`load_wavjepa_runtime`)을 조정해야 할 수 있습니다.
