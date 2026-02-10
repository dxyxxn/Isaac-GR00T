# GR00T N1.6 VLA 모델 구현 상세 분석 보고서

> **NVIDIA Isaac GR00T N1.6** — Vision-Language-Action (VLA) 모델의 아키텍처, 데이터 흐름, 학습 로직에 대한 심층 기술 분석

---

## 목차

1. [프로젝트 오버뷰 & 구조](#1-프로젝트-오버뷰--구조)
2. [모델 아키텍처 상세 분석](#2-모델-아키텍처-상세-분석)
3. [데이터 흐름 및 텐서 쉐이프](#3-데이터-흐름-및-텐서-쉐이프)
4. [학습 로직](#4-학습-로직)
5. [핵심 코드 스니펫 요약](#5-핵심-코드-스니펫-요약)

---

## 1. 프로젝트 오버뷰 & 구조

### 1.1 핵심 목적

GR00T N1.6은 **범용 휴머노이드 로봇 스킬**을 위한 오픈 VLA(Vision-Language-Action) 모델입니다. 이 모델은 다음과 같은 과정을 통해 로봇을 제어합니다:

1. **이미지(Vision)** 관측과 **자연어(Language)** 명령을 입력으로 받음
2. Vision-Language Model (VLM)을 통해 멀티모달 이해를 수행
3. **Diffusion Transformer (DiT)** 기반 Action Head에서 **Flow Matching** 기법으로 연속적 로봇 액션을 디노이징(denoise)하여 생성

### 1.2 주요 폴더 구조

```
Isaac-GR00T/
├── gr00t/
│   ├── configs/                    # 설정(Config) 모듈
│   │   ├── base_config.py          #   통합 Config 클래스 (Model + Data + Training)
│   │   ├── model/gr00t_n1d6.py     #   모델 하이퍼파라미터 (Gr00tN1d6Config)
│   │   ├── training/training_config.py  # 학습 하이퍼파라미터
│   │   └── data/data_config.py     #   데이터셋 설정
│   ├── model/                      # 모델 정의 모듈
│   │   ├── gr00t_n1d6/
│   │   │   ├── gr00t_n1d6.py       #   ★ 메인 모델 클래스 (Gr00tN1d6, Gr00tN1d6ActionHead)
│   │   │   ├── processing_gr00t_n1d6.py  # 데이터 전처리/후처리 프로세서
│   │   │   └── setup.py            #   모델 파이프라인 (Gr00tN1d6Pipeline)
│   │   ├── modules/
│   │   │   ├── eagle_backbone.py    #   ★ Vision-Language 백본 (EagleBackbone)
│   │   │   ├── dit.py              #   ★ Diffusion Transformer (DiT, AlternateVLDiT)
│   │   │   ├── embodiment_conditioned_mlp.py  # 멀티 에이전트 MLP 모듈
│   │   │   └── nvidia/Eagle-Block2A-2B-v2/    # Eagle VLM 원본 구현
│   │   └── base/model_pipeline.py  #   파이프라인 베이스 클래스
│   ├── data/                       # 데이터 로딩 모듈
│   │   ├── dataset/
│   │   │   ├── factory.py          #   데이터셋 팩토리
│   │   │   ├── sharded_single_step_dataset.py  # 단일 스텝 샤딩 데이터셋
│   │   │   └── lerobot_episode_loader.py       # LeRobot 에피소드 로더
│   │   ├── collator/collators.py   #   데이터 콜레이터
│   │   └── state_action/           #   상태/액션 정규화 처리
│   ├── experiment/                 # 학습 실험 관리
│   │   ├── experiment.py           #   ★ 학습 진입점 (run 함수)
│   │   ├── trainer.py              #   ★ 커스텀 Trainer (Gr00tTrainer)
│   │   ├── launch_finetune.py      #   파인튜닝 런처
│   │   └── launch_train.py         #   학습 런처
│   └── policy/
│       └── gr00t_policy.py         #   ★ 추론 정책 (Gr00tPolicy)
├── examples/                       # 다양한 로봇/환경 파인튜닝 예제
└── scripts/deployment/             # 배포/추론 스크립트
```

### 1.3 진입점(Entry Point) 파일

| 기능 | 파일 | 설명 |
|------|------|------|
| **학습 (Training)** | `gr00t/experiment/launch_finetune.py` | 파인튜닝 CLI 진입점. `tyro` 기반 설정 후 `experiment.py:run()` 호출 |
| **학습 루프** | `gr00t/experiment/experiment.py` | HuggingFace `TrainingArguments` + `Gr00tTrainer`로 학습 실행 |
| **추론 (Inference)** | `gr00t/policy/gr00t_policy.py` | `Gr00tPolicy.get_action()`으로 관측 -> 액션 생성 |
| **데이터 로딩** | `gr00t/data/dataset/sharded_single_step_dataset.py` | `ShardedSingleStepDataset`로 에피소드를 단일 스텝 단위로 샤딩 |

---

## 2. 모델 아키텍처 상세 분석

### 2.1 Core Class: `Gr00tN1d6`

**파일:** `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` (라인 411-539)

`Gr00tN1d6`는 HuggingFace `PreTrainedModel`을 상속하며, 전체 VLA 파이프라인의 최상위 모델 클래스입니다. 이 클래스는 두 개의 핵심 하위 모듈로 구성됩니다:

```
Gr00tN1d6 (PreTrainedModel)
├── backbone: EagleBackbone       # Vision-Language 인코더
├── action_head: Gr00tN1d6ActionHead  # Flow Matching Diffusion Policy
└── collator: Gr00tN1d6DataCollator   # 데이터 콜레이션
```

### 2.2 Component 1 — Vision-Language Backbone (`EagleBackbone`)

**파일:** `gr00t/model/modules/eagle_backbone.py`

Eagle은 NVIDIA의 Cosmos-Reason-2B VLM 변형으로, 내부적으로 다음 3개의 서브 모듈을 포함합니다:

| 서브 모듈 | 역할 | 구현 |
|-----------|------|------|
| `vision_model` | 이미지 인코딩 (SigLIP-2 기반) | `Siglip2VisionModel` (Flash Attention 2) |
| `mlp1` | Vision-to-Language 프로젝터 | `LayerNorm → Linear → GELU → Linear` (pixel unshuffle 후) |
| `language_model` | 텍스트 이해 및 멀티모달 융합 | `Qwen2ForCausalLM` (2B 파라미터급) |

**핵심 설계 결정:**

- **select_layer = 16**: LLM의 전체 레이어 중 16번째까지만 사용하고, 나머지는 제거합니다. 이는 액션 생성에 필요한 중간 표현만 추출하기 위함입니다 (라인 52-53):
  ```python
  while len(self.model.language_model.model.layers) > select_layer:
      self.model.language_model.model.layers.pop(-1)
  ```
- **tune_top_llm_layers = 4**: N1.6에서는 VLM의 상위 4개 LLM 레이어를 학습 가능하도록 설정합니다 (N1.5의 post-VLM 4-layer transformer adapter를 대체).

**Eagle VLM의 Modality Fusion (Vision + Language 융합):**

`modeling_eagle3_vl.py` (라인 175-250)에서 이미지와 텍스트가 융합되는 방식은 **Interleaved Token Replacement** 입니다:

1. 텍스트 토큰을 LLM의 입력 임베딩으로 변환: `input_embeds = language_model.get_input_embeddings()(input_ids)`
2. 이미지를 Vision Encoder로 처리 후 `mlp1` 프로젝터로 LLM 차원에 맞춤
3. `input_ids`에서 `image_token_index`에 해당하는 위치를 찾아 해당 토큰을 Vision 임베딩으로 교체:
   ```python
   selected = (input_ids == self.image_token_index)
   input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
   ```
4. 교체된 임베딩 시퀀스를 LLM에 통과시켜 hidden states 생성

**결과:** Backbone은 `[B, seq_len, 2048]` 형태의 멀티모달 특징(backbone_features)과 `[B, seq_len]` 어텐션 마스크를 출력합니다.

### 2.3 Component 2 — Action Head (`Gr00tN1d6ActionHead`)

**파일:** `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` (라인 19-402)

Action Head는 **Flow Matching** 기반 **Diffusion Policy**로, VLM의 출력을 조건(condition)으로 사용하여 연속적 로봇 액션 시퀀스를 생성합니다.

#### 주요 서브 모듈:

| 모듈 | 클래스 | 역할 |
|------|--------|------|
| `model` | `AlternateVLDiT` (기본) / `DiT` | 32-layer Diffusion Transformer |
| `state_encoder` | `CategorySpecificMLP` | 로봇 상태 → 임베딩 변환 (embodiment별 가중치) |
| `action_encoder` | `MultiEmbodimentActionEncoder` | 노이즈 액션 + 타임스텝 → 임베딩 (embodiment별 가중치) |
| `action_decoder` | `CategorySpecificMLP` | DiT 출력 → 액션 velocity 예측 (embodiment별 가중치) |
| `vlln` | `LayerNorm` | VLM 출력 정규화 |

#### Diffusion Transformer (`AlternateVLDiT`)

**파일:** `gr00t/model/modules/dit.py` (라인 289-364)

N1.6의 핵심 혁신으로, 32-layer DiT가 **self-attention**과 **cross-attention** 블록을 교대로 배치합니다:

- **홀수 블록 (idx % 2 == 1)**: Self-attention — state/action 토큰 간의 관계 학습
- **짝수 블록 (idx % 2 == 0)**: Cross-attention — VLM 출력(image/text 토큰)을 조건으로 사용
  - `idx % (2 * attend_text_every_n_blocks) == 0`이면 **텍스트 토큰**에 cross-attend
  - 그 외에는 **이미지 토큰**에 cross-attend

이 교대 패턴은 이미지와 텍스트 정보를 분리하여 처리함으로써 더 정교한 조건화를 가능하게 합니다.

각 `BasicTransformerBlock`은:
1. **AdaLayerNorm** (Adaptive Layer Norm): 타임스텝 임베딩으로 정규화 스케일/시프트 조정
2. **Cross/Self-Attention**: diffusers 라이브러리의 `Attention` 모듈
3. **Feed-Forward Network**: GEGLU 활성화 함수

#### 멀티 에이전트 MLP 모듈 (`CategorySpecificMLP` / `MultiEmbodimentActionEncoder`)

**파일:** `gr00t/model/modules/embodiment_conditioned_mlp.py`

단일 모델로 다양한 로봇(embodiment)을 지원하기 위해, **카테고리별 독립 가중치**를 사용합니다:

```python
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]  # [B, input_dim, hidden_dim]
        selected_b = self.b[cat_ids]  # [B, hidden_dim]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
```

`max_num_embodiments = 32`개까지의 서로 다른 로봇 형태를 지원하며, 각 로봇마다 고유한 상태/액션 인코딩/디코딩 가중치를 갖습니다.

### 2.4 아키텍처 구조도 요약

```
입력: [이미지들, 텍스트 명령, 로봇 상태]
          │
          ▼
┌─────────────────────────────┐
│   Eagle VLM (Backbone)       │
│  ┌──────────┐  ┌──────────┐ │
│  │ SigLIP-2 │→│  mlp1    │ │  Vision Encoder → Projector
│  │ Vision   │  │Projector │ │
│  └──────────┘  └────┬─────┘ │
│                     │        │
│  ┌──────────────────▼──────┐│
│  │ Qwen2 LLM (16 layers)  ││  텍스트+이미지 토큰 인터리빙
│  │  (상위 4 레이어 학습)     ││
│  └──────────┬──────────────┘│
└─────────────┼───────────────┘
              │ backbone_features [B, S, 2048]
              ▼
┌─────────────────────────────┐
│   Action Head                │
│  ┌──────────┐               │
│  │State Enc.│→ state_feat   │  CategorySpecificMLP
│  └──────────┘   [B,1,1536]  │
│  ┌──────────┐               │
│  │Action Enc│→ action_feat  │  MultiEmbodimentActionEncoder
│  └──────────┘ [B,16,1536]   │
│       │                      │
│  cat([state, action]) = sa  │  [B, 17, 1536]
│       │                      │
│  ┌──────────────────────┐   │
│  │ AlternateVLDiT       │   │  32-layer DiT
│  │ (Cross-attn to VLM   │   │  교대 self/cross attention
│  │  + Self-attn)         │   │
│  └───────────┬──────────┘   │
│              │               │
│  ┌──────────▼──────────┐    │
│  │  Action Decoder     │    │  CategorySpecificMLP
│  └──────────┬──────────┘    │
└─────────────┼───────────────┘
              │
              ▼
   예측 velocity → Euler 적분 → 최종 Action [B, 16, 29]
```

---

## 3. 데이터 흐름 및 텐서 쉐이프

### 3.1 학습 시 Forward Pass 단계별 추적

`Gr00tN1d6.forward()` (라인 496-513)을 기준으로 설명합니다. 배치 크기 `B`, 시퀀스 길이 `S`, 액션 호라이즌 `T=16`, 최대 액션 차원 `D=29`, hidden `H=1024`, 입력 임베딩 `E=1536`, backbone 임베딩 `C=2048`으로 가정합니다.

---

#### 단계 1: 입력 준비 (`prepare_input`)

```
입력 딕셔너리:
  - input_ids:    [B, S]          (토큰화된 텍스트 + 이미지 플레이스홀더)
  - attention_mask: [B, S]        (패딩 마스크)
  - pixel_values: [B*N_img, C_img, H_img, W_img]  (전처리된 이미지)
  - state:        [B, 1, 29]      (정규화된 로봇 상태, 패딩됨)
  - action:       [B, 16, 29]     (정규화된 액션 청크, 패딩됨)
  - action_mask:  [B, 16, 29]     (유효 액션 차원 마스크)
  - embodiment_id: [B]            (로봇 종류 ID, 정수)
```

#### 단계 2: Backbone Forward (`EagleBackbone.forward`)

**파일:** `eagle_backbone.py` 라인 105-120

```python
outputs = self.model(**vl_input, output_hidden_states=True)
outputs = outputs["hidden_states"][-1]  # 마지막 hidden state 추출
```

내부 흐름:
1. `input_ids` → LLM 임베딩 테이블 → `input_embeds: [B, S, 2048]`
2. `pixel_values` → SigLIP-2 Vision Encoder → `vit_embeds: [N_patches, vit_hidden]`
3. Pixel Unshuffle + `mlp1` 프로젝터 → `vit_embeds: [N_patches_down, 2048]`
4. `input_embeds`에서 이미지 토큰 위치를 찾아 `vit_embeds`로 교체
5. 교체된 임베딩을 Qwen2 LLM 16 레이어에 통과

```
출력:
  backbone_features:      [B, S, 2048]   (VL 멀티모달 특징)
  backbone_attention_mask: [B, S]         (유효 토큰 마스크)
  image_mask:             [B, S]          (이미지 토큰 위치)
```

#### 단계 3: VLM 출력 후처리 (`process_backbone_output`)

```python
backbone_features = self.vlln(backbone_features)  # LayerNorm
```

```
backbone_features: [B, S, 2048] → [B, S, 2048]  (정규화됨)
```

#### 단계 4: State 인코딩

```python
state_features = self.state_encoder(action_input.state, embodiment_id)
# CategorySpecificMLP: [B, 1, 29] → [B, 1, 1536]
```

- `state: [B, 1, 29]` → `CategorySpecificLinear(29→1024)` → ReLU → `CategorySpecificLinear(1024→1536)`
- 선택적 State Dropout (학습 시): 확률적으로 mask_token으로 교체
- 선택적 Gaussian Noise 추가

```
state_features: [B, 1, 1536]
```

#### 단계 5: Flow Matching 노이즈 생성 및 Action 인코딩

```python
# 노이즈 샘플링 및 보간
noise = torch.randn(actions.shape)           # [B, 16, 29]
t = Beta(1.5, 1.0).sample([B]) * 0.999      # [B] ∈ (0, 0.999)
noisy_trajectory = (1 - t) * noise + t * actions  # [B, 16, 29]
velocity = actions - noise                    # [B, 16, 29] (학습 목표)

# 타임스텝 이산화
t_discretized = (t * 1000).long()            # [B] ∈ {0, ..., 999}

# 액션 인코딩
action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
```

`MultiEmbodimentActionEncoder` 내부:
1. `W1`: `[B, 16, 29] → [B, 16, 1536]` (CategorySpecificLinear)
2. Sinusoidal Positional Encoding: `t_discretized: [B, 16] → tau_emb: [B, 16, 1536]`
3. Concat + `W2`: `[B, 16, 3072] → [B, 16, 1536]` (Swish 활성화)
4. `W3`: `[B, 16, 1536] → [B, 16, 1536]`

```
action_features: [B, 16, 1536]
```

#### 단계 6: 위치 임베딩 추가 (선택적)

```python
pos_ids = torch.arange(16)
pos_embs = self.position_embedding(pos_ids)  # [1, 16, 1536]
action_features = action_features + pos_embs
```

#### 단계 7: State + Action 결합 및 DiT Forward

```python
sa_embs = torch.cat((state_features, action_features), dim=1)
# [B, 1, 1536] + [B, 16, 1536] = [B, 17, 1536]
```

`AlternateVLDiT.forward()` (라인 299-364):
- `hidden_states = sa_embs: [B, 17, 1536]`
- `encoder_hidden_states = vl_embeds: [B, S, 2048]`
- 타임스텝 인코딩: `t_discretized → temb: [B, 1536]`
- 32개 Transformer 블록 순차 처리:
  - **짝수 블록**: Cross-attention (sa_embs가 query, vl_embeds가 key/value)
    - 이미지 토큰 또는 텍스트 토큰에 교대 attend
  - **홀수 블록**: Self-attention (sa_embs 내부)
  - 모든 블록: AdaLayerNorm(temb로 조건화) + FFN

```
DiT 출력 (proj_out_2 후): [B, 17, 1024]
```

#### 단계 8: Action 디코딩 및 Loss 계산

```python
pred = self.action_decoder(model_output, embodiment_id)
# CategorySpecificMLP: [B, 17, 1024] → [B, 17, 29]

pred_actions = pred[:, -16:]  # 마지막 16 스텝 (action 부분만)
# [B, 16, 29]

# MSE Loss (Flow Matching velocity 예측)
action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

### 3.2 추론 시 Denoising Process

`get_action_with_features()` (라인 288-364):

```python
# 초기: 순수 노이즈에서 시작
actions = torch.randn([B, 16, 29])
dt = 1.0 / 4  # num_inference_timesteps = 4

for t in range(4):
    t_cont = t / 4.0  # 0, 0.25, 0.5, 0.75
    # ... DiT forward로 velocity 예측 ...
    actions = actions + dt * pred_velocity  # Euler 적분
```

4번의 디노이징 스텝만으로 순수 노이즈 → 최종 액션으로 변환합니다 (매우 효율적).

### 3.3 텐서 쉐이프 요약 테이블

| 단계 | 텐서 | 쉐이프 | 설명 |
|------|-------|--------|------|
| 입력 | `pixel_values` | `[B*N, C, H, W]` | 전처리된 이미지 |
| 입력 | `input_ids` | `[B, S]` | 토큰화된 텍스트+이미지 |
| 입력 | `state` | `[B, 1, 29]` | 패딩된 정규화 상태 |
| 입력 | `action` | `[B, 16, 29]` | 패딩된 정규화 액션 |
| Backbone | `backbone_features` | `[B, S, 2048]` | VLM 멀티모달 특징 |
| State Enc. | `state_features` | `[B, 1, 1536]` | 인코딩된 상태 |
| Action Enc. | `action_features` | `[B, 16, 1536]` | 인코딩된 노이즈 액션 |
| DiT 입력 | `sa_embs` | `[B, 17, 1536]` | state + action concat |
| DiT 출력 | `model_output` | `[B, 17, 1024]` | 변환된 특징 |
| Decoder | `pred_actions` | `[B, 16, 29]` | 예측 velocity |
| Loss | `loss` | 스칼라 | 마스킹된 MSE |

---

## 4. 학습 로직

### 4.1 Loss Function: Flow Matching MSE

**파일:** `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` 라인 198-248

GR00T N1.6는 **Flow Matching** (Conditional Flow Matching) 프레임워크를 사용합니다. 이는 Diffusion Policy의 변형으로, 노이즈에서 데이터로의 **확률적 흐름(probability flow)**의 속도장(velocity field)을 학습합니다.

#### 핵심 수식:

1. **노이즈 보간 (Forward Process)**:

   \[
   x_t = (1 - t) \cdot \epsilon + t \cdot x_1
   \]

   여기서 \(\epsilon \sim \mathcal{N}(0, I)\)는 노이즈, \(x_1\)은 실제 액션, \(t \in [0, 1)\)는 시간.

2. **목표 Velocity**:

   \[
   v^* = x_1 - \epsilon
   \]

3. **Loss**:

   \[
   \mathcal{L} = \frac{\sum_{b,t,d} \| \hat{v}_{b,t,d} - v^*_{b,t,d} \|^2 \cdot m_{b,t,d}}{\sum_{b,t,d} m_{b,t,d}}
   \]

   여기서 \(m\)은 action_mask (유효 차원만 학습).

**코드:**
```python
noise = torch.randn(actions.shape)
t = self.sample_time(B, device, dtype)  # Beta(1.5, 1.0) 분포에서 샘플링
noisy_trajectory = (1 - t) * noise + t * actions
velocity = actions - noise  # 목표 velocity

# ... DiT forward ...

action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

#### 시간 샘플링 전략:

시간 `t`는 **Beta(1.5, 1.0)** 분포에서 샘플링된 후 `(1-t) * 0.999`로 변환됩니다. Beta(1.5, 1.0) 분포는 작은 값(노이즈에 가까운 상태) 쪽으로 약간 편향되어, 디노이징 초기 단계에서의 학습을 강화합니다.

### 4.2 Optimizer & Scheduler

**파일:** `gr00t/configs/training/training_config.py` + `gr00t/experiment/experiment.py`

| 설정 | 기본값 | 설명 |
|------|--------|------|
| Optimizer | `adamw_torch_fused` (사전학습) / `adamw_torch` (파인튜닝) | Fused AdamW가 기본 |
| Learning Rate | `1e-4` | |
| LR Scheduler | `cosine` | Cosine Annealing |
| Weight Decay | `1e-5` | |
| Warmup Ratio | `0.05` (전체 스텝의 5%) | Linear warmup |
| Max Grad Norm | `1.0` | Gradient clipping |
| Max Steps | `30,000` | |
| Mixed Precision | `bf16 = True` | BFloat16 학습 |

**코드 (experiment.py 라인 191-221):**
```python
training_args = TrainingArguments(
    learning_rate=config.training.learning_rate,         # 1e-4
    lr_scheduler_type=config.training.lr_scheduler_type, # "cosine"
    weight_decay=config.training.weight_decay,           # 1e-5
    warmup_ratio=config.training.warmup_ratio,           # 0.05
    max_grad_norm=config.training.max_grad_norm,         # 1.0
    bf16=config.training.bf16,                           # True
    optim=config.training.optim,                         # "adamw_torch_fused"
    gradient_checkpointing=config.training.gradient_checkpointing,
    deepspeed=deepspeed_config,                          # ZeRO Stage 2
    ...
)
```

### 4.3 학습 전략 (Freeze / Unfreeze)

GR00T N1.6은 **선택적 파라미터 동결(Selective Freezing)** 전략을 사용합니다:

| 모듈 | 기본 동결 상태 | 설정 파라미터 |
|------|---------------|---------------|
| Vision Encoder (SigLIP-2) | **동결** | `tune_visual=False` |
| Vision Projector (mlp1) | **동결** | (tune_visual과 연동) |
| LLM (Qwen2) 하위 레이어 | **동결** | `tune_llm=False` |
| LLM 상위 4 레이어 | **학습** | `tune_top_llm_layers=4` |
| VL LayerNorm (vlln) | **학습** | `tune_vlln=True` |
| Action Head Projectors | **학습** | `tune_projector=True` |
| DiT (32 layers) | **학습** | `tune_diffusion_model=True` |

**핵심 포인트:**
- N1.5에서 사용하던 VLM 이후의 4-layer Transformer Adapter를 제거하고, 대신 **VLM 상위 4개 LLM 레이어를 직접 학습**합니다.
- 학습 가능한 backbone 파라미터는 **FP32로 유지**합니다 (`backbone_trainable_params_fp32=True`).
- 동결된 모듈은 학습 중 자동으로 `eval()` 모드로 전환되어 Dropout/BatchNorm이 올바르게 동작합니다.

### 4.4 분산 학습 (DeepSpeed)

- **기본 설정**: DeepSpeed ZeRO Stage 2 (optimizer states + gradients 분산)
- **설정 파일**: `gr00t/configs/deepspeed/zero2_config.json`
- Multi-GPU에서는 `ddp_find_unused_parameters=False`

### 4.5 State Augmentation

학습 시 로봇 상태에 대한 2가지 증강 기법이 적용됩니다:

1. **State Dropout** (`state_dropout_prob`): 학습 시 확률적으로 상태 특징을 학습 가능한 mask_token으로 교체하여, 모델이 상태 없이도 동작하도록 학습
2. **Additive Gaussian Noise** (`state_additive_noise_scale`): 상태 특징에 가우시안 노이즈를 추가하여 로버스트성 향상

---

## 5. 핵심 코드 스니펫 요약

### 5.1 전체 모델 Forward Pass

**파일:** `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` 라인 496-513

```python
class Gr00tN1d6(PreTrainedModel):
    def forward(self, inputs: dict) -> BatchFeature:
        # 1. 입력 준비: backbone(VLM)용과 action_head용으로 분리
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # 2. VLM Backbone 실행: 이미지+텍스트 → 멀티모달 특징 추출
        backbone_outputs = self.backbone(backbone_inputs)
        #    backbone_features: [B, S, 2048]

        # 3. Action Head 실행: Flow Matching으로 loss 계산
        action_outputs = self.action_head(backbone_outputs, action_inputs)
        #    loss: scalar (MSE)

        return action_outputs
```

### 5.2 Eagle Backbone — 이미지-텍스트 융합

**파일:** `gr00t/model/modules/eagle_backbone.py` 라인 105-120

```python
class EagleBackbone(torch.nn.Module):
    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        # VLM에 input_ids, attention_mask, pixel_values 전달
        # 내부적으로 이미지 토큰을 Vision 임베딩으로 교체 후 LLM 통과
        outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs["hidden_states"][-1]  # 마지막 hidden state

        # 이미지 토큰 위치 마스크 생성 (AlternateVLDiT에서 사용)
        image_mask = vl_input["input_ids"] == self.model.config.image_token_index

        return BatchFeature(data={
            "backbone_features": outputs,        # [B, S, 2048]
            "backbone_attention_mask": attention_mask,
            "image_mask": image_mask,            # AlternateVLDiT를 위한 마스크
        })
```

### 5.3 Action Head — Flow Matching 학습

**파일:** `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` 라인 148-256

```python
class Gr00tN1d6ActionHead(nn.Module):
    def forward(self, backbone_output, action_input) -> BatchFeature:
        # 1. VLM 출력 정규화
        vl_embeds = self.vlln(backbone_output.backbone_features)  # [B, S, 2048]

        # 2. 상태 인코딩 (embodiment별 가중치)
        state_features = self.state_encoder(
            action_input.state, embodiment_id  # [B, 1, 29] → [B, 1, 1536]
        )

        # 3. Flow Matching: 노이즈 보간 + velocity 목표 생성
        noise = torch.randn(actions.shape)
        t = self.sample_time(B, device, dtype)  # Beta(1.5, 1.0)
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise  # 학습 목표

        # 4. 노이즈 액션 인코딩
        action_features = self.action_encoder(
            noisy_trajectory, t_discretized, embodiment_id  # [B, 16, 29] → [B, 16, 1536]
        )

        # 5. State + Action 결합
        sa_embs = torch.cat((state_features, action_features), dim=1)  # [B, 17, 1536]

        # 6. AlternateVLDiT (32-layer): Cross-attn 조건화
        model_output = self.model(
            hidden_states=sa_embs,              # [B, 17, 1536]
            encoder_hidden_states=vl_embeds,    # [B, S, 2048] (조건)
            timestep=t_discretized,
            image_mask=image_mask,              # 이미지/텍스트 교대 attend
        )  # → [B, 17, 1024]

        # 7. 디코딩 및 Loss 계산
        pred = self.action_decoder(model_output, embodiment_id)  # [B, 17, 29]
        pred_actions = pred[:, -16:]  # action 부분만 추출

        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        return {"loss": loss, ...}
```

### 5.4 AlternateVLDiT — 이미지/텍스트 교대 Cross-Attention

**파일:** `gr00t/model/modules/dit.py` 라인 289-364

```python
class AlternateVLDiT(DiT):
    def forward(self, hidden_states, encoder_hidden_states, timestep,
                image_mask, backbone_attention_mask, ...):
        # 이미지/비이미지 토큰 마스크 생성
        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask

        for idx, block in enumerate(self.transformer_blocks):  # 32 블록
            if idx % 2 == 1:
                # 홀수: Self-Attention (state/action 간)
                hidden_states = block(hidden_states, temb=temb)
            else:
                # 짝수: Cross-Attention
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    # 텍스트 토큰에 attend
                    mask = non_image_attention_mask
                else:
                    # 이미지 토큰에 attend
                    mask = image_attention_mask
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=mask,
                    temb=temb,
                )

        # Adaptive LayerNorm으로 최종 출력 조건화
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)
```

### 5.5 추론 — Euler Denoising

**파일:** `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` 라인 288-364

```python
@torch.no_grad()
def get_action_with_features(self, backbone_features, state_features, embodiment_id, ...):
    # 순수 노이즈에서 시작
    actions = torch.randn([B, 16, action_dim])
    dt = 1.0 / 4  # 4 스텝 추론

    for t in range(4):
        t_cont = t / 4.0                           # 0.0, 0.25, 0.5, 0.75
        t_disc = int(t_cont * 1000)                # 0, 250, 500, 750

        action_features = self.action_encoder(actions, t_disc, embodiment_id)
        sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output = self.model(sa_embs, backbone_features, t_disc, ...)
        pred = self.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -16:]

        actions = actions + dt * pred_velocity      # Euler 적분

    return actions  # [B, 16, action_dim]
```

---

## 부록: 모델 파라미터 요약

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `backbone_embedding_dim` | 2048 | VLM 출력 차원 |
| `hidden_size` | 1024 | DiT 내부 및 decoder 차원 |
| `input_embedding_dim` | 1536 | State/Action 인코딩 차원 (= DiT inner_dim: 32 heads * 48 dim) |
| `action_horizon` | 16 | 예측할 미래 액션 스텝 수 |
| `max_action_dim` | 29 | 최대 액션 차원 |
| `max_state_dim` | 29 | 최대 상태 차원 |
| `max_num_embodiments` | 32 | 지원 가능한 최대 로봇 종류 수 |
| `num_inference_timesteps` | 4 | 추론 시 디노이징 스텝 수 |
| `diffusion num_layers` | 32 | DiT 레이어 수 (N1.5: 16) |
| `diffusion num_attention_heads` | 32 | DiT 어텐션 헤드 수 |
| `diffusion attention_head_dim` | 48 | DiT 헤드 차원 |
| `noise_beta_alpha` | 1.5 | 시간 샘플링 Beta 분포 alpha |
| `noise_beta_beta` | 1.0 | 시간 샘플링 Beta 분포 beta |
| `select_layer` | 16 | VLM에서 사용할 LLM 레이어 수 |
| `tune_top_llm_layers` | 4 | 학습할 상위 LLM 레이어 수 |

---

*본 보고서는 Isaac-GR00T 리포지토리의 코드를 직접 분석하여 작성되었습니다.*
