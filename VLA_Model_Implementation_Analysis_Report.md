# GR00T N1.6 VLA 모델 구현 상세 분석 보고서

## 목차
1. [프로젝트 오버뷰 & 구조](#1-프로젝트-오버뷰--구조)
2. [모델 아키텍처 상세 분석](#2-모델-아키텍처-상세-분석)
3. [데이터 흐름 및 텐서 쉐이프](#3-데이터-흐름-및-텐서-쉐이프)
4. [학습 로직](#4-학습-로직)
5. [핵심 코드 스니펫 요약](#5-핵심-코드-스니펫-요약)

---

## 1. 프로젝트 오버뷰 & 구조

### 1.1 프로젝트 핵심 목적

**NVIDIA Isaac GR00T N1.6**는 범용 휴머노이드 로봇 기술을 위한 오픈소스 Vision-Language-Action (VLA) 모델입니다. 이 모델은:

- **멀티모달 입력**: 이미지(비전), 언어 명령어, 로봇 상태(state)를 통합 처리
- **크로스 임바디먼트(Cross-Embodiment)**: 다양한 로봇 플랫폼에 적응 가능한 통합 모델
- **Diffusion 기반 Action 생성**: Flow Matching 기법을 활용한 연속적 action trajectory 예측
- **3B 파라미터 규모**: VLM backbone + Diffusion Transformer로 구성

모델은 10,000+ 시간의 로봇 데모 데이터로 사전 학습되었으며, fine-tuning을 통해 특정 작업에 적응할 수 있습니다.

### 1.2 폴더 구조

```
gr00t/
├── model/                          # 모델 정의
│   ├── gr00t_n1d6/                 # N1.6 모델 구현
│   │   ├── gr00t_n1d6.py          # 메인 모델 클래스
│   │   ├── processing_gr00t_n1d6.py # 데이터 전처리 및 collator
│   │   └── image_augmentations.py  # 이미지 증강
│   ├── modules/                    # 모델 구성 요소
│   │   ├── eagle_backbone.py       # Vision-Language Backbone
│   │   ├── dit.py                  # Diffusion Transformer
│   │   ├── embodiment_conditioned_mlp.py # Multi-embodiment projectors
│   │   └── nvidia/Eagle-Block2A-2B-v2/ # Eagle VLM 모델
│   └── base/
│       └── model_pipeline.py       # 학습 파이프라인 추상화
├── data/                           # 데이터 로딩
│   ├── dataset/
│   │   ├── factory.py              # 데이터셋 팩토리
│   │   ├── sharded_mixture_dataset.py
│   │   └── lerobot_episode_loader.py
│   ├── collator/
│   │   └── collators.py            # 데이터 배치 구성
│   └── state_action/
│       └── state_action_processor.py # State/Action 정규화
├── experiment/                     # 학습 스크립트
│   ├── launch_finetune.py          # Fine-tuning 진입점
│   ├── launch_train.py             # Full training 진입점
│   ├── trainer.py                  # Custom HuggingFace Trainer
│   └── experiment.py               # 실험 실행 로직
├── configs/                        # 설정 파일
│   ├── model/gr00t_n1d6.py        # 모델 설정
│   ├── data/data_config.py        # 데이터 설정
│   └── training/training_config.py # 학습 설정
└── eval/                           # 평가 및 추론
    ├── rollout_policy.py           # Policy rollout
    ├── run_gr00t_server.py         # Policy 서버 (추론 진입점)
    └── open_loop_eval.py           # 오픈루프 평가

examples/                           # 각종 벤치마크 예제
scripts/deployment/                 # TensorRT 등 배포 스크립트
```

### 1.3 주요 진입점 (Entry Points)

| 목적 | 파일 경로 | 설명 |
|------|----------|------|
| **Fine-tuning** | `gr00t/experiment/launch_finetune.py` | 사전학습된 모델을 특정 embodiment에 fine-tune |
| **Full Training** | `gr00t/experiment/launch_train.py` | 처음부터 또는 커스텀 설정으로 학습 |
| **Inference (Server)** | `gr00t/eval/run_gr00t_server.py` | Policy 서버 실행 (REST API) |
| **Standalone Inference** | `scripts/deployment/standalone_inference_script.py` | 단일 추론 스크립트 |
| **Open-loop Evaluation** | `gr00t/eval/open_loop_eval.py` | 데이터셋 대비 action 정확도 측정 |

---

## 2. 모델 아키텍처 상세 분석

### 2.1 Core Class: `Gr00tN1d6`

**파일**: `gr00t/model/gr00t_n1d6/gr00t_n1d6.py`

**메인 클래스**: `Gr00tN1d6(PreTrainedModel)`

이 클래스는 HuggingFace `PreTrainedModel`을 상속받아 표준 인터페이스를 제공합니다.

```python
class Gr00tN1d6(PreTrainedModel):
    config_class = Gr00tN1d6Config
    
    def __init__(self, config, transformers_loading_kwargs):
        super().__init__(config)
        self.backbone = EagleBackbone(...)      # Vision-Language Encoder
        self.action_head = Gr00tN1d6ActionHead(...)  # Diffusion-based Action Decoder
        self.collator = Gr00tN1d6DataCollator(...)   # Data preprocessing
```

### 2.2 Components: 모델 구성 요소

모델은 크게 **3개의 주요 모듈**로 구성됩니다:

#### 2.2.1 Vision-Language Backbone: `EagleBackbone`

**파일**: `gr00t/model/modules/eagle_backbone.py`

**역할**: 이미지와 텍스트를 통합된 임베딩 공간으로 인코딩

**구조**:
- **Vision Model**: SigLIP 기반 비전 인코더 (NVIDIA Cosmos-Reason-2B VLM의 변형)
- **Language Model**: 2B 파라미터 언어 모델 (16개 레이어 중 상위 4개 레이어만 사용)
- **MLP Projector** (`mlp1`): Vision feature를 Language embedding space로 투영

**주요 특징**:
1. **Flexible Resolution**: 고정 크기가 아닌 원본 aspect ratio 유지 가능
2. **Flash Attention 2**: 메모리 효율적인 attention 구현 (필수)
3. **BF16 정밀도**: 기본적으로 bfloat16으로 로드
4. **Layer Selection**: 전체 LLM이 아닌 선택된 레이어까지만 사용 (`select_layer=16`)

**Forward Pass**:
```python
def forward(self, vl_input):
    outputs = self.model(
        input_ids=vl_input["input_ids"],          # [B, seq_len]
        attention_mask=vl_input["attention_mask"], # [B, seq_len]
        pixel_values=vl_input["pixel_values"],     # [B, num_images, C, H, W]
        output_hidden_states=True
    )
    hidden_states = outputs["hidden_states"][-1]   # [B, seq_len, 2048]
    image_mask = input_ids == image_token_index    # 이미지 토큰 위치 마스크
    
    return {
        "backbone_features": hidden_states,         # [B, seq_len, 2048]
        "backbone_attention_mask": attention_mask,  # [B, seq_len]
        "image_mask": image_mask                    # [B, seq_len]
    }
```

**설정 가능 파라미터**:
- `tune_llm`: LLM 레이어 학습 여부 (기본: False)
- `tune_visual`: Vision encoder 학습 여부 (기본: False)
- `tune_top_llm_layers`: 상위 N개 LLM 레이어만 학습 (기본: 4)
- `select_layer`: 사용할 LLM 레이어 깊이 (기본: 16)

#### 2.2.2 Action Head: `Gr00tN1d6ActionHead`

**파일**: `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` (19-402 라인)

**역할**: Vision-Language 특징과 로봇 상태를 받아 action trajectory를 생성

**구조**:
1. **State Encoder**: `CategorySpecificMLP` (embodiment별 별도 가중치)
2. **Action Encoder**: `MultiEmbodimentActionEncoder` (noisy action + timestep encoding)
3. **Diffusion Model**: `AlternateVLDiT` (32-layer Diffusion Transformer)
4. **Action Decoder**: `CategorySpecificMLP` (velocity 예측)

**핵심 하위 모듈**:

##### A. State Encoder & Action Encoder/Decoder

**파일**: `gr00t/model/modules/embodiment_conditioned_mlp.py`

**Multi-Embodiment Support**: 각 로봇 플랫폼(embodiment)마다 별도의 가중치 세트를 유지

```python
class CategorySpecificMLP(nn.Module):
    """각 embodiment마다 별도의 Linear 가중치를 가진 MLP"""
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)
    
    def forward(self, x, cat_ids):
        # cat_ids: [B] - embodiment ID
        # x: [B, T, input_dim]
        # 각 배치별로 해당 embodiment의 가중치 선택
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)  # [B, T, output_dim]
```

**Action Encoder 상세**:
```python
class MultiEmbodimentActionEncoder(nn.Module):
    def forward(self, actions, timesteps, cat_ids):
        # actions: [B, T, action_dim] - noisy action trajectory
        # timesteps: [B] - diffusion timestep (0~1000)
        
        # 1. Action embedding
        a_emb = self.W1(actions, cat_ids)  # [B, T, hidden_size]
        
        # 2. Sinusoidal timestep encoding
        tau_emb = self.pos_encoding(timesteps)  # [B, T, hidden_size]
        
        # 3. Concatenate and process
        x = torch.cat([a_emb, tau_emb], dim=-1)  # [B, T, 2*hidden_size]
        x = swish(self.W2(x, cat_ids))           # [B, T, hidden_size]
        x = self.W3(x, cat_ids)                  # [B, T, hidden_size]
        return x
```

##### B. Diffusion Transformer: `AlternateVLDiT`

**파일**: `gr00t/model/modules/dit.py` (289-364 라인)

**N1.6의 주요 변경점**: N1.5 대비 32개 레이어로 증가 (2배)

**구조**:
- **32 Transformer Layers** (N1.5는 16개)
- **Interleaved Architecture**: Self-attention과 Cross-attention 교대 배치
- **Alternate Vision-Language Attention**: 이미지 토큰과 텍스트 토큰을 번갈아 attend

```python
class AlternateVLDiT(DiT):
    def forward(self, hidden_states, encoder_hidden_states, timestep, 
                image_mask, backbone_attention_mask):
        # hidden_states: [B, T, 1536] - (state + action) embeddings
        # encoder_hidden_states: [B, S, 2048] - VL backbone features
        # timestep: [B] - discrete timestep bucket (0~999)
        
        # Timestep encoding
        temb = self.timestep_encoder(timestep)  # [B, 1536]
        
        # Separate image and text tokens
        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask
        
        # Process through 32 layers
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Odd layers: Self-attention
                hidden_states = block(
                    hidden_states, 
                    encoder_hidden_states=None,
                    temb=temb
                )
            else:
                # Even layers: Cross-attention
                # 2개 블록마다 텍스트/이미지 교대
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_mask = non_image_attention_mask  # Attend to text
                else:
                    curr_mask = image_attention_mask      # Attend to images
                
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_mask,
                    temb=temb
                )
        
        # Output projection with adaptive layer norm
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)  # [B, T, 1024]
```

**Attention 패턴** (`attend_text_every_n_blocks=2`):
```
Layer 0:  Cross-attention to TEXT tokens
Layer 1:  Self-attention
Layer 2:  Cross-attention to IMAGE tokens
Layer 3:  Self-attention
Layer 4:  Cross-attention to TEXT tokens
...
```

이 구조는 텍스트와 이미지 정보를 균형있게 활용하도록 설계되었습니다.

### 2.3 Modality Fusion: Vision-Language-Action 융합 방식

**융합 방식**: **Cross-Attention 기반 조건부 생성 (Conditional Generation)**

#### 융합 단계별 흐름:

**Stage 1: Vision + Language 융합 (Backbone 내부)**
```
Images → Vision Encoder → Visual tokens
Text   → Tokenizer      → Text tokens
                         ↓
              Language Model (with visual tokens interleaved)
                         ↓
              Unified VL embeddings [B, seq_len, 2048]
```

**Stage 2: VL embeddings + State + Action 융합 (Action Head)**

```python
# Action Head forward() 구조:

# 1. State encoding (embodiment-specific)
state_features = self.state_encoder(state, embodiment_id)  
# [B, state_dim] → [B, 1, 1536]

# 2. Noisy action encoding
noisy_action = (1-t) * noise + t * action_gt  # Flow matching interpolation
action_features = self.action_encoder(noisy_action, timestep, embodiment_id)
# [B, action_horizon, action_dim] → [B, action_horizon, 1536]

# 3. Concatenate state and action
sa_embeds = torch.cat([state_features, action_features], dim=1)
# [B, 1 + action_horizon, 1536]

# 4. Cross-attention to VL features
for layer in diffusion_transformer:
    if cross_attention_layer:
        # Query: state-action embeddings
        # Key, Value: VL backbone features
        sa_embeds = cross_attention(
            query=sa_embeds,                    # [B, T_sa, 1536]
            key_value=vl_embeds,                # [B, T_vl, 2048]
            attention_mask=image_or_text_mask   # 선택적으로 image/text attend
        )
    elif self_attention_layer:
        sa_embeds = self_attention(sa_embeds)

# 5. Decode velocity
velocity_pred = self.action_decoder(sa_embeds, embodiment_id)
# [B, action_horizon, action_dim]
```

**핵심 메커니즘**:
1. **Separate Encoding**: State, Action, VL을 각각 인코딩 후 통합
2. **Cross-Attention Conditioning**: VL 정보를 key-value로 사용하여 action 생성 조건화
3. **Embodiment-Specific Projectors**: 각 로봇별 별도 encoder/decoder로 domain gap 해결
4. **Alternate Attention**: 이미지와 텍스트 정보를 분리하여 선택적 참조

---

## 3. 데이터 흐름 및 텐서 쉐이프

### 3.1 Training Forward Pass

전체 데이터 흐름을 단계별로 추적합니다.

#### 입력 데이터 구조 (from Dataset)

```python
batch = {
    # Vision-Language inputs
    "input_ids": [B, seq_len],           # Tokenized text with image placeholders
    "attention_mask": [B, seq_len],      # Attention mask
    "pixel_values": [B, num_imgs, 3, H, W],  # Images (가변 해상도)
    
    # Robot state & action
    "state": [B, max_state_dim],         # Padded state (실제 dim은 작을 수 있음)
    "action": [B, max_action_horizon, max_action_dim],  # Padded action
    "action_mask": [B, max_action_horizon, max_action_dim],  # Valid action 마스크
    "embodiment_id": [B],                # Embodiment index (0~31)
}
```

**Dimension 예시**:
- `seq_len`: ~100-500 (이미지 개수와 텍스트 길이에 따라 가변)
- `num_imgs`: 예: 4 (4개 카메라 뷰 × 1 timestep, 또는 1개 뷰 × 4 timesteps)
- `H, W`: 가변 (Eagle은 any-resolution 지원)
- `max_state_dim`: 29 (기본값, 실제 사용은 embodiment마다 다름)
- `max_action_dim`: 29 (기본값)
- `max_action_horizon`: 40 (기본값, 실제 사용은 16 등 더 작을 수 있음)

#### Step-by-Step Tensor Transformations

```python
# ============ BACKBONE (EagleBackbone) ============
inputs:
    input_ids:      [B, seq_len=300]
    pixel_values:   [B, 4, 3, 384, 384]  # 예시: 4개 이미지, 384x384
    attention_mask: [B, seq_len=300]

# Vision Encoder (SigLIP)
visual_features = vision_model(pixel_values)
# → [B, 4, num_patches, vision_dim]
# → [B, 4, 256, 1152]  (예: 24x24 patches=576, 하지만 variable)

# MLP Projection
visual_features = mlp1(visual_features)
# → [B, 4, num_patches, 2048]

# Language Model (with visual tokens merged)
# input_ids에서 image_token_index 위치에 visual_features 삽입
hidden_states = language_model(input_ids, visual_embeds)
# → [B, seq_len + 4*num_patches, 2048]
# → [B, ~1300, 2048]  (텍스트 토큰 + 이미지 토큰 통합)

backbone_output:
    backbone_features:      [B, 1300, 2048]
    backbone_attention_mask: [B, 1300]
    image_mask:             [B, 1300]  # True for image tokens


# ============ ACTION HEAD (Gr00tN1d6ActionHead) ============

# --- State Encoding ---
state_input: [B, 29]  # max_state_dim
state_features = state_encoder(state_input, embodiment_id)
# → [B, 1, 1536]  (unsqueeze를 통해 sequence dim 추가)

# State dropout (training only)
if training and state_dropout_prob > 0:
    # Randomly mask some state features
    state_features = state_features * (1 - dropout_mask) + mask_token * dropout_mask

# --- Action Encoding ---
actions: [B, 16, 29]  # [action_horizon, max_action_dim]

# Flow matching: noisy trajectory 생성
t = sample_time()  # [B, 1, 1] ~ Beta(1.5, 1.0) distribution
noise = torch.randn_like(actions)  # [B, 16, 29]
noisy_trajectory = (1 - t) * noise + t * actions
velocity_gt = actions - noise  # Ground truth velocity

# Discretize timestep for embedding
t_discretized = (t * 1000).long()  # [B] in [0, 999]

# Encode noisy action + timestep
action_features = action_encoder(noisy_trajectory, t_discretized, embodiment_id)
# → [B, 16, 1536]

# Position embedding (optional)
if add_pos_embed:
    pos_embs = position_embedding(torch.arange(16))  # [16, 1536]
    action_features = action_features + pos_embs

# --- Concatenate State + Action ---
sa_embeds = torch.cat([state_features, action_features], dim=1)
# → [B, 17, 1536]  (1 state + 16 action steps)

# --- Diffusion Transformer (AlternateVLDiT) ---
# 32 layers of alternating self/cross attention
hidden_states = sa_embeds  # [B, 17, 1536]
encoder_hidden_states = backbone_features  # [B, 1300, 2048]

temb = timestep_encoder(t_discretized)  # [B, 1536]

for idx in range(32):
    if idx % 2 == 1:
        # Self-attention layer
        hidden_states = transformer_block(
            hidden_states,  # Q, K, V from hidden_states
            temb=temb
        )
        # → [B, 17, 1536]
    else:
        # Cross-attention layer
        # Select image or text mask based on layer index
        if idx % 4 == 0:
            mask = non_image_mask  # Attend to text
        else:
            mask = image_mask      # Attend to images
        
        hidden_states = transformer_block(
            hidden_states,                 # Q from hidden_states
            encoder_hidden_states,         # K, V from VL features
            encoder_attention_mask=mask,
            temb=temb
        )
        # → [B, 17, 1536]

# --- Output Projection ---
# Adaptive layer norm with timestep conditioning
shift, scale = proj_out_1(silu(temb)).chunk(2)  # [B, 1536] → 2x [B, 1536]
hidden_states = norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
# → [B, 17, 1536]

model_output = proj_out_2(hidden_states)  # [B, 17, 1024]

# --- Action Decoding ---
pred_output = action_decoder(model_output, embodiment_id)
# → [B, 17, 29]  (max_action_dim)

pred_velocity = pred_output[:, -16:, :]  # Slice action part
# → [B, 16, 29]

# --- Loss Calculation ---
action_loss = F.mse_loss(pred_velocity, velocity_gt, reduction='none') * action_mask
loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

### 3.2 Inference Forward Pass (Action Generation)

추론 시에는 **Denoising 과정**을 반복합니다.

```python
# Initial setup
actions = torch.randn([B, 16, 29])  # Random noise
dt = 1.0 / num_inference_timesteps  # 예: 1/4 = 0.25

# Iterative denoising (예: 4 steps)
for t in range(4):  # num_inference_timesteps = 4
    t_cont = t / 4.0  # 0, 0.25, 0.5, 0.75
    t_discretized = int(t_cont * 1000)  # 0, 250, 500, 750
    
    # Encode current (noisy) actions
    action_features = action_encoder(actions, t_discretized, embodiment_id)
    # [B, 16, 1536]
    
    sa_embeds = torch.cat([state_features, action_features], dim=1)
    # [B, 17, 1536]
    
    # Run through DiT
    model_output = diffusion_model(
        sa_embeds, 
        encoder_hidden_states=vl_embeds,
        timestep=t_discretized
    )
    # [B, 17, 1024]
    
    pred_velocity = action_decoder(model_output, embodiment_id)[:, -16:, :]
    # [B, 16, 29]
    
    # Euler integration step (Flow Matching ODE)
    actions = actions + dt * pred_velocity
    # [B, 16, 29]

# Final output
final_actions = actions  # [B, 16, 29]
```

**ODE 수식**:
```
dz/dt = v_θ(z_t, c)
```
여기서:
- `z_t`: 시간 t에서의 noisy action
- `v_θ`: 모델이 예측하는 velocity field
- `c`: conditioning (VL features, state)

**Euler method**로 이산화:
```
z_{t+dt} = z_t + dt * v_θ(z_t, c)
```

---

## 4. 학습 로직

### 4.1 Loss Function: Flow Matching

**파일**: `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` (148-256 라인)

**Loss Type**: **Velocity Prediction Loss (MSE)**

#### Flow Matching 원리

일반적인 diffusion과 달리, **Flow Matching**은 noise → data로의 연속적인 flow를 학습합니다.

**Training Objective**:
```
L = E_{t~Beta, ε~N(0,I)} [ ||v_θ(z_t, t, c) - v_gt||^2 ]
```

여기서:
- `t ~ Beta(α=1.5, β=1.0)`: 시간 샘플링 (0에 가까운 값 선호)
- `z_t = (1-t)·ε + t·x`: Interpolated trajectory
- `v_gt = x - ε`: Ground truth velocity (data에서 noise 방향)
- `v_θ`: 모델이 예측하는 velocity

#### 구현 코드

```python
def forward(self, backbone_output, action_input):
    # Get ground truth actions
    actions = action_input.action  # [B, action_horizon, action_dim]
    
    # Sample time from Beta distribution
    # Beta(1.5, 1.0)은 0에 가까운 값을 더 많이 샘플링
    # → 초기 노이즈 제거에 집중
    t = self.sample_time(actions.shape[0])  # [B]
    t = t[:, None, None]  # [B, 1, 1] for broadcasting
    
    # Sample Gaussian noise
    noise = torch.randn_like(actions)  # [B, T, D]
    
    # Flow matching interpolation
    noisy_trajectory = (1 - t) * noise + t * actions
    # t=0: pure noise
    # t=1: pure data
    # t=0.5: 50-50 mixture
    
    # Ground truth velocity
    velocity = actions - noise
    
    # ... (encode and process through DiT) ...
    
    # Predict velocity
    pred_velocity = self.action_decoder(model_output, embodiment_id)
    pred_actions = pred_velocity[:, -actions.shape[1]:]  # [B, T, D]
    
    # Compute loss with masking
    action_mask = action_input.action_mask  # [B, T, D]
    action_loss = F.mse_loss(pred_actions, velocity, reduction='none') * action_mask
    loss = action_loss.sum() / (action_mask.sum() + 1e-6)
    
    return {"loss": loss}
```

**Loss 특징**:
1. **Masked MSE**: `action_mask`를 통해 유효한 action dimension과 horizon만 학습
2. **Velocity Space**: Action을 직접 예측하지 않고 velocity를 예측
3. **Time Weighting**: Beta distribution으로 초기 denoising에 더 집중

#### Action Mask의 역할

```python
# Example: libero_panda embodiment
# Actual action dim: 7 (joint positions)
# max_action_dim: 29 (padding)
# Actual horizon: 16
# max_action_horizon: 40 (padding)

action_mask = torch.ones([B, 40, 29])
action_mask[:, 16:, :] = 0   # Mask out unused horizon
action_mask[:, :, 7:] = 0    # Mask out padded dimensions
# → Only [B, 16, 7] region contributes to loss
```

### 4.2 Optimization

**파일**: `gr00t/experiment/trainer.py`, `gr00t/configs/training/training_config.py`

#### Optimizer 설정

```python
# Default configuration (from launch_finetune.py)
optimizer: "adamw_torch"
learning_rate: 1e-4
weight_decay: 0.01
warmup_ratio: 0.03
max_steps: 2000
gradient_accumulation_steps: 자동 계산 (global_batch_size 기반)
```

#### Learning Rate Schedule

**Scheduler**: `linear` (HuggingFace default with warmup)

```
LR
 ^
 |     /----------------\
 |    /                  \
 |   /                    \
 |  /                      \___
 | /                            \___
 |/________________________________\___> Steps
 0   warmup               max_steps
     (3% of max)
```

**Warmup 설정**:
```python
warmup_steps = max_steps * warmup_ratio  # 2000 * 0.03 = 60 steps
```

#### 학습 전략: Selective Fine-tuning

**설정 파라미터** (from `Gr00tN1d6Config`):

```python
# Backbone (Eagle VLM)
tune_llm: False              # LLM 전체 freeze
tune_visual: False           # Vision encoder freeze
tune_top_llm_layers: 4       # 상위 4개 LLM layer만 학습
reproject_vision: False      # Vision projection layer freeze

# Action Head
tune_projector: True         # State/Action encoder-decoder 학습
tune_diffusion_model: True   # DiT 전체 학습
tune_vlln: True              # Vision-language layer norm 학습

# Precision
load_bf16: True              # Backbone을 BF16으로 로드
backbone_trainable_params_fp32: True  # 학습 가능 파라미터는 FP32로 변환
```

**실제 학습 파라미터 수** (3B 모델 기준):
```
Total parameters: ~3B
Trainable parameters: ~700M (23%)
  - Top 4 LLM layers: ~200M
  - DiT (32 layers): ~400M
  - Projectors (state/action encoder-decoder): ~50M
  - VLLN: ~5M
```

#### Gradient Checkpointing

**설정**: 기본적으로 활성화되지 않음 (메모리 충분할 경우)

```python
# 활성화 방법 (메모리 부족 시)
model.gradient_checkpointing_enable()
```

#### Batch Size 계산

```python
# launch_finetune.py example
global_batch_size = 32
num_gpus = 1
per_device_batch_size = 8  # GPU memory에 따라 조정

gradient_accumulation_steps = global_batch_size // (per_device_batch_size * num_gpus)
# = 32 // 8 = 4 steps
```

#### Mixed Precision Training

**설정**: `bf16=True` (config에서 자동 적용)

```python
# DeepSpeed ZeRO-2 config (multi-GPU의 경우)
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}
```

### 4.3 Data Augmentation

**파일**: `gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py`, `image_augmentations.py`

#### Image Augmentation (Training only)

```python
# Default augmentation pipeline
train_image_transform = A.Compose([
    A.RandomResizedCrop(
        height=shortest_edge,  # 256
        width=shortest_edge,
        scale=(crop_fraction, 1.0),  # 0.95~1.0
        ratio=(0.9, 1.1)
    ),
    A.ColorJitter(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08
    ),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])
```

**Consistency across views**: 동일한 augmentation을 모든 카메라 뷰에 적용 (replay 메커니즘)

#### State Augmentation (Training only)

```python
# State dropout (default: 0.0)
state_dropout_prob = 0.0  # 0으로 설정 시 사용하지 않음

if training and state_dropout_prob > 0:
    do_dropout = torch.rand(B) < state_dropout_prob
    state_features = state_features * (1 - do_dropout) + mask_token * do_dropout

# State additive noise (default: 0.0)
state_additive_noise_scale = 0.0  # 0으로 설정 시 사용하지 않음

if training and state_additive_noise_scale > 0:
    noise = torch.randn_like(state_features) * state_additive_noise_scale
    state_features = state_features + noise
```

### 4.4 Normalization

**파일**: `gr00t/data/state_action/state_action_processor.py`

#### State Normalization

**방식**: Min-Max 또는 Percentile 기반

```python
# Compute statistics from dataset
stats = {
    "embodiment_tag": {
        "state": {
            "joint_positions": {
                "min": [0.1, 0.2, ...],  # Per-dimension min
                "max": [1.5, 1.8, ...],  # Per-dimension max
                "mean": [...],
                "std": [...]
            }
        },
        "action": {...}
    }
}

# Normalization formula
def normalize_state(state, stats):
    min_val = stats["min"]
    max_val = stats["max"]
    # Min-max normalization to [-1, 1]
    normalized = 2 * (state - min_val) / (max_val - min_val) - 1
    
    # Clip outliers (optional)
    if clip_outliers:
        normalized = torch.clamp(normalized, -1, 1)
    
    return normalized
```

#### Action Normalization & Relative Actions

**N1.6의 중요 변경**: **State-Relative Actions**

```python
# use_relative_action = True (N1.6 default)

# Training: Convert absolute action to relative
def compute_relative_action(action, state):
    # action: [T, D] absolute joint positions
    # state: [D] current state
    relative_action = action - state[None, :]  # Broadcast
    return relative_action

# Inference: Convert relative back to absolute
def compute_absolute_action(relative_action, state):
    absolute_action = relative_action + state[None, :]
    return absolute_action
```

**이유**: 상대 action은 일반화 성능을 향상시킴 (starting pose에 덜 민감)

---

## 5. 핵심 코드 스니펫 요약

### 5.1 모델 초기화 및 로딩

**파일**: `gr00t/model/gr00t_n1d6/gr00t_n1d6.py`

```python
from transformers import AutoModel, AutoProcessor
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

# Load pretrained model
config = Gr00tN1d6Config.from_pretrained("nvidia/GR00T-N1.6-3B")
model = AutoModel.from_pretrained(
    "nvidia/GR00T-N1.6-3B",
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Load processor (for data preprocessing)
processor = AutoProcessor.from_pretrained(
    "nvidia/GR00T-N1.6-3B",
    trust_remote_code=True
)
```

**주요 설정 파라미터**:

```python
class Gr00tN1d6Config:
    # Backbone
    model_name = "nvidia/Eagle-Block2A-2B-v2"
    backbone_embedding_dim = 2048
    tune_top_llm_layers = 4
    
    # Action head
    hidden_size = 1024
    input_embedding_dim = 1536
    action_horizon = 16
    max_state_dim = 29
    max_action_dim = 29
    
    # Diffusion
    num_layers = 32  # DiT depth
    num_attention_heads = 32
    attention_head_dim = 48
    num_inference_timesteps = 4
    
    # Training
    tune_projector = True
    tune_diffusion_model = True
    tune_llm = False
    tune_visual = False
```

### 5.2 Forward Pass (Training)

```python
def forward(self, inputs: dict):
    """
    inputs:
        - input_ids: [B, seq_len]
        - pixel_values: [B, num_imgs, 3, H, W]
        - attention_mask: [B, seq_len]
        - state: [B, max_state_dim]
        - action: [B, action_horizon, max_action_dim]
        - action_mask: [B, action_horizon, max_action_dim]
        - embodiment_id: [B]
    """
    # 1. Prepare inputs
    backbone_inputs, action_inputs = self.prepare_input(inputs)
    
    # 2. Backbone forward (Vision-Language encoding)
    backbone_outputs = self.backbone(backbone_inputs)
    # Returns:
    #   - backbone_features: [B, seq_len, 2048]
    #   - backbone_attention_mask: [B, seq_len]
    #   - image_mask: [B, seq_len]
    
    # 3. Action head forward (Action prediction)
    action_outputs = self.action_head(backbone_outputs, action_inputs)
    # Returns:
    #   - loss: scalar
    #   - action_loss: [B, T, D] (for logging)
    
    return action_outputs
```

### 5.3 Inference (Action Generation)

```python
@torch.no_grad()
def get_action(self, inputs: dict):
    """
    inputs:
        - input_ids: [B, seq_len]
        - pixel_values: [B, num_imgs, 3, H, W]
        - attention_mask: [B, seq_len]
        - state: [B, state_dim]
        - embodiment_id: [B]
    
    Returns:
        - action_pred: [B, action_horizon, action_dim]
    """
    # 1. Encode vision-language
    backbone_inputs, action_inputs = self.prepare_input(inputs)
    backbone_outputs = self.backbone(backbone_inputs)
    
    # 2. Encode state
    features = self.action_head._encode_features(backbone_outputs, action_inputs)
    state_features = features.state_features  # [B, 1, 1536]
    vl_features = features.backbone_features  # [B, seq_len, 2048]
    
    # 3. Initialize random noise
    actions = torch.randn([B, action_horizon, action_dim])
    dt = 1.0 / self.num_inference_timesteps
    
    # 4. Iterative denoising
    for t in range(self.num_inference_timesteps):
        t_cont = t / self.num_inference_timesteps
        t_discrete = int(t_cont * 1000)
        
        # Encode action
        action_features = self.action_encoder(actions, t_discrete, embodiment_id)
        
        # Concatenate with state
        sa_embeds = torch.cat([state_features, action_features], dim=1)
        
        # Run DiT
        model_output = self.diffusion_model(
            sa_embeds,
            encoder_hidden_states=vl_features,
            timestep=t_discrete
        )
        
        # Decode velocity
        pred_velocity = self.action_decoder(model_output, embodiment_id)
        pred_velocity = pred_velocity[:, -action_horizon:, :]
        
        # Update actions (Euler step)
        actions = actions + dt * pred_velocity
    
    # 5. Unnormalize and convert to absolute
    actions = processor.decode_action(actions, embodiment_tag, state)
    
    return {"action_pred": actions}
```

### 5.4 Data Preprocessing (Processor)

```python
class Gr00tN1d6Processor:
    def __call__(self, messages: list[dict]):
        """
        messages: [
            {
                "content": {
                    "images": {"camera_0": [PIL.Image, ...], ...},
                    "text": "pick up the apple",
                    "states": {"joint_positions": np.array([...])},
                    "actions": {"joint_positions": np.array([[...], ...])}
                    "embodiment": EmbodimentTag.LIBERO_PANDA
                }
            }
        ]
        """
        content = messages[0]["content"]
        
        # 1. Normalize state and action
        normalized_states, normalized_actions = \
            self.state_action_processor.apply(
                state=content.states,
                action=content.actions,
                embodiment_tag=content.embodiment
            )
        
        # 2. Image augmentation and stacking
        images = []
        for view in image_keys:
            transformed = [self.image_transform(img) for img in content.images[view]]
            images.append(torch.stack(transformed))
        stacked_images = torch.stack(images).flatten(0, 1)  # [T*V, C, H, W]
        
        # 3. Language processing
        language = content.text.lower()
        language = re.sub(r"[^\w\s]", "", language)  # Remove punctuation
        
        # 4. VLM preprocessing (tokenization)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": language},
                    *[{"type": "image", "image": img} for img in stacked_images]
                ]
            }
        ]
        text = self.processor.apply_chat_template(conversation)
        
        # 5. Padding and batching
        return {
            "vlm_content": {
                "text": text,
                "images": stacked_images,
                "conversation": conversation
            },
            "state": normalized_states,      # [state_dim]
            "action": normalized_actions,    # [action_horizon, action_dim]
            "action_mask": action_mask,      # [action_horizon, action_dim]
            "embodiment_id": embodiment_id   # scalar
        }
```

### 5.5 Training Script (Fine-tuning)

```python
# Launch command
"""
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /path/to/lerobot_dataset \
    --embodiment-tag LIBERO_PANDA \
    --modality-config-path examples/LIBERO/modality.json \
    --output-dir outputs/libero_finetune \
    --num-gpus 1 \
    --global-batch-size 32 \
    --max-steps 2000 \
    --save-steps 500 \
    --learning-rate 1e-4 \
    --use-wandb
"""

# Core training loop (simplified)
def run(config):
    # 1. Setup model and dataset
    model = Gr00tN1d6(config.model)
    processor = Gr00tN1d6Processor(config.model.processor_kwargs)
    train_dataset = ShardedMixtureDataset(config.data, processor)
    
    # 2. Setup trainer
    trainer = Gr00tTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=processor.collator
    )
    
    # 3. Train
    trainer.train(resume_from_checkpoint=config.training.start_from_checkpoint)
    
    # 4. Save
    trainer.save_model(config.training.output_dir)
    processor.save_pretrained(config.training.output_dir)
```

---

## 요약

### 모델 구조 핵심 요약

```
[Input]
  Images (multi-view, multi-timestep) → Vision Encoder (SigLIP)
  Text (task description) → Tokenizer
  State (robot joint positions) → State Encoder (embodiment-specific MLP)
                                   ↓
[Backbone: EagleBackbone]
  Vision Features + Text Tokens → Language Model (16 layers, top 4 tunable)
                                   ↓
  VL Unified Embeddings [B, seq_len, 2048]
                                   ↓
[Action Head: Gr00tN1d6ActionHead]
  State Embedding [B, 1, 1536]
  + Noisy Action Embedding [B, 16, 1536]
                                   ↓
  32-Layer Diffusion Transformer (AlternateVLDiT)
    - Cross-attention to VL features (alternating image/text)
    - Self-attention
    - Timestep conditioning (flow matching)
                                   ↓
  Action Decoder (embodiment-specific MLP)
                                   ↓
[Output]
  Velocity Prediction [B, 16, action_dim]
  → Iterative denoising (4 steps) → Final Actions
```

### 학습 특징

1. **Flow Matching Loss**: Velocity prediction in interpolated space
2. **Selective Fine-tuning**: Backbone mostly frozen, top LLM layers + DiT tunable
3. **Multi-Embodiment**: Separate projectors for each robot platform
4. **State-Relative Actions**: Better generalization across starting poses
5. **Efficient Inference**: 4 denoising steps, 22-27 Hz on RTX 4090/5090

### 주요 혁신점 (N1.6 vs N1.5)

1. **2x Deeper DiT**: 16 → 32 layers
2. **No Adapter**: Top LLM layers directly tuned (removed 4-layer adapter)
3. **Alternate VL Attention**: Separate image and text token attention
4. **Relative Actions**: Default state-relative action space
5. **Any-Resolution Vision**: Eagle VLM's flexible resolution support

이 보고서는 GR00T N1.6 모델의 전체 구현을 코드 레벨에서 분석한 내용입니다.
