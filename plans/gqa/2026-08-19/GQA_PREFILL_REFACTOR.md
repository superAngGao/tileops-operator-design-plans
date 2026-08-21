# GQA 前向算子重构详解

> 2026-08-20 决策更新：本节优先于文档中与之冲突的早期草案。

## Dense 两层 dispatch 的最终边界

Dense Op 实例只保存构造期语义配置，不保存任何一次调用的 shape。Op 层先按 target 选择并缓存 backend callable；该 callable 每次调用读取当前 `q/k/v` 的 shape/dtype，再选择并调用具体 Kernel。callable 不再读 device 选择 backend，也不把 per-call shape 写回 Op。

callable 不是为了减少透传参数，而是 backend 边界：它固定该 backend 的构造期配置，在具体 Kernel 缓存未命中时按当前 shape 选择、构造并持有 Kernel。Op 的 `kernel_map` 提供 `role → Kernel class`，callable 的缓存保存 `call signature → Kernel instance`。

RoPE 采用 caller-owned ABI：

```python
op = GroupedQueryAttentionPrefillDenseFwdOp(
    is_causal=True,
    pos_encoding_mode="rope",
    rotary_dim=64,
    rope_layout="interleaved",
)
out = op(q, k, v, rope_cos=cos, rope_sin=sin)
```

- `rope_cos` / `rope_sin` 是 manifest 中成对出现的 `optional: true` tensor inputs。
- `pos_encoding_mode` / `rotary_dim` / `rope_layout` 只声明如何解释表；Op/callable 不生成、不缓存 RoPE GPU tensor。
- table 的分配、stream readiness 和 CUDA Graph 指针生命周期归调用方管理，与 Q/K/V 相同。
- 未启用 RoPE 时两张表必须同时省略；启用时必须同时提供。

本轮不修改 generic external signature cache 的 LRU/并发/autotune 政策，不修改 MHA composite compile boundary，不重新定义 causal/square specialization region。这些不是 Dense 两层 dispatch 成立的前提。builtin callable 仅保留必要的具体 Kernel ownership，使已构造 Kernel 可被 `iter_kernels()` 和 `autotune()` 看到；TileLang cache 继续负责编译产物复用。

## 一、GQA 算子库的功能需求

对于一个完整的 Attention 算子库，GQA (Grouped Query Attention) 需要支持多个维度的变体：

### 1.1 输入数据 Layout（主要变体）

这是最大的变体维度，直接决定了 Op 的接口和实现：

#### **Dense Layout**
```python
q: [B, S_q, H, D]      # 标准的 4D tensor
k: [B, S_kv, H_kv, D]
v: [B, S_kv, H_kv, D]
```
- **特点**：batch 内所有序列共享相同的 tensor 维度
- **适用场景**：
  - Batch=1 的低延迟推理
  - Batch 内序列长度相同或相近
  - 同一 ABI 下的 dense prefill 与 contiguous decode

#### **Varlen/Packed Layout**
```python
q: [total_q, H, D]          # 所有序列 token 打包在一起
k: [total_kv, H_kv, D]
v: [total_kv, H_kv, D]
cu_seqlens_q: [B+1]          # Cumulative sequence lengths
cu_seqlens_kv: [B+1]
max_seqlen_q: int
max_seqlen_kv: int
```
- **特点**：变长序列无 padding，通过 `cu_seqlens` 分割
- **适用场景**：
  - Batch 内序列长度差异大（如 [128, 512, 1024, 2048]）
  - 避免 padding 带来的计算和内存浪费
  - 训练、离线批处理及无 paged KV cache 的 ragged attention
- **发布决策**：保留设计与内部实现价值，但暂缓作为 release-facing Op 发布

#### **Paged Layout (with KV Cache)**
```python
q: [total_q, H, D]                       # 当前 query tokens
qo_indptr: [B+1]                         # 每个 request 的 query 边界
k_pages: [num_pages, page_size, H_kv, D]  # 分页的 KV cache
v_pages: [num_pages, page_size, H_kv, D]
page_indptr: [B+1]
page_indices: [num_referenced_pages]
last_page_len: [B]
```
- **特点**：KV cache 以 page 为单位管理；同一只读 ABI 覆盖 paged prefill 与 paged decode
- **适用场景**：continuous batching、prefix cache、chunked prefill 和多用户 serving
- **边界**：调用方在 attention 前完成 KV cache 写入；Op 不 append、不修改 pages

### 1.2 其他功能维度（作为参数变体）

这些是相对较小的变体，通常作为函数参数或配置选项：

#### **KV Cache Support**
- Dense/Varlen 调用方传入本次 attention 可见的完整连续 K/V。
- Paged 调用方传入已经写好的只读 K/V pages 与 page metadata。
- KV allocation、slot assignment 和 append 均由 runtime 管理，不属于 Attention Op 契约。

#### **Causal Masking**
- `is_causal: bool` - 是否使用因果 mask（autoregressive 场景）

#### **Sliding Window**
- `window_size_left: int`
- `window_size_right: int`
- 限制 attention 范围，用于长序列场景

#### **Position Encoding**
- `pos_encoding_mode: 'none' | 'rope'`
- RoPE (Rotary Position Embedding) 可在 kernel 内部 fuse

#### **FP8 Support**
- 输入支持 `float8_e4m3fn`
- 需要额外的 scale tensors: `q_scale`, `k_scale`, `v_scale`

#### **Other Parameters**
- `sm_scale: float | None` - Attention scale（默认 `1/sqrt(D)`）
- `softcap: float | None` - Score 截断，用于稳定训练
- `dtype: torch.dtype | None` - 输出类型（FP8 输入时指定 FP16/BF16 输出）

### 1.3 Op 拆分策略

基于上述分析，我们的拆分策略是：

```
主要变体（Layout）→ 独立的 Op
次要变体（功能）  → 函数参数
```

**下一步计划发布的 2 个 Op：**
1. **`GroupedQueryAttentionDenseFwdOp`** - Dense layout；内部按 workload 选择 prefill 或 decode kernel
2. **`GroupedQueryAttentionPagedFwdOp`** - Paged layout；内部按每个 request 的 query metadata 选择 prefill 或 decode kernel

**暂缓发布：** `GroupedQueryAttentionVarlenFwdOp`。Varlen/Packed 的设计保留，但在明确训练、离线批处理或框架接入需求前不作为 release-facing Op 推进。

Prefill/decode 是同一 layout 下的 kernel family，不再体现在 public Op 名称中。每个 Op 通过参数支持 causal/non-causal、sliding window、RoPE、FP8 等功能。

**采用这种拆分的考虑：**
- Paged 以及未来可能发布的 Varlen 需要评估 metadata 缓存，以减少 CPU 端重复处理（详见[七、待讨论：Metadata 缓存机制](#七待讨论metadata-缓存机制)）
- 如果未来需要进一步收敛接口，这种拆分方式的重构成本相对较小
- 每个 layout 有独立的优化空间和性能特征

**业界参考：** FlashInfer 采用了类似的拆分策略：
- **用户接口层**：提供统一的函数式 API（如 `single_prefill_with_kv_cache()`），layout 作为参数传入
- **Wrapper 层**：按 layout 拆分为独立的 Wrapper 类
  - `BatchPrefillWithRaggedKVCacheWrapper` - Ragged/Varlen
  - `BatchPrefillWithPagedKVCacheWrapper` - Paged
- Wrapper 负责 metadata 预处理和缓存（详见[七、待讨论：Metadata 缓存机制](#七待讨论metadata-缓存机制)）

**TileOps 与 FlashInfer 的设计差异：**
- FlashInfer：用户直接调用函数，Wrapper 用于优化 batch 场景的 metadata 缓存
- TileOps：Op 类是一等公民，每个 layout 是独立的 Op 实例

---

## 二、重构前的现状与问题

### 2.1 重构前的 GQA Prefill Ops（origin/main）

```
1. GroupedQueryAttentionFwdOp                          # "Square" GQA，固定 batch/seq_len
2. GroupedQueryAttentionPrefillFwdOp                   # Varlen/Packed prefill（误导性命名）
3. GroupedQueryAttentionSlidingWindowFwdOp             # Sliding window 版本
4. GroupedQueryAttentionSlidingWindowVarlenFwdOp       # Varlen + sliding window
5. GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp   # Paged prefill
```

### 2.2 核心问题

#### 问题 1：缺少真正的 Dense Prefill Op

**现状：**
- `GroupedQueryAttentionFwdOp` 是 "square" 版本，要求固定的 `batch` 和 `seq_len`
  ```python
  # 构造时必须指定固定 shape
  op = GroupedQueryAttentionFwdOp(
      batch=4, seq_len=512,  # 固定！
      heads=32, heads_kv=8, dim=128,
  )
  # forward 时只能接受这个 shape 的输入
  out = op(q, k, v)  # q 必须是 [4, 512, 32, 128]
  ```

- `GroupedQueryAttentionPrefillFwdOp` 虽然名字叫 "Prefill"，但实际是 **Varlen/Packed** 模式
  ```python
  # 需要 cu_seqlens 和 max_seqlen
  out = op(q, k, v, cu_seqlens_q, cu_seqlens_kv, 
           q_scale, k_scale, v_scale,  # FP8 scales（强制必需！）
           max_seqlen_q, max_seqlen_kv)
  ```

**缺失的功能：**

一个真正的 **Dense Prefill Op**，需要满足：
- ✅ Dense tensor 输入 `[B, S_q, H, D]`（不需要 cu_seqlens）
- ✅ 支持**可变的 S_q 和 S_kv**（不像 square 版本要求固定）
- ✅ 支持 FP8 + optional scales（16-bit 模式不需要 scales）
- ✅ 支持 RoPE、sliding window、softcap 等高级特性
- ✅ 接口简单，batch/heads/dim 自动从输入推断

#### 问题 2：FP8 Scales 被错误声明为必需输入

**Manifest 声明（origin/main）：**
```yaml
GroupedQueryAttentionPrefillFwdOp:
  signature:
    inputs:
      q: {dtype: "float16 | bfloat16 | float8_e4m3fn"}
      q_scale: {dtype: "float32"}        # ❌ 必需输入
      k_scale: {dtype: "float32"}        # ❌ 必需输入
      v_scale: {dtype: "float32"}        # ❌ 必需输入
```

**问题：**
- Scales 只在 FP8 模式下需要
- 16-bit 模式（float16/bfloat16）不需要 scales
- 但被声明为必需输入，用户必须传入 dummy values

**期望：**
```yaml
inputs:
  q_scale: {dtype: "float32", optional: true}  # ✅ 可选
  k_scale: {dtype: "float32", optional: true}
  v_scale: {dtype: "float32", optional: true}
shape_rules:
  - "(q_scale is None) == (k_scale is None)"  # 一致性：要么都有，要么都没有
```

#### 问题 3：RoPE 处理逻辑分散

**问题：**
- RoPE 的启用方式不统一
- 有的通过动态判断 `position_ids` 是否为 None
- 有的通过 `backend` 参数隐式控制
- Kernel 选择分散在 Op 与 kernel 包装层，职责和覆盖范围不清晰

**期望：**
- 在 Op 构造时通过 `pos_encoding_mode` 显式声明：`'none'` 或 `'rope'`
- Op 首次调用时选择并缓存 target 对应的 callable
- callable 每次调用根据当前 tensor 的 shape/dtype 选择具体 kernel；不把 shape 固化在 Op 上

#### 问题 4：命名混乱，职责不清

| Op 名称 | 实际功能 | 问题 |
|---------|---------|------|
| `GroupedQueryAttentionFwdOp` | Square GQA（固定 shape） | "Fwd" 太泛，不知道是 prefill 还是 decode |
| `GroupedQueryAttentionPrefillFwdOp` | **Varlen/Packed** prefill | 名字没体现是 packed 模式，误导性强 |
| 缺失 | Dense prefill | **完全没有** |

---

## 三、Dispatch 逻辑示例

### 3.1 重构前的 Dispatch（origin/main）

`GroupedQueryAttentionFwdOp` 的 dispatch 逻辑（简化）：

```python
class GroupedQueryAttentionFwdOp(Op):
    def __init__(self, batch, seq_len, heads, heads_kv, dim, 
                 is_causal=True, ...):
        self.batch = batch
        self.seq_len = seq_len  # 固定！
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        
        # 在构造时选择 kernel
        self.dispatch_kernel(kernel_map)
    
    def dispatch_kernel(self, kernel_map):
        """根据配置选择合适的 kernel"""
        if self.is_causal and self.seq_len == self.seq_len:  # square
            self.kernel = kernel_map["gqa_prefill_square_fwd_kernel"]
        elif self.is_causal:
            self.kernel = kernel_map["gqa_prefill_causal_fwd_kernel"]
        else:
            self.kernel = kernel_map["gqa_prefill_fwd_kernel"]
    
    def forward(self, q, k, v):
        # 运行时检查 shape 是否匹配
        assert q.shape == (self.batch, self.seq_len, self.heads, self.dim)
        assert k.shape == (self.batch, self.seq_len, self.heads_kv, self.dim)
        
        # 需要将 dense tensor 转换为 packed 格式（即使是固定长度）
        device = q.device
        if device not in self._cu_seqlens:
            # 为固定长度的 batch 构造 cu_seqlens: [0, S, 2S, 3S, ..., B*S]
            self._cu_seqlens[device] = torch.arange(
                0, (self.batch + 1) * self.seq_len, self.seq_len,
                dtype=torch.int32, device=device
            )
        
        cu_seqlens = self._cu_seqlens[device]
        
        # 调用 packed kernel（即使输入是 dense）
        return self.kernel(
            q.reshape(-1, self.heads, self.dim),      # [B*S, H, D]
            k.reshape(-1, self.heads_kv, self.dim),   # [B*S, H_kv, D]
            v.reshape(-1, self.heads_kv, self.dim),
            cu_seqlens, cu_seqlens,
            self.seq_len, self.seq_len,
        ).reshape(self.batch, self.seq_len, self.heads, self.dim)
```

**问题：**
- ❌ 构造时必须指定固定的 `seq_len`
- ❌ Dense 输入被强制转换为 packed 格式
- ❌ 每次 forward 都需要 reshape 和 cu_seqlens 构造
- ❌ 不支持 S_q ≠ S_kv 的场景（如 KV cache prefill）

### 3.2 重构后的 Dispatch（本次 PR）

`GroupedQueryAttentionPrefillDenseFwdOp` 使用两级 dispatch：

```text
Op 构造期配置
    ↓
target 解析并缓存 backend callable
    ↓
每次调用 callable
    ↓
读取当前 q/k/v 的 shape、dtype
    ↓
查询具体 Kernel 缓存
    ↓
未命中时按 `Kernel.applies/refusal` 选择 role
    ↓
kernel_map[role] → 构造并缓存具体 Kernel
```

第一级只决定由哪个 target/backend 提供实现；第二级由该 backend 的 callable 自己完成 shape 相关的 kernel dispatch。Dense callable 不读取 device 来再次选择 backend，device/target 已经在上一级确定。

内置实现的结构如下：

```python
class GroupedQueryAttentionPrefillDenseFwdOp(Op):
    """Shape-agnostic BSHD GQA prefill with caller-owned optional RoPE tables."""
    
    def __init__(self,
                 is_causal: bool = True,
                 sm_scale: Optional[float] = None,
                 softcap: Optional[float] = None,
                 window_size_left: int = -1,
                 window_size_right: int = -1,
                 dtype: Optional[torch.dtype] = None,
                 pos_encoding_mode: str = 'none',
                 rotary_dim: Optional[int] = None,
                 rope_layout: str = 'neox',
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False,
                 target: Target = None):
        """保存配置参数并安装 kernel_map；不保存任何输入 shape。"""
        # 验证配置参数的语义约束
        _validate_pos_encoding_config(pos_encoding_mode, rotary_dim)
        _validate_sm_scale(sm_scale)
        
        # 保存配置参数（不保存 shape）
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        
        # 只安装 role -> kernel class 映射，不选择具体 kernel
        self.dispatch_kernel(kernel_map)
    
    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        """声明 Op 需要的 kernel 族"""
        return {
            "gqa_prefill_dense_fwd_kernel": GQAPrefillDenseFwdKernel,
            "gqa_prefill_causal_fwd_kernel": GQAPrefillFwdWsPersistentCausalKernel,
            "gqa_prefill_square_fwd_kernel": GQAFwdWsPersistentCausalKernel,
        }
    
    def _build_builtin_callable(self, target_facts):
        return DensePrefillBuiltinCallable(
            kernel_map=self.kernel_map,
            target_facts=target_facts,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            pos_encoding_mode=self.pos_encoding_mode,
            rotary_dim=self.rotary_dim,
            rope_layout=self.rope_layout,
            output_dtype=self.dtype,
            tune=self.tune,
        )

    def _eager_forward(
        self, q, k, v,
        q_scale=None, k_scale=None, v_scale=None,
        rope_cos=None, rope_sin=None,
    ):
        _validate_dense_inputs(
            q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin
        )
        inputs = _normalize_dense_inputs(
            q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin
        )

        # Cache 中保存 backend callable，而不是第一次 shape 的 kernel。
        callable = self.get_or_build_kernel(
            "gqa_prefill_dense",
            inputs,
            key=(q.device, q.dtype, self.dtype),
            build=lambda: self._build_builtin_callable(
                resolve_builtin_target_facts(q.device)
            ),
        )
        return callable(*inputs)


class DensePrefillBuiltinCallable:
    # Bounded ownership cache. It is backend-callable state, not Op dispatch
    # state: retained Kernel objects remain enumerable/autotunable. After a
    # same-signature warmup, CUDA Graph capture follows the lock-free cache-hit
    # path and does not construct a Kernel.
    specializations: OrderedDict[DenseSignature, DensePrefillKernel]

    def __call__(self, q, k, v, *optional_inputs):
        # 所有 shape 派生值均为本次调用的局部变量。
        batch, seq_len_q, heads, dim = q.shape
        _, seq_len_kv, heads_kv, _ = k.shape
        sm_scale = self.sm_scale if self.sm_scale is not None else dim**-0.5
        rotary_dim = self.rotary_dim if self.rotary_dim is not None else dim

        # call 只是本次选择所需的临时 facts，不写回 Op。
        call = dense_selection_facts(
            batch=batch,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            input_dtype=q.dtype,
            output_dtype=self.output_dtype or q.dtype,
            sm_scale=sm_scale,
            rotary_dim=rotary_dim,
            target_facts=self.target_facts,
            config=self.config,
        )
        signature = DenseSignature.from_call(call)
        kernel = self.specializations.get(signature)
        if kernel is None:
            role = select_kernel_key(DENSE_PREFILL_KEYS, call)
            kernel = self.kernel_map[role](...)
            self.specializations.put_bounded(signature, kernel)
        return kernel(q, k, v, *optional_inputs)
```

**关键设计点：**

1. **`__init__`**：只保存构造期配置并安装 `kernel_map`，不读取设备、不保存 shape
2. **Op cache**：缓存 target/backend callable；builtin 与外部 target 各自拥有具体实现策略
3. **callable**：每次读取当前 tensor 的 shape/dtype，解析 shape-dependent 默认值并选择具体 kernel
   - builtin callable 在内部完成选择，并有界持有已构造的 specialization
   - external builder 仍按 TensorSpec signature 构造 callable；Op 在调用 builder 前把 `sm_scale`、输出 `dtype`、启用 RoPE 时的 `rotary_dim` 解析为该 signature 的确定值
4. **`default_kernel_map`**：只维护 role → kernel class，不隐含优先级
5. **selection**：沿用 #1896 的 `Kernel.applies/refusal` + `select_kernel_key` 契约；顺序不决定结果，重叠区域必须报歧义
6. **缓存边界**：Op 只缓存 backend callable；callable 有界持有已经构造的具体 Kernel，使 `iter_kernels()`、`autotune()` 和 CUDA Graph 生命周期可见。TileLang cache 继续负责底层编译产物复用

**与重构前的对比：**
- ❌ 重构前：构造时固定 shape，forward 时验证 shape 是否匹配
- ✅ 重构后：构造时只保存配置，cached callable 每次从输入推断 shape

**遵循的规则：**
- ✅ 配置参数在构造时验证
- ✅ Shape 相关的选择位于 backend callable 内
- ✅ 可选输入的语义约束在公共 Op 边界验证
- ✅ `batch/seq_len/dim/resolved_scale` 等 per-call facts 不写回 Op
- ✅ callable 不重新探测 target/device
- ✅ `rope_cos` / `rope_sin` 是 caller-owned optional tensors；Op/callable 不生成、不缓存，也不接管其 stream/CUDA Graph 生命周期
- ✅ Op 不维护 shape → kernel cache；有界 specialization ownership 属于 backend callable
- ✅ 临时 selection facts 只用于一次选择，不写回 Op；只有稳定的 construction signature 进入 callable 的有界缓存
- ✅ cache miss 的 Kernel 构造发生在正常调用/显式预热阶段；已构造 Kernel 可被枚举和预先 autotune，不能把首次调优推迟到 CUDA Graph capture

### 3.3 功能 Dispatch 对比

| 功能 | 重构前 | 重构后 |
|------|--------|--------|
| **输入格式** | Dense → Packed 转换 | 直接接受 Dense |
| **Shape 约束** | 构造时固定 batch/seq_len | 构造时无 shape，forward 时推断 |
| **FP8 scales** | 强制必需 | 可选 + 运行时验证 |
| **RoPE** | 运行时判断 | 构造时确定解释方式；tables 由调用方作为 optional tensors 提供 |
| **Kernel 选择** | 部分在构造时 | target callable 缓存；具体 kernel 按每次调用的 shape/dtype 选择 |
| **S_q ≠ S_kv** | 不支持 | ✅ 支持 |
| **动态 shape** | 不支持（需创建新实例）| ✅ 支持（同实例处理多种 shape）|

---

## 四、本次 PR 的边界

### 4.1 本次 PR 完成的工作

本次 PR（#1926）专注于 **Dense Prefill** 的重构：

#### 1. 创建 `GroupedQueryAttentionPrefillDenseFwdOp`

**核心功能：**
- ✅ Dense tensor 输入 `[B, S_q, H, D]`
- ✅ 支持可变的 S_q 和 S_kv
- ✅ FP8 + optional scales
- ✅ Fused RoPE（通过 `pos_encoding_mode='rope'`）
- ✅ Causal/non-causal masking
- ✅ Sliding window attention
- ✅ Softcap support
- ✅ Torch.compile fullgraph 支持

**Manifest 改进：**
- ✅ 将 `q_scale`, `k_scale`, `v_scale` 声明为 `optional: true`
- ✅ 添加 `shape_rules` 验证 scales 的一致性
- ✅ 修复可空参数的类型声明（`float | None`）
- ✅ 添加 `torch_compile_fullgraph: true` 声明

**测试覆盖：**
- ✅ 16-bit 模式（float16/bfloat16）
- ✅ FP8 模式 + scales
- ✅ Causal/non-causal
- ✅ RoPE fused
- ✅ Sliding window
- ✅ 可选输入的边界条件
- ✅ Compile contract 测试

#### 2. 后端 Kernel 整合

**新增/整合的 kernels：**
- `GQADensePrefillKernel` - 基础 dense prefill
- `GQADensePrefillCausalKernel` - Causal dense prefill
- `GQADensePrefillRoPEKernel` - Dense prefill with fused RoPE
- `GQADensePrefillFP8Kernel` - FP8 dense prefill

#### 3. Benchmark 和测试更新

- ✅ 更新 `benchmarks/ops/attention/bench_gqa.py`
- ✅ 添加 FP8 dense prefill benchmark cases
- ✅ 覆盖 Llama 3.1 8B/70B 的典型 workload
- ✅ 添加 sliding window、softcap 等高级特性的测试

#### 4. 文档和标注

- ✅ 在代码中标注旧 Op 为 deprecated
- ✅ 添加迁移指南（inline comments）
- ⚠️ 保留 `GroupedQueryAttentionFwdOp` 作为 legacy adapter

### 4.2 本次 PR 的边界

**本次 PR 不包含：**

❌ **Packed/Varlen Prefill 的重构**
- `GroupedQueryAttentionPrefillFwdOp` 保持不变；新的 release-facing Varlen Op 暂缓

❌ **Paged Prefill 的重构**
- `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` 保持不变

❌ **Decode Ops**
- `GroupedQueryAttentionDecodeWithKVCacheFwdOp` 等不在本次范围

❌ **移除 Legacy Adapter**
- `GroupedQueryAttentionFwdOp` 保留作为兼容层

❌ **Sliding Window 独立 Ops**
- `GroupedQueryAttentionSlidingWindowFwdOp` 等保持不变（功能已整合到新 Op 的参数中）

### 4.3 核心变更文件

```
src/tileops/ops/attention/gqa.py                   # 新增 PrefillDenseFwdOp
src/tileops/kernels/attention/gqa_dense_prefill.py # Dense prefill kernels
src/tileops/kernels/attention/gqa_fwd_fp8.py       # FP8 kernel 整合
src/tileops/manifest/attention.yaml                # Manifest 更新
tests/ops/attention/test_gqa.py                    # 测试更新
tests/test_compile.py                              # Compile contract 注册
benchmarks/ops/attention/bench_gqa.py              # Benchmark 更新
```

---

## 五、后续计划

### 5.1 Phase 2: Dense Prefill/Decode 合并（下一步）

**目标：** 将 Dense prefill 与 contiguous decode 收敛为 `GroupedQueryAttentionDenseFwdOp`。

**主要工作：**
1. 统一 Dense public ABI，移除 public 名称中的 phase。
2. builtin callable 每次读取当前 shape 和静态 metadata，在 dense prefill/decode kernel family 中选择具体 Kernel。
3. 合并 causal、window、RoPE、softcap、16-bit/FP8 的 manifest、test 和 benchmark 覆盖。
4. 对相同 workload 做迁移前后 correctness、latency 和 compile/CUDA Graph 回归。

### 5.2 Phase 3: Paged Prefill/Decode 合并

**目标：** 将 paged prefill 与 paged decode 收敛为 `GroupedQueryAttentionPagedFwdOp`。

**主要工作：**
1. 统一只读 K/V pages ABI，以及 `qo_indptr`、`page_indptr`、`page_indices`、`last_page_len` 的语义。
2. callable 根据每个 request 的 query metadata、batch、KV length 和其他构造事实选择 prefill/decode kernel；不把 phase 暴露为独立 Op。
3. 保留独立 prefill/decode kernel family，并统一 FP8、RoPE、window、softcap 等能力矩阵。
4. cache append 和 mutation 明确在 Op 外完成，本计划不增加 `append` 参数。

### 5.3 Varlen 发布暂缓

Varlen/Packed 的连续 ragged topology 仍有训练和离线批处理价值，但不是当前在线 serving 的发布主线。本阶段：

- 不新增或重命名 release-facing Varlen Op；
- 不为了接口对称扩展独立 Varlen specialization；
- 保留必要的现有实现和研究结果；
- 等明确的框架消费者、workload 与性能目标后再决定是否发布 `GroupedQueryAttentionVarlenFwdOp`。

### 5.4 Phase 4: 移除 Legacy Adapters

**目标：** 清理历史遗留代码

**主要工作：**
1. **迁移所有调用方**
   - 更新内部代码使用新 Ops
   - 提供迁移脚本和文档
   - 发布 deprecation warning（至少 2 个版本）

2. **移除 Legacy Ops**
   ```
   ❌ GroupedQueryAttentionFwdOp                    # 移除
   ❌ GroupedQueryAttentionSlidingWindowFwdOp       # 移除（功能已整合）
   ```

   Varlen 相关 legacy API 在发布决策前不纳入本阶段清理。

3. **文档更新**
   - 更新 API 文档
   - 添加迁移指南
   - 发布 changelog

### 5.5 Phase 5: 性能优化（持续）

**基于清晰边界的优化机会：**

1. **Dense 优化**
   - 针对固定 shape 的 fast path（编译时优化）
   - 更激进的 kernel fusion（如 softmax + output scaling）
   - FP8 Tensor Core 充分利用

2. **Paged 优化**
   - Page 访问的 locality 优化
   - Prefetch 策略
   - Prefill/decode kernel family 的 workload dispatch

3. **Cross-cutting 优化**
   - Flash Attention 3 后端整合
   - Custom CUDA kernel 优化
   - Torch.compile AOT 编译

---

## 六、总结

### 6.1 重构核心思想

```
Layout 是主要变体 → 拆分为独立 Ops
功能是次要变体   → 统一为函数参数
```

**当前发布目标：**
```
1. GroupedQueryAttentionDenseFwdOp  - Dense [B, S, H, D]，统一 prefill/decode
2. GroupedQueryAttentionPagedFwdOp  - Paged KV cache，统一 prefill/decode
```

`GroupedQueryAttentionVarlenFwdOp` 暂缓发布。每个已发布 Op 通过参数支持 causal、sliding window、RoPE、FP8、softcap 等功能，prefill/decode 只作为内部 kernel family 存在。

### 6.2 本次 PR 的价值

✅ **职责清晰**：Dense Prefill 有了专门的 Op  
✅ **接口简化**：可选输入 + 自动推断  
✅ **性能优化**：RoPE 前移 + compile contract  
✅ **类型安全**：Manifest + runtime 双重验证  
✅ **可扩展性**：为后续 Dense/Paged phase 合并奠定基础

### 6.3 对用户的影响

**Migration Path：**
```python
# 旧代码（Square GQA）
op = GroupedQueryAttentionFwdOp(batch=4, seq_len=512, heads=32, heads_kv=8, dim=128)

# 新代码（Dense Prefill）- Shape-agnostic
op = GroupedQueryAttentionPrefillDenseFwdOp(is_causal=True)
# batch/heads/heads_kv/seq_len/dim 自动从输入推断

# 带配置参数的示例
op = GroupedQueryAttentionPrefillDenseFwdOp(
    is_causal=True,
    sm_scale=0.125,
    softcap=2.0,
    window_size_left=256,
    pos_encoding_mode='rope',
    rotary_dim=64
)
```

**新功能解锁：**
- ✅ S_q ≠ S_kv（KV cache prefill）
- ✅ FP8 native support（无需 dummy scales）
- ✅ Fused RoPE（性能更好）
- ✅ Torch.compile 友好
- ✅ 动态 shape 支持（同一个 Op 实例处理不同 shape）

---

这次重构是 GQA 算子库走向成熟的关键一步，为后续的功能扩展和性能优化打下了坚实的基础。

---

## 七、待讨论：Metadata 缓存机制

### 7.1 问题背景

#### 本次 PR 的 callable 与编译缓存边界

本次 Dense 设计区分两种缓存：

1. `Op.get_or_build_kernel` 保存 target/backend 返回的 callable；builtin 使用不含动态 shape 的 key，使同一个 callable 可以服务多个 Dense shape。
2. callable 每次根据当前 tensor metadata 选择具体 kernel；相同具体配置的编译产物由 TileLang 自身的编译缓存复用。

```text
Op cache:        target/config → backend callable
每次调用:        shape/dtype → kernel role
Callable cache:  bounded construction signature → concrete Kernel object
TileLang cache:  concrete kernel config → compiled program
```

本次增加的是 callable 内部的**有界 Kernel ownership cache**，不是把最后一次
调用的 shape 写回 Op，也不是缓存 Varlen/Paged 的运行期 metadata。它解决的是
Kernel 生命周期问题：枚举、autotune、预热和 CUDA Graph capture 必须看到同一个
已构造对象；TileLang cache 只复用编译产物，不能替代这层对象所有权。

#### 对于 Dense Layout：无需运行期 metadata cache

```python
# Dense GQA
op = GroupedQueryAttentionPrefillDenseFwdOp(is_causal=True)

# Layer 1
output = op(q, k, v)  # shape = [B, S, H, D]
# → callable 选择 kernel；TileLang 编译或命中编译缓存

# Layer 2-80
output = op(q, k, v)  # 相同 shape
# → callable 直接命中已构造的 Kernel object
```

**开销分析**：
- ✅ Kernel 编译：只做一次
- ✅ Kernel 选择：只在新 call signature 的缓存未命中时执行
- ✅ 参数验证：每层都做，但开销很小（shape 检查）
- ✅ Metadata：几乎没有（只有 shape）

**结论**：对于 Dense layout，轻量 runtime dispatch + 有界 Kernel ownership cache +
TileLang 编译缓存足够；暂不引入 cu_seqlens、block table 等运行期 metadata cache。

#### 对于 Varlen/Packed Layout：Kernel 缓存不够

```python
# 假设未来发布 Varlen GQA
op = GroupedQueryAttentionVarlenFwdOp(is_causal=True)

# Layer 1
output = op(q, k, v, 
            cu_seqlens_q, cu_seqlens_kv,      # [B+1] 累积索引
            max_seqlen_q=128, max_seqlen_kv=256)
# → callable 选择 kernel；TileLang 编译或命中编译缓存
# → 验证 cu_seqlens 的一致性
# → 传输 cu_seqlens 到 GPU（如果在 CPU）
# → Kernel 内部解析 cu_seqlens

# Layer 2-80（相同 batch）
output = op(q, k, v,
            cu_seqlens_q, cu_seqlens_kv,      # 重复传递
            max_seqlen_q=128, max_seqlen_kv=256)
# → 命中 kernel 缓存 ✅
# → 但每层都要：
#    - 重复验证 cu_seqlens ❌
#    - 重复传输到 GPU（如果需要）❌
#    - Kernel 内部重复解析 ❌
```

**开销分析**：
- ✅ Kernel 编译：相同具体配置命中 TileLang 编译缓存
- ✅ Kernel 选择：每次执行轻量 metadata dispatch
- ❌ Metadata 验证：每层都做（80 次）
- ❌ Metadata 传输：每层都可能发生（80 次）
- ❌ Metadata 解析：kernel 内部每次都解析（80 次）

**结论**：对于 Varlen/Packed layout，kernel 缓存解决了编译问题，但 **metadata 处理开销仍然存在**。

#### 对于 Paged Layout：问题更严重

Paged layout 的 metadata 更复杂：
- `page_table: [B, max_blocks]` - 逻辑到物理 page 的映射
- `cu_seqlens: [B+1]` - 累积序列长度
- `kv_indptr` - Page 索引映射
- Address translation logic

每层都要传输、验证、解析这些 metadata，开销更大。

### 7.2 业界方案：FlashInfer 的 Metadata 缓存

FlashInfer 通过 **Wrapper 类**提供显式的 metadata 缓存：

```python
# 创建 wrapper
wrapper = BatchPrefillWithPagedKVCacheWrapper(...)

# 一次性预处理 metadata 并缓存
wrapper.begin_forward(
    kv_indptr=indptr,
    kv_page_indices=page_indices,
    kv_last_page_lens=last_lens
)

# 多层 attention 共享缓存的 metadata
for layer in transformer_layers:
    output = wrapper.forward(q)  # 不需要重复传递 metadata

# 清理缓存
wrapper.end_forward()
```

**优化效果**：
- ✅ Kernel 编译：只做一次
- ✅ Metadata 验证和传输：只做一次
- ✅ Metadata 预处理：只做一次
- 多层 transformer 的 metadata 开销从 O(layers) 降到 O(1)

**代价**：
- 用户需要手动管理 `begin_forward` / `end_forward`
- 容易忘记调用，导致错误

### 7.3 待讨论议题

#### 议题 1：是否支持 Metadata 缓存？

**支持的理由**：
- 优化多层 transformer 的 metadata 处理开销（特别是 Varlen/Paged）
- 与 kernel 缓存互补，形成完整的优化链
- 业界实践（FlashInfer）证明了有效性

**不支持的理由**：
- 增加 API 复杂度
- Dense layout 不需要（已经够快）
- 实现和维护成本
- TileOps 设计哲学：Op 应该尽量无状态

**需要讨论**：
- 优先级如何？是否在本次 PR 考虑？
- 是否只为 Varlen/Paged 提供，Dense 不需要？

#### 议题 2：如果支持，采用哪种方案？

**方案 A：显式缓存（手动管理）**

```python
class GroupedQueryAttentionVarlenFwdOp(Op):
    def prepare_batch(self, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
        """显式准备并缓存 batch metadata"""
        self._cached_metadata = self._preprocess_metadata(
            cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
        )
    
    def __call__(self, q, k, v):
        """使用缓存的 metadata"""
        if self._cached_metadata is None:
            raise RuntimeError("Must call prepare_batch() before forward")
        kernel = self.get_or_build_kernel("fwd", ...)
        return kernel(q, k, v, self._cached_metadata)
    
    def clear_batch(self):
        """清理缓存"""
        self._cached_metadata = None


# 用户代码
op = GroupedQueryAttentionVarlenFwdOp(is_causal=True)

op.prepare_batch(cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
for layer in transformer_layers:
    output = op(q, k, v)  # 使用缓存的 metadata
op.clear_batch()
```

**优点**：
- 性能可预测，生命周期明确
- 类似 FlashInfer 的 `begin_forward()` / `end_forward()`
- 无内存泄漏风险

**缺点**：
- 用户需要记住调用 `prepare_batch()` / `clear_batch()`
- 不符合 PyTorch 的直觉
- API 相对繁琐

---

**方案 B：Context Manager（显式缓存 + 自动清理）**

```python
class GroupedQueryAttentionVarlenFwdOp(Op):
    @contextmanager
    def batch_scope(self, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
        """Context manager for batch metadata caching"""
        self._cached_metadata = self._preprocess_metadata(
            cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
        )
        try:
            yield
        finally:
            self._cached_metadata = None
    
    def __call__(self, q, k, v, 
                 cu_seqlens_q=None, cu_seqlens_kv=None,
                 max_seqlen_q=None, max_seqlen_kv=None):
        """支持两种模式：带 metadata 或使用缓存"""
        if self._cached_metadata is None:
            # 模式 1：单次调用，每次传 metadata
            if cu_seqlens_q is None:
                raise ValueError("Must provide cu_seqlens_q or use batch_scope()")
            metadata = self._preprocess_metadata(
                cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
            )
        else:
            # 模式 2：使用缓存的 metadata
            metadata = self._cached_metadata
        
        kernel = self.get_or_build_kernel("fwd", ...)
        return kernel(q, k, v, metadata)


# 用户代码 - 模式 1：单次调用（无优化）
op = GroupedQueryAttentionVarlenFwdOp(is_causal=True)
output = op(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)

# 用户代码 - 模式 2：批量优化（多层 transformer）
with op.batch_scope(cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    for layer in transformer_layers:
        output = op(q, k, v)  # metadata 已缓存
# 自动清理
```

**优点**：
- 两种模式并存：简单场景直接调用，优化场景用 context manager
- 自动清理（Pythonic）
- 向后兼容：不用优化时 API 不变
- 生命周期清晰

**缺点**：
- API 稍复杂（两种调用模式）
- 需要文档明确说明使用场景

---

**方案对比**：

| 维度 | 方案 A（手动管理）| 方案 B（Context Manager）|
|------|------------------|-------------------------|
| **易用性** | 需要记住 prepare/clear | Context manager 自动清理 |
| **灵活性** | 只支持缓存模式 | 支持有/无缓存两种模式 |
| **向后兼容** | 不兼容（必须 prepare）| 兼容（可选优化）|
| **实现复杂度** | 低 | 中等 |
| **风险** | 用户可能忘记调用 | API 有两种模式，需要文档说明 |

### 7.4 当前建议

**短期（本次 PR #1926）：**
- ❌ **不实现** metadata 缓存
- ✅ 保持当前的 API 设计（每次传递 metadata）
- ✅ 在文档中说明这是已知的优化空间

**中期（团队讨论后）：**
- 讨论是否需要 metadata 缓存
- 如果需要，选择方案 A 或方案 B
- 评估实现成本和收益

**长期（如果决定支持）：**
- **推荐方案 B（Context Manager）**
  - 原因：向后兼容 + 自动清理 + 灵活性
- 优先为 Paged Op 评估 metadata 缓存；Varlen 仅在未来决定发布后纳入
- Dense Op 不需要（已经足够高效）
