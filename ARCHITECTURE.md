# Llama-3 + mHC 架构（llama3-mhc）

## 组件：

**所需class**：LlamaHCConfig、RMSNorm、LlamaAttention、LlamaMLP、LlamaHCBlock、LlamaHC；混合层 SinkhornMHCResidual（静态拓扑用）。

**层归一化方法**：RMSNorm。Pre-Norm 架构，舍弃均值计算与偏置项 $\beta$，保留可学习缩放 $\gamma$；内部以 float32 计算后回投原 dtype，$\epsilon = 10^{-5}$。

**序列混合器**：Attention（分组查询注意力 GQA）。

**通道混合器**：SwiGLU FFN。

**流混合器（本架构核心扩展）**：mHC 流形约束超连接混合。$n=4$ 条并行残差流，逐 token 动态注记（读 $\mathbf{h}_{\text{pre}}$ / 写 $\mathbf{h}_{\text{post}}$，mHC Eq.7）+ 逐 token Sinkhorn 双随机混合矩阵（mHC Eq.8-9，20 次交替归一化）。

**激活函数**：3 种。SiLU（SwiGLU 门控）、Softmax（注意力）、Sigmoid（mHC 动态注记门控；Sinkhorn 归一化本身不是激活函数）。

**位置编码**：全维度 RoPE，底数 $\theta_{\text{base}} = 500{,}000$，仅作用于 $Q$、$K$；实现采用 HF `rotate_half` 约定（与 Meta 的复数配对约定相差一个通道置换，数学等价）。

**偏置项**：主干变换层（Attention、MLP、RMSNorm 及嵌入/解嵌入投影）全网无偏置；偏置仅存在于 mHC 的静态映射注记中（$\mathbf{b}_{\text{pre}}, \mathbf{b}_{\text{post}}, \mathbf{b}_{\text{res}}$，mHC 论文 Eq.7 定义）。

**Dropout**：3 处（默认全部 0.0）。残差流嵌入融合后、注意力权重（Softmax）后（SDPA `dropout_p`）、MLP 输出投影后。注意：mHC 混合与写入路径上无 Dropout。

**嵌入解嵌入权重绑定**：开启（本仓前提；Llama-3 8B 同款）。`lm_head.weight` 与 `wte.weight` 共享同一参数。

**优化器**：AdamW（$\beta_1=0.9,\ \beta_2=0.95$），矩阵参数 weight decay $=0.1$，其余（偏置类、一维参数、mHC 原始参数 `mixer.raw`/`theta`/`b_res`/`b_post`）decay $=0$；梯度裁剪 1.0；学习率 warmup + 余弦退火。

**forward函数**：5 个。RMSNorm.forward、LlamaAttention.forward、LlamaMLP.forward、LlamaHCBlock.forward、LlamaHC.forward（另有两个 mHC 辅助前向：`_dyn_raw`/`_dyn_stage`，以及混合层 `SinkhornMHCResidual.mixing_matrices`）。

**Attention**：4 个无偏置线性层：$W_Q \in \mathbb{R}^{d \times d}$，$W_K, W_V \in \mathbb{R}^{d \times d_{\text{kv}}}$（$d_{\text{kv}} = n_{\text{kv\_heads}} \cdot d_{\text{head}}$，GQA），$W_O \in \mathbb{R}^{d \times d}$。

**MLP/FFN**：3 个无偏置线性层（SwiGLU）：$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$，$W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$，其中 $d_{\text{ffn}}$ 由 $\text{round\_to\_multiple}(\lfloor \tfrac{8}{3}d \cdot \text{ffn\_dim\_multiplier} \rceil,\ 64)$ 给出（Meta 参考顺序：先乘 multiplier、后对齐倍数）。

---

## LlamaHCConfig：

    block_size: int = 256              # 最大上下文 S_max
    vocab_size: int = 65               # shakespeare_char 字符级词表
    n_layer: int = 6
    n_head: int = 6
    n_kv_heads: int = 0                # 0 -> n_head（普通 MHA）；须整除 n_head
    n_embd: int = 384
    multiple_of: int = 64              # SwiGLU 隐藏维对齐倍数
    ffn_dim_multiplier: float = 1.0
    rope_theta: float = 500000.0       # Llama-3 值（Llama-2 为 10000）
    dropout: float = 0.0
    n_streams: int = 4                 # mHC 残差流数 n（论文同款 hc_mult=4）
    b_res_init: float = 0.0            # b_res 对角标度；0 -> sqrt(n)（默认），调大（如 4.0）可得更严格的初始恒等
    mixer: str = "sinkhorn"            # sinkhorn（mHC 论文方法）| none（裸 Llama-3）
    dynamic_topology: bool = True      # 逐 token 动态注记（mHC Eq.7）；默认开启
    tie_weights: bool = True           # 输入/输出嵌入绑定；默认开启

---

### 训练：

###### 无偏置全连接线性变换层（后简称线性层/矩阵）：

输入张量 $X \in \mathbb{R}^{B \times T \times d_{\text{in}}}$，在最后一个通道维度与权重矩阵 $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ 相乘，输出 $(B, T, d_{\text{out}})$：
$$Y = X W$$

###### RMSNorm.forward：

独立作用于每个 $(b, t)$ 的通道维 $d$，以 float32 计算均方根（LLaMA-3 同款，$\epsilon=10^{-5}$）：
$$\text{RMS}(X) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} X_{b, t, i}^2 + \epsilon}$$
$$Y = \frac{X}{\text{RMS}(X)} \odot \gamma, \quad \gamma \in \mathbb{R}^d \text{（可学习，无 } \beta\text{）}$$

###### 旋转位置编码 RoPE（HF rotate_half 实现）：

**第一步**：角频率预计算。对头维度偶通道索引 $i \in [0, \frac{d_{\text{head}}}{2}-1]$：
$$\theta_i = \theta_{\text{base}}^{-\frac{2i}{d_{\text{head}}}}, \quad \text{freqs}_{t,i} = t \cdot \theta_i$$

**第二步**：HF 约定拼接。将 $\text{freqs}$ 自身水平拼接为 $(T, d_{\text{head}})$ 矩阵并取余弦/正弦，得到 $\cos, \sin \in \mathbb{R}^{T \times d_{\text{head}}}$（复数形式 $\text{freqs\_cis}_{t,i} = e^{j\,t\theta_i}$ 与之等价，仅通道配对约定不同：Meta 配对相邻通道，HF 配对 $i$ 与 $i + \frac{d_{\text{head}}}{2}$）。

**第三步**：旋转注入。把 $x \in \mathbb{R}^{d_{\text{head}}}$ 拆成两半 $x = (x^{(1)}, x^{(2)})$，定义 $\text{rotate\_half}(x) = (-x^{(2)}, x^{(1)})$，则
$$\tilde{x} = x \odot \cos + \text{rotate\_half}(x) \odot \sin$$
仅对 $Q$、$K$ 施加；对任意一对位置的注意力得分只依赖相对位移（已由 RoPE 范数保持 + 相对位移不变性数值验证，至 float32 噪声）。

###### repeat_kv（GQA）：

当 $n_{\text{kv\_heads}} < n_{\text{heads}}$ 时，将 $K/V$ 在头维复制 $n_{\text{rep}} = n_{\text{heads}} / n_{\text{kv\_heads}}$ 倍：$(B, n_{\text{kv}}, T, d_{\text{head}}) \to (B, n_{\text{heads}}, T, d_{\text{head}})$（expand + reshape，无拷贝语义等价）。

###### LlamaAttention.forward：

**第一步**：线性投影。$Q = X W_Q \in (B,T,n_{\text{heads}},d_{\text{head}})$，$K = X W_K,\ V = X W_V \in (B,T,n_{\text{kv\_heads}},d_{\text{head}})$（代码中 reshape + 转置到 $(B, n_h, T, d_h)$）。

**第二步**：RoPE 注入。$\tilde{Q} = \text{RoPE}(Q),\ \tilde{K} = \text{RoPE}(K)$；$V$ 不旋转。

**第三步**：GQA 广播。$\tilde{K}, V \xleftarrow{\text{repeat\_kv}}$ 头维对齐至 $n_{\text{heads}}$。

**第四步**：缩放点积注意力（SDPA，一次算子内完成得分、因果掩码、Softmax、加权汇聚与注意力 Dropout）：
$$Y_{\text{head}} = \text{Dropout}\left(\text{Softmax}\left(\frac{\tilde{Q}\tilde{K}^\top}{\sqrt{d_{\text{head}}}} + \text{Mask}\right)\right)V, \quad \text{Mask 为下三角因果掩码（is\_causal=True）}$$

**第五步**：多头拼接与输出投影：
$$y = Y_{\text{concat}} W_O \in \mathbb{R}^{B \times T \times d}$$

###### LlamaMLP.forward（SwiGLU）：

**第一步**：门控与增维并行投影：$h_{\text{gate}} = x W_{\text{gate}},\ h_{\text{up}} = x W_{\text{up}} \in \mathbb{R}^{d_{\text{ffn}}}$。

**第二步**：SiLU 门控融合：$h_{\text{mid}} = \text{SiLU}(h_{\text{gate}}) \odot h_{\text{up}}$。

**第三步**：降维投影 + Dropout（第 3 处）：$y = \text{Dropout}(h_{\text{mid}} W_{\text{down}})$。

###### 残差流构造（mHC，LlamaHC.forward 第一步）：

本架构不使用单一残差向量，而是维护 $n=4$ 条并行残差流张量 $S \in \mathbb{R}^{B \times T \times n \times d}$：

**第一步**：词嵌入查表 $\text{tok} = \text{wte}(\text{idx}) \in \mathbb{R}^{B \times T \times d}$。

**第二步**：流初始化。全部 $n$ 条流以词嵌入的副本初始化（官方 `expand_to_mhc` 约定，TileKernels / mHC.cu 同款）：
$$S_{b,t,k,:} = \text{tok}_{b,t,:}, \quad \forall k \in [0, n-1]$$

**第三步**：嵌入 Dropout（第 1 处）后送入 $N_{\text{layer}}$ 个 Block 串行更新：
$$S^{(l)} = \text{Block}_l(S^{(l-1)}), \quad l \in [1, N_{\text{layer}}]$$

###### LlamaHCBlock.forward（动态拓扑，mHC Eq.7 + Eq.8-9；默认路径）：

每个 Block 含两个阶段（attn、mlp），阶段内完全同构。记当前流为 $S \in \mathbb{R}^{B \times T \times n \times d}$：

**第一步**：注记源（Eq.7）。将整束流展平为 $n \cdot d$ 维向量并做 RMSNorm（论文："flatten ... to preserve full context information"；$\gamma$ 可学习）：
$$\vec{S}_{b,t} = \text{vec}(S_{b,t}) \in \mathbb{R}^{n d}, \quad \tilde{x} = \text{RMSNorm}(\vec{S}) \in \mathbb{R}^{n d}$$

**第二步**：融合投影与注记（Eq.7）。单一融合投影 $\Theta \in \mathbb{R}^{n(n+2) \times nd}$（无偏置）一次产出全部注记混合，再由可学习门控标量 $\alpha$（论文附录：初始化 $0.01$）与偏置 $\mathbf{b}$ 生成三组原始注记——无 tanh（tanh 只出现在 Eq.5 的 HC 预备式，mHC 的 Eq.7 是直接线性投影）：
$$\text{mixes} = \tilde{x}\,\Theta^\top \in \mathbb{R}^{n(n+2)}$$
$$\tilde{\mathcal{H}}^{\text{pre}} = \alpha_{\text{pre}} \cdot \text{mixes}_{[:n]} + \mathbf{b}_{\text{pre}}, \quad \tilde{\mathcal{H}}^{\text{post}} = \alpha_{\text{post}} \cdot \text{mixes}_{[n:2n]} + \mathbf{b}_{\text{post}}, \quad \tilde{\mathcal{H}}^{\text{res}} = \text{mat}\!\left(\alpha_{\text{res}} \cdot \text{mixes}_{[2n:]}\right) + \mathbf{b}_{\text{res}}$$
偏置初始化：$\mathbf{b}_{\text{pre}} = \text{logit}(1/n)$（均匀读），$\mathbf{b}_{\text{post}} = 0$，$\mathbf{b}_{\text{res}} = \frac{n}{\sqrt{n}} I$（恒等主导）。

**第三步**：注记激活（Eq.8）。读注记、写注记、混合矩阵分别经 Sigmoid、$2\times$Sigmoid、Sinkhorn-Knopp 投影：
$$\mathbf{h}_{\text{pre}} = \sigma\!\left(\tilde{\mathcal{H}}^{\text{pre}}\right) \in (0,1)^n, \quad \mathbf{h}_{\text{post}} = 2\sigma\!\left(\tilde{\mathcal{H}}^{\text{post}}\right) \in (0,2)^n$$
$$W = \text{Sinkhorn-Knopp}\!\left(\tilde{\mathcal{H}}^{\text{res}}\right)$$

**第四步**：读汇聚（$\mathcal{H}^{\text{pre}}$ 即聚合权重）：
$$v = \sum_{k=1}^{n} \mathbf{h}_{\text{pre},k} \cdot S_{:,:,k,:} \in \mathbb{R}^{B \times T \times d}$$

**第五步**：子层计算（先 RMSNorm 后子层）：
$$f = \text{SubLayer}(\text{RMSNorm}(v)) \quad (\text{attn 阶段为 LlamaAttention，mlp 阶段为 LlamaMLP})$$

**第六步**：流形投影混合（Eq.8-9）。混合矩阵先经元素级指数与上界截断（$e^{20}$ 内防溢出，mHC.cu 同款），再逐 token 做 20 次交替归一化（Sinkhorn-Knopp，向 Birkhoff 多面体投影）——**每次迭代先行归一化、后列归一化**（与代码一致），20 次迭代后末次操作为列归一化，故代码在循环外再做一次行归一化收尾，保证行和精确为 1：
$$H^{(0)} = \exp\!\left(\text{clamp}_{\max}\!\left(\tilde{\mathcal{H}}^{\text{res}},\ 20\right)\right), \quad H^{(t)} = T_c\!\left(T_r\!\left(H^{(t-1)}\right)\right) \ (t = 1, \dots, 20), \quad W = T_r\!\left(H^{(20)}\right)$$
$$\text{其中 } T_r(M)_{jk} = \frac{M_{jk}}{\sum_{k'} M_{jk'}}, \quad T_c(M)_{jk} = \frac{M_{jk}}{\sum_{j'} M_{j'k}}$$
（行/列先后次序在收敛域内数学等价，仅影响有限迭代的数值残差；本初始化下 20 步已使行/列和误差为 0，但训练中 $\tilde{\mathcal{H}}^{\text{res}}$ 逐 token 变化，末次行归一化作为保证不可省略。）$W$ 严格双随机：$\sum_k W_{jk} = 1$（精确），$\sum_j W_{jk} = 1$（至 float32 噪声，实测 $\le 10^{-7}$），且 $W_{jk} \ge 0$（凸组合混合）。此流形约束恢复残差流的恒等映射性质。

**第七步**：流混合。逐 token 以 $W$ 混合 $n$ 条流：
$$S'_{:,:,j,:} = \sum_{k=1}^{n} W_{:,,:,j,k} \cdot S_{:,:,k,:}$$

**第八步**：写入（Eq.3 语义）。子层输出 $f$ 以逐流标度 $\mathbf{h}_{\text{post}}$ 广播写入全部 $n$ 条流：
$$S^{\text{out}}_{:,:,j,:} = S'_{:,:,j,:} + \mathbf{h}_{\text{post},j} \cdot f$$

**初始状态性质**：$W\big|_{\text{init}} \approx I$（$\mathbf{b}_{\text{res}}=\sqrt{n}I$ 经指数+Sinkhorn 后强对角占优，初始对角元素 $\approx 0.71$、非对角 $\approx 0.096$；调大 `b_res_init`（如 $4.0$）可得更严格的初始恒等，对角 $\approx 0.95$）；$\mathbf{h}_{\text{pre}}\big|_{\text{init}} \approx \frac{1}{n}\mathbf{1}$（$\mathbf{b}_{\text{pre}}=\text{logit}(1/n)$，$\alpha$ 极小）；$\mathbf{h}_{\text{post}}\big|_{\text{init}} \approx 2\sigma(0) = 1$（$\mathbf{b}_{\text{post}}=0$，即子层输出以满幅写入所有流，非零写入）。初始行为近似标准残差：各流均为词嵌入副本、均匀读出得 $\text{tok}$、$f$ 全幅写回——与官方实现的恒等出发点一致。

###### LlamaHC.forward（后续步骤）：

**最终读出**。全部 Block 结束后，以可学习读出向量 $\mathbf{r}$（初始化 $\frac{1}{\sqrt{n}}\mathbf{1}$，均匀读）将 $n$ 条流汇聚回单流：
$$x = \sum_{k=1}^{n} \mathbf{r}_k \cdot S^{(N_{\text{layer}})}_{:,:,k,:} \in \mathbb{R}^{B \times T \times d}$$

**最终层归一化**：$x_{\text{final}} = \text{RMSNorm}_f(x)$。

**解嵌入投影（权重绑定）**：
$$\text{Logits} = x_{\text{final}} W_{\text{head}}^\top, \quad W_{\text{head}} = \text{wte.weight} \in \mathbb{R}^{V \times d}$$

**损失计算**。若传入 targets（$(B,T)$），展平后计算交叉熵：
$$\mathcal{L} = \text{CrossEntropy}(\text{Logits}, \text{targets}), \quad \text{ignore\_index}=-1$$

**返回**：$(\text{Logits}, \mathcal{L})$ 二元组；若未传 targets，仅对最后一个时间步计算 Logits（$(B, 1, V)$），损失为 None。

### 拓扑与混合器变体：

* **静态拓扑**（`--no_dynamic_topology`）：$\mathbf{h}_{\text{pre}}, \mathbf{h}_{\text{post}}, \Omega$ 不再逐 token 生成，替换为每阶段可学习的静态读/写向量（初始化 $\frac{1}{\sqrt{n}}\mathbf{1}$）与静态混合矩阵 $W = \text{Sinkhorn}_{20}(\exp(\text{raw}))$（`SinkhornMHCResidual`，raw 初始化 $\mathcal{N}(0, 0.5)$）。
* **裸基线**（`--mixer=none`）：强制 $n=1$、静态拓扑、无混合矩阵，读写向量初始化 $1$ —— 严格退化为无超连接的裸 pre-norm Llama-3，作教学/对照开关。

---

### 推理（本仓实现：朴素整段重前向，无 KV Cache）

本仓为参考实现，`generate` 采用朴素自回归：每步对整个前缀重算前向（$S_{\text{max}}=256$ 代价可接受），未实现 KV Cache（Meta 工程版缓存 $\mathbf{K}_{\le t}, \mathbf{V}_{\le t}$ 逐层复用；这是工程优化，数学等价，列入 Roadmap）。

* 当前已生成序列 $\mathbf{idx} \in \mathbb{N}^{1 \times t'}$（$t' \le S_{\text{max}}$，超长则截取末尾 $S_{\text{max}}$ 个）。

**第一步**：整段前向（训练同款，Dropout 关闭），仅取最后位置的 Logits：$\mathbf{z} = \text{Logits}_{:, -1, :} \in \mathbb{R}^{V}$。

**第二步**：Top-K 截断（默认 $k=40$）。保留得分最高的 $k$ 个候选，其余置 $-\infty$：
$$z_j = \begin{cases} z_j & j \in \text{TopK}(k) \\ -\infty & \text{否则} \end{cases}$$

**第三步**：温度缩放采样：
$$P(i_{t'+1}) = \text{Softmax}\left(\frac{\mathbf{z}}{\max(T_{\text{emp}}, 10^{-8})}\right), \quad i_{t'+1} \sim P$$

**第四步**：拼接 $i_{t'+1}$，重复直至生成 `max_new_tokens` 个。采样入口 `sample_hc.py` 另提供 `--chat` REPL（逐轮 prompt → 续写）与 `--seed`/`--temperature`/`--top_k`。

---

## 与 LLaMA-3 官方实现的差异（全部刻意，均有依据）：

| 项 | LLaMA-3 官方 | 本仓 | 原因 |
|---|---|---|---|
| 嵌入绑定 | 8B 绑定 / 70B 独立 | 全配置绑定 | 仓库前提（小模型参数效率） |
| RoPE 实现约定 | 复数配对（相邻通道） | HF rotate_half（配对 $i, i{+}d_h/2$） | 数学等价，与 HF 参考 diff 对齐 |
| 激活统计 | 无 Dropout | 3 处 Dropout（默认 0.0） | nanoGPT 训练器惯例，保留开关 |
| 初始化 | 官方未公开细节 | 全线性层/嵌入 $\mathcal{N}(0, 0.02)$ | nanoGPT 小模型惯例（README 已声明） |
| QK-norm | 3.1+ 特性 | 无 | 保持 Llama-3.0 忠实（README 已声明） |
| 推理 | KV Cache + 张量并行 | 朴素重前向，单卡 | 参考实现优先可读性 |
| Sinkhorn 反向 | 自定义 autograd Function（只存输入、backward 重算） | 原生 Autograd（2×20 次归一化激活图，约数百 MB 量级） | 当前规模开销可忽略；规模化（长上下文/大 batch/AMP）时换 `torch.compile` 或 TileKernels 式自定义 backward |
| 残差结构 | 单流残差 | **mHC：$n=4$ 流 + 流形约束混合** | 本仓库的存在意义（arXiv:2512.24880） |
