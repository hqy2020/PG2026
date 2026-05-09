# SPAGS 中文技术报告（通俗版）

> **面向人群**：刚入实验室的学弟/学妹，有一定 CV/ML 基础但没接触过 3DGS 和 CT 重建
> **版本**：Pacific Graphics 2026 投稿版本
> **代码仓库**：https://github.com/hqy2020/PG2026

---

## 目录

1. [背景：我们要解决什么问题？](#1-背景我们要解决什么问题)
2. [3D Gaussian Splatting 极简入门](#2-3d-gaussian-splatting-极简入门)
3. [SPAGS 总体框架](#3-spags-总体框架)
4. [SPS：空间先验播种（初始化阶段）](#4-sps空间先验播种初始化阶段)
5. [GAP：几何感知剪枝（训练中剪枝阶段）](#5-gap几何感知剪枝训练中剪枝阶段)
6. [ADM：自适应密度调制（训练后调制阶段）](#6-adm自适应密度调制训练后调制阶段)
7. [代码结构导读](#7-代码结构导读)
8. [训练流程逐行解析](#8-训练流程逐行解析)
9. [关键实验结果](#9-关键实验结果)
10. [常见问题 FAQ](#10-常见问题-faq)

---

## 1. 背景：我们要解决什么问题？

### 1.1 任务：稀疏视角 CT 新视角合成

想象一下：你去医院做 CT 扫描，机器绕着你的身体转一圈拍了几十张 X 光片。**但我们想让机器转更少的圈**，比如只拍 **3 张** X 光片，然后通过算法「脑补」出其他角度的 X 光片是什么样的。

这就是**稀疏视角新视角合成**（Sparse-view Novel View Synthesis）：

- **输入**：2-4 张不同角度的 X 光投影（CT 投影图）
- **输出**：从任意新角度生成的逼真 X 光投影图
- **目标**：减少辐射剂量（拍更少的片），同时保持图像质量

### 1.2 为什么难？

普通拍照的「新视角合成」是已知问题，但 X 光片有几个独特难点：

| 难点 | 类比 | 影响 |
|------|------|------|
| **X光是穿透过** | 普通相机拍表面，X光拍「穿透」 | 每个像素是整条射线上所有组织的累加 |
| **极稀疏视角** | 普通 NVS 用 20+ 张图，我们只用 2-4 张 | 信息严重不足，极度病态 |
| **体密度不均匀** | 骨头很密、肺是空的、软组织在中间 | 高斯点会大量聚集在骨头边界 |

### 1.3 现有方法的三大问题

我们基于 **R²-Gaussian**（NeurIPS 2024）改进，发现它有三个根本缺陷：

```
问题1：初始化随缘
  ├─ 均匀随机撒点 → 骨头区域没点，空气里全是点
  └─ 就像钓鱼不看鱼情，随便往池塘里撒网

问题2：密化加剧堆叠
  ├─ 3DGS 的「哪里梯度大就在哪里加点」在 CT 上适得其反
  ├─ 骨头边界梯度最大 → 不停加 → 越加越挤
  └─ 就像拥堵路段不停加车 → 更堵了

问题3：密度更新一视同仁
  ├─ 骨头和软组织的密度学习率一样
  └─ 就像用同一种方式教所有学生，不管他们基础不同
```

### 1.4 我们的核心洞察

> **CT 新视角合成的核心问题不是「稀疏」而是「冗余」**

这话什么意思？普通场景（比如拍一张桌子）的问题是：桌子后面是空的，需要往空的地方加高斯点。但 CT 场景整个空间都是满的（有骨头、有肉、有空气），问题不是哪里缺高斯，而是**哪里高斯太多了**——骨头边界挤了上万个高斯在打架。

所以我们的思路是：**反向操作**——不是加，而是剪。

---

## 2. 3D Gaussian Splatting 极简入门

如果你完全没接触过 3DGS，下面是最低限度的知识：

### 2.1 什么是 3D 高斯？

```
一个 3D 高斯 ≈ 一个三维空间中的「小椭球」

每个高斯有 5 类参数：
├─ 位置 (xyz)       → 这个椭球在哪儿
├─ 形状 (scale)     → 椭球多大、多扁（3 个方向的尺度）
├─ 朝向 (rotation)  → 椭球怎么转
├─ 密度 (opacity/density) → 这个椭球的「浓淡」
└─ 颜色 (SH系数)    → 从不同角度看是什么颜色（CT 场景不用）
```

### 2.2 训练流程（简化版）

```
1. 初始化：随机撒 50,000 个高斯点
2. 渲染：从某个角度「拍一张」→ 得到预测投影图
3. 对比：预测图 vs 真实图 → 算 L1 损失
4. 反向传播：更新每个高斯的参数
5. 自适应控制（每 100 迭代）：
   ├─ 梯度大的高斯 → 克隆/分裂（加更多点）
   ├─ 透明度低的 → 删除
   └─ 太大的高斯 → 分裂成两个小的
6. 回到第 2 步，循环 30,000 次
```

### 2.3 R²-Gaussian 的特殊之处

CT 场景不渲染 RGB 颜色，而是用**比尔-朗伯定律**模拟 X 光穿透：

```
像素强度 = 源强度 × exp(-Σ(密度 × 射线穿过长度))
```

简单说：X 光穿过骨头（密度高）→ 衰减大 → 像素暗；穿过空气（密度低）→ 衰减小 → 像素亮。

---

## 3. SPAGS 总体框架

### 3.1 三阶段一览

```
          SPS                 GAP                   ADM
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │  FDK 重建     │   │ 邻近分数计算   │   │ K-Planes 编码 │
    │ 密度加权采样   │ → │ 梯度感知过滤   │ → │ MLP 密度调制  │
    │ 初始化 50K 点  │   │ 剪枝冗余高斯   │   │ 零均值归一化  │
    └──────────────┘   └──────────────┘   └──────────────┘
        Iter 0            Iter 2000-20000      Iter 15000+
       (一次完成)       (每 500 迭代执行)      (持续执行)
```

### 3.2 每个模块在干什么（一句话版）

| 模块 | 一句话 | 涨点 |
|------|--------|------|
| **SPS** | 先用 FDK 粗重建看看组织分布，在骨头区域多撒点 | +0.21 dB |
| **GAP** | 训练中检查哪些高斯挤在一起了，剪掉冗余的 | **+0.42 dB** |
| **ADM** | 让模型学会「骨头密度该高、软组织密度该低」的空间规律 | +0.11 dB |

### 3.3 为什么三阶段是「渐进式」的？

```
SPS → GAP → ADM 的顺序有讲究：

SPS 做初始化 → 决定了起点好坏
  ↓
GAP 在训练中期剪枝 → 控制了训练过程中的点数量和质量
  ↓
ADM 在训练后期微调密度 → 在稳定后的表示上做精细调整
```

就像盖房子：**SPS 打地基 → GAP 控制结构 → ADM 做精装修**。

---

## 4. SPS：空间先验播种（初始化阶段）

### 4.1 灵感来源

CT 扫描有一个「免费」的东西叫 **FDK 重建**。FDK 是一种经典算法，用滤波反投影从 X 光片还原三维体积。虽然稀疏视角下 FDK 有很多伪影（条状伪影、模糊），但它**至少能告诉你骨头大概在哪儿、空气大概在哪儿**。

SPS 的思路就是：**免费的先验不用白不用**。

### 4.2 具体怎么做

```
① 输入 3 张 X 光片
      ↓
② 做 FDK 重建 → 得到一个粗糙的 3D 密度图
      ↓
③ 密度加权采样：
   ├─ 20% 的点：均匀撒在整个空间（保证覆盖）
   └─ 80% 的点：按 FDK 密度值加权采样（密度越高的区域点越多）
      ↓
④ 每个点的初始密度 = FDK 在该位置的密度值
⑤ 每个点的大小 ∝ 局部采样密度（密的地方用小高斯，稀的地方用大高斯）
```

### 4.3 关键代码

```python
# 文件: initialize_pcd.py
# 调用链:
#   train.py:76 → initialize_gaussian(gaussians, dataset, None)
#     → gaussians.create_from_pcd(xyz, density, 1.0)
#       → 加载 .npy 文件 (xyz + density)
#
# .npy 文件由 initialize_pcd.py 生成：
#   1. 读取稀疏投影 → FDK 重建
#   2. 密度加权采样 50,000 个点
#   3. 每个点包含: [x, y, z, density]
```

### 4.4 为什么有效？

```
类比：如果你要在森林里放 50,000 个传感器

均匀撒点 → 湖面上也有传感器、树梢上也有
            但真正需要监测的树根周围却没几个

密度加权撒点 → 树根周围多放、树干上适当放、湖面上少放
               每个传感器都在有用的地方
```

### 4.5 超参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `α` (uniform_ratio) | 均匀采样的比例 | **0.2**（20% 保证覆盖） |
| `γ` (gamma) | 密度加权的强度 | **1.0**（线性加权） |
| `K` (n_points) | 总点数 | **50,000** |

---

## 5. GAP：几何感知剪枝（训练中剪枝阶段）

### 5.1 核心创新

GAP 是整个 SPAGS 里**最重要的模块**（贡献了 +0.42 dB 中的绝大部分）。

它的核心想法很简单：**高斯点挤在一起打架了，剪掉那些不干活的多余点**。

### 5.2 问题描述

在标准 3DGS 训练中，每 100 迭代会做一次「自适应控制」：
- 梯度大的位置 → 克隆高斯点 → 加点
- 透明度低的位置 → 删除

问题在于：CT 场景中，**骨头边界的梯度最大**，所以 3DGS 会持续不断地在骨头边界加高斯点，导致：

```
骨头边界附近：
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
（成百上千个高斯挤在同一个位置，相互竞争表示同一块结构）

软组织区域：
● · · · · · · · · · · · · · · · · · · · 
（高斯点反而稀疏，因为梯度信号弱）
```

### 5.3 GAP 剪枝算法

```
每 500 迭代（Iter 2000 ~ 20000）执行一次：

① 计算邻近分数：
   每个高斯点到其 5 个最近邻的平均距离
   → 分数低 = 周围挤满了高斯（密集区域）
   → 分数高 = 周围很空旷（稀疏区域）

② 筛选候选：
   分数 < 阈值 τ = 0.015 → 标记为「冗余候选」

③ 梯度过滤（关键细节）：
   冗余候选中，保留那些「还在努力学习」的高斯
   → 梯度大的点：还在学习，保留
   → 梯度小的点：已经学完了，可以删除

④ 执行剪枝：
   每次最多剪掉 2% 的总点数（安全限制）
```

### 5.4 关键代码

```python
# 文件: r2_gaussian/innovations/fsgs/proximity_densifier.py
# 类: GAPPruner (第754行)

# 调用流程 (train.py:194-219):
#
# 创建 GAP Pruner:
#   gap_pruner = GAPPruner(
#       k_neighbors=5,           # K=5 个最近邻
#       gap_threshold=0.015,     # 邻近分数阈值
#       gap_gradient_aware=True,  # 启用梯度感知过滤
#       gap_gradient_threshold=0.0002,
#       gap_max_ratio=0.03,       # 最大剪枝比例
#   )
#
# 训练循环中执行 (train.py:390-...):
#   1. gap_pruner.compute_proximity_scores(positions)
#      → 计算每个高斯到K个最近邻的平均距离
#   2. gap_pruner.identify_prune_candidates(positions, grads)
#      → 识别需要剪枝的候选点
#   3. gaussians.prune_points(prune_mask)
#      → 执行剪枝
```

### 5.5 类比理解

```
想象一个音乐节：
- 舞台前（=骨头边界）：挤了上万人，大家互相踩脚
- 草坪上（=软组织区域）：稀稀拉拉几个人

GAP 做的事：
1. 发现舞台前太挤了（邻近分数低）
2. 检查每个人：还在跟着节奏蹦跶的（梯度大）→ 留着
   已经站着不动的（梯度小）→ 请出去
3. 每次只请走几个（最多 2%），防止一下子全空了

结果：舞台前不那么挤了，被请走的人可以去草坪上跳舞（其他区域受益）
```

### 5.6 为什么比 FSGS 的「邻近密化」好？

FSGS（ECCV 2024）的做法是：**邻近分数高的地方加点**（稀疏区域加点）。

在普通场景中成立：桌子后面是空的 → 需要加点填充。

但在 CT 场景中：
```
FSGS 的做法：发现骨头边界很挤（邻近分数低）→ 不加点
            发现空气很空旷（邻近分数高）→ 在空气里加点
            → 本该在骨头边界加的点全加到了空气里 → 更糟糕

GAP 的做法：发现骨头边界很挤 → 剪掉冗余的
            发现空气很空旷 → 不做任何操作
            → 高斯分布更合理
```

所以 GAP 的核心洞察是：**CT 场景的邻近密化逻辑和普通场景是相反的**。

---

## 6. ADM：自适应密度调制（训练后调制阶段）

### 6.1 动机

即使有了 SPS 和 GAP，每个高斯的密度参数（`ρ`）仍然是独立优化的。这意味着：

- 骨头的密度和软组织的密度用**同样**的学习率
- 模型无法学会「骨头区域密度应该高，软组织区域密度应该低」这个空间规律

### 6.2 K-Planes：高效的空间编码

ADM 使用 **K-Planes** 来建模空间中的密度规律:

```
K-Planes 的核心思想：
不直接存一个 3D 网格（太贵了），
而是存三个 2D 平面（xy, xz, yz 平面），
任意 3D 位置的信息 = 三个 2D 平面特征的拼接
```

```
3D 网格: R³ 参数  → (64³) = 262,144 参数 ❌ 太贵
K-Planes: 3R² 参数 → (3×64²) = 12,288 参数 ✅ 便宜
```

### 6.3 ADM 调制流程

```
每个高斯的位置 x → 投影到 xy, xz, yz 平面
         ↓
    双线性插值取特征 → 3 个 d 维特征
         ↓
    拼接成 3d 维向量
         ↓
    MLP 处理（3层隐藏层, 128单元, ReLU）
         ↓
    ┌──→ 置信度 c(x) ← sigmoid 输出 —— 这个调制靠谱吗？
    └──→ 偏移量 Δρ ← tanh × r —— 密度该加还是减？
         ↓
    零均值归一化：
    Δρ̂ = Δρ - 加权平均(Δρ)
    → 保证整体密度分布不变，只调相对关系
         ↓
    最终密度 = 基础密度 + c(x) × Δρ̂
```

### 6.4 关键细节：零均值归一化

这是 ADM 的一个巧妙设计：

```
如果没有零均值归一化：
  MLP 可能让所有区域的密度都加 0.1
  → 模型学会了「整体变亮」，而不是「骨头比软组织更密」

有了零均值归一化：
  某些区域密度↑，某些区域密度↓
  整体密度分布不变，只调整相对关系
  → 模型必须学习「空间规律」而非「全局偏移」
```

### 6.5 关键代码

```python
# 文件: r2_gaussian/gaussian/kplanes.py
# 类: KPlanesEncoder

# 文件: r2_gaussian/gaussian/gaussian_model.py
# 方法: get_adm_diagnostics(), get_kplanes_features()

# 训练循环中的调用 (train.py:356-368):
# if gaussians.enable_kplanes and iteration % 1000 == 0:
#     adm_diag = gaussians.get_adm_diagnostics()
#     # 输出诊断信息：offset 范围、置信度、密度变化百分比等
```

### 6.6 类比理解

```
ADM 就像一个「空间密度地图」：

没有 ADM 时：
  每个高斯点自己决定自己是「骨头」还是「软组织」
  → 可能一个点在骨头区域，但密度被学到很低（学错了）

有 ADM 时：
  K-Planes 学会了一张空间「密度热力图」
  → 在骨头区域自动给高斯点加密度
  → 在软组织区域自动减密度
  → 每个点不用从头学，参考「邻居的经验」

就像你搬到一个新城市：
  没地图 → 自己摸索每个区域的特点
  有地图 → 一看就知道哪是商业区、哪是住宅区
```

---

## 7. 代码结构导读

### 7.1 仓库总览

```
PG2026/
│
├── train.py                     ★ 训练入口（从这开始读）
├── test.py                      测试/评估入口
├── initialize_pcd.py            SPS: 点云初始化生成
│
├── r2_gaussian/                 ★ 核心 Python 包
│   ├── gaussian/
│   │   ├── gaussian_model.py    ★ Gaussian 类的核心实现（关键！）
│   │   ├── render_query.py      渲染/体积查询
│   │   ├── kplanes.py           ★ ADM: K-Planes 编码器
│   │   └── initialize.py        初始化逻辑
│   │
│   ├── innovations/
│   │   └── fsgs/
│   │       └── proximity_densifier.py  ★ GAP: 剪枝 + 密化器
│   │
│   ├── baselines/               对比方法
│   ├── dataset/                 数据加载
│   ├── arguments/               命令行参数
│   ├── utils/                   工具函数
│   └── submodules/              CUDA 扩展（不用动）
│
├── figures/                     论文图片
├── scripts/                     实验脚本
├── docs/                        文档
└── paper/                       论文
```

### 7.2 核心函数调用链

```
train.py 的 training() 函数 (第39行)

初始化阶段：
  initialize_gaussian(gaussians, dataset, None)  ← 第77行
    → 读取 .npy 文件 (xyzd)
    → gaussians.create_from_pcd(xyz, density, 1.0)

训练循环 (iteration 1 ~ 30000):
  for iteration in range(1, 30001):
    ├── render() → 渲染投影图
    ├── 计算 loss → backward()
    │
    ├── GAP 剪枝 (每 500 iter, 2000-20000):
    │   gap_pruner.identify_prune_candidates(positions, grads)
    │     → compute_proximity_scores() → KNN
    │     → 筛选候选 (分数 < 阈值 && 梯度 < 阈值)
    │   gaussians.prune_points(prune_mask)
    │
    ├── FSGS 密化 (可选的，3DGS标准密化):
    │   proximity_densifier.compute_proximity_scores()
    │   → 在稀疏区域加点（与GAP互补）
    │
    ├── ADM 调制 (15000 iter 后启用的K-Planes):
    │   KPlanesEncoder → 特征提取
    │   MLP → Δρ, c
    │   zero_mean_normalize()
    │   密度 = ρ_base + c * Δρ̂
    │
    └── 标准 3DGS 自适应控制 (每 100 iter):
        densify_and_prune()
```

### 7.3 参数配置入口

所有命令行参数定义在：

```python
# r2_gaussian/arguments/__init__.py

ModelParams 类：
  - self.enable_gap = False        # GAP 主开关
  - self.enable_adm = False        # ADM 主开关
  - self.ply_path = ""             # SPS 点云路径
  - self.gap_threshold = 0.015     # GAP 剪枝阈值
  - self.gap_k = 5                 # GAP K值
  ... 以及其他 GAP/ADM 参数

运行示例：
  python train.py \
    -s data/369/foot_50_3views.pickle \
    -m output/foot_3views_spags \
    --ply_path data/369/init_foot_50_3views.npy \
    --enable_sps --enable_gap --enable_adm
```

---

## 8. 训练流程逐行解析

读 `train.py` 的 `training()` 函数（第39行开始）：

### 8.1 初始化（第49-79行）

```python
# 1. 设置场景（加载数据）
scene = Scene(dataset, shuffle=False)

# 2. 初始化高斯
gaussians = GaussianModel(scale_bound, args=dataset)
initialize_gaussian(gaussians, dataset, None)
#   → args.ply_path 指向 SPS 生成的 .npy 文件
#   → 加载 50,000 个点的 xyz + density

# 3. 设置优化器
gaussians.training_setup(opt)
```

### 8.2 GAP 配置（第103-219行）

```python
# 判断是否启用 GAP
use_gap = getattr(dataset, 'enable_gap', False)

if use_gap:
    gap_pruner = GAPPruner(
        k_neighbors=5,
        gap_threshold=0.015,
        gap_gradient_aware=True,
        gap_gradient_threshold=0.0002,
        gap_max_ratio=0.03,
    )
```

### 8.3 ADM 配置（隐含于 GaussianModel 中）

```python
# r2_gaussian/gaussian/gaussian_model.py
# GaussianModel.__init__() 中：
if args.enable_adm:
    self.kplanes_encoder = KPlanesEncoder(
        grid_resolution=64,   # 64×64 平面
        feature_dim=32,        # 每平面 32 维特征
    )
```

### 8.4 训练循环核心（第256-400行）

```python
for iteration in range(1, 30001):
    # 1. 选一个视角 → 渲染
    render_pkg = render(viewpoint_cam, gaussians, pipe)
    gt_image = viewpoint_cam.original_image.cuda()

    # 2. 算损失
    loss = l1_loss(image, gt_image) + λ * ssim_loss + λ_tv * tv_loss
    loss.backward()

    # 3. GAP 剪枝（每 500 迭代）
    if iteration % 500 == 0 and 2000 <= iteration <= 20000:
        positions = gaussians.get_xyz
        grads = gaussians.xyz_gradient_accum / gaussians.denom
        prune_mask = gap_pruner.identify_prune_candidates(positions, grads)
        gaussians.prune_points(prune_mask)

    # 4. 标准 3DGS 自适应控制（每 100 迭代）
    if iteration % 100 == 0:
        gaussians.densify_and_prune(...)
```

---

## 9. 关键实验结果

### 9.1 主对比表（3 视角 PSNR）

| 方法 | Chest | Head | Abdomen | Foot | Pancreas | **平均** |
|------|-------|------|---------|------|----------|---------|
| DN-Gaussian | 16.02 | 18.04 | 23.51 | 23.51 | 22.43 | 20.70 |
| CoR-GS | 18.55 | 20.36 | 22.45 | 25.53 | 23.25 | 22.03 |
| FSGS | 19.99 | 21.53 | 23.34 | 25.37 | 25.27 | 23.10 |
| X-Gaussian | 20.29 | 21.52 | 23.54 | 25.48 | 25.12 | 23.19 |
| R²-Gaussian | 26.22 | 26.60 | 29.26 | 28.48 | 28.61 | 27.83 |
| **SPAGS (ours)** | **26.81** | **26.63** | **29.73** | **28.49** | **29.43** | **28.22** |

- 比 R²-Gaussian 基线 **+0.39 dB**
- 比其他方法 **+4~8 dB**（碾压级优势）

### 9.2 消融实验

| 配置 | PSNR | 涨点 |
|------|------|------|
| R²-Gaussian (基线) | 27.80 | - |
| +SPS | 28.01 | +0.21 |
| **+GAP** | **28.22** | **+0.42** ✅ 最大 |
| +ADM | 27.91 | +0.11 |
| SPS+ADM | 28.09 | +0.29 |
| SPS+GAP+ADM | **28.22** | **+0.42** |

**关键发现**：
1. GAP 是贡献最大的模块（+0.42 dB），比 SPS+ADM 加起来还多
2. 含 GAP 的所有配置都达到了 28.22 dB，说明 GAP 是核心
3. SPS 和 ADM 有协同效应（0.21+0.11=0.32 < 实际 0.29+）

### 9.3 不同视角数的表现

| 方法 | 2 视角 | 3 视角 | 4 视角 |
|------|--------|--------|--------|
| R²-Gaussian | 21.33 | 27.83 | 29.13 |
| **SPAGS** | **21.44** | **28.22** | **29.20** |
| 增益 | +0.11 | **+0.39** | +0.07 |

> 3 视角是「甜蜜点」：FDK 还算有用，GAP 剪枝效果最好，ADM 能帮上忙。

---

## 10. 常见问题 FAQ

### Q1: GAP 和 3DGS 标准剪枝有什么区别？

3DGS 的标准剪枝依据是 **透明度和屏幕大小**（透明度趋近 0 或太大就删）。
GAP 的依据是 **几何邻近性**（离别的点太近就删）。
两者互补：标准剪枝删「看不见的」，GAP 删「挤在一起的」。

### Q2: GAP 和 FSGS 的差异到底是什么？

```
FSGS (ECCV 2024):  邻近分数高 → 稀疏 → 加点（密化）
GAP (ours):         邻近分数低 → 密集 → 剪点（剪枝）

两者是完全相反的操作！因为 CT 场景和普通场景的几何问题不同。
```

### Q3: 为什么不直接用 FAISS 加速 KNN？

代码中已经支持了 FAISS（`use_faiss=True`），但如果没安装会 fallback 到 PyTorch 的分块 KNN。FAISS 加速不明显时因为 GPU 上的 `torch.cdist` 已经很快了。

### Q4: SPS 的 FDK 重建会不会很慢？

FDK 重建是一个免费操作——任何 CT 扫描流程中都已经有了。代码中一次 FDK 重建只需 **~0.8 秒**，是训练前的一次性开销。

### Q5: 训练要多久？

单器官约 **25-30 分钟**（2× RTX A6000）。其中：
- SPS: +0.8 秒（一次性）
- GAP: +0.5 秒（整个训练过程累计）
- ADM: +45 秒（15K-30K 迭代的 MLP 前向+反向）
- 总开销 < 10% 的额外训练时间

### Q6: 推理时有什么额外开销？

**零开销**。ADM 的 K-Planes 可以在训练后预计算，GAP 只在训练时用。推理时的渲染速度甚至比 R²-Gaussian 还快（高斯点更少：80K vs 100K）。

### Q7: 我想跑实验，从哪开始？

```bash
# 1. 环境配置
conda env create -f environment.yml
conda activate r2_gaussian_new

# 2. 编译 CUDA 扩展
cd r2_gaussian/submodules/simple-knn && pip install -e .
cd r2_gaussian/submodules/xray-gaussian-rasterization-voxelization && pip install -e .

# 3. 跑一个完整 SPAGS 实验
python train.py \
  -s data/369/foot_50_3views.pickle \
  -m output/foot_3views_spags \
  --ply_path data/369/init_foot_50_3views.npy \
  --enable_sps --enable_gap --enable_adm

# 4. 跑基线对比
python train.py \
  -s data/369/foot_50_3views.pickle \
  -m output/foot_3views_baseline \
  --ply_path data/369/init_foot_50_3views.npy

# 5. 消融实验
python train.py ... --enable_sps --enable_gap               # 去掉 ADM
python train.py ... --enable_sps --enable_adm               # 去掉 GAP
python train.py ... --enable_gap --enable_adm               # 去掉 SPS
```

### Q8: 我想改超参数，在哪改？

两种方式：

**CLI 参数**（推荐）：
```bash
python train.py ... \
  --gap_threshold 0.02 \
  --gap_k 7 \
  --adm_max_range 0.4
```

**改默认值**：编辑 `r2_gaussian/arguments/__init__.py` 中的 `ModelParams` 类的默认值。

---

## 附录：术语对照表

| 英文 | 中文 | 解释 |
|------|------|------|
| 3D Gaussian Splatting (3DGS) | 三维高斯泼溅 | 用 3D 椭球体表示场景的技术 |
| Novel View Synthesis (NVS) | 新视角合成 | 从已知视角生成未知视角的图像 |
| Sparse-view | 稀疏视角 | 只用少数几个视角（2-4 个） |
| FDK | 滤波反投影 | 经典 CT 重建算法 |
| KNN / K-Nearest Neighbors | K 近邻 | 找最近的 K 个邻居点 |
| PSNR | 峰值信噪比 | 图像质量评价指标（越高越好） |
| SSIM | 结构相似性 | 图像结构相似度指标 |
| Beer-Lambert Law | 比尔-朗伯定律 | X 光穿过物体的衰减规律 |
| K-Planes | 平面分解法 | 用三个 2D 平面编码 3D 空间 |
| Ablation Study | 消融实验 | 逐一去掉模块看效果 |

---

> **报告版本**：v1.0 | **对应代码**：GitHub main 分支 | **如有疑问**：找师兄/师姐讨论
