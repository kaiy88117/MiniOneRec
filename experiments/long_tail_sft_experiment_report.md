# MiniOneRec Long-tail-aware SFT 实验记录

## 1. 实验目标

本实验基于 MiniOneRec 的 SID-only SFT 训练链路，引入 long-tail-aware weighted loss，目标是在不显著牺牲 HR@10 / NDCG@10 的前提下，提高长尾 item 的曝光比例和长尾目标召回能力。

项目面向腾讯搜推算法岗求职展示，重点体现：

- 生成式推荐 SFT 闭环复现能力
- 数据分布分析能力
- 长尾推荐问题定位能力
- loss 层面的最小侵入式改造能力
- 离线评估与 ablation 分析能力
- 工程 debug 和问题修复能力

---

## 2. 数据集与 baseline

数据集：

- Category: `Industrial_and_Scientific`
- Train file: `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- Valid file: `data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- Test file: `data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- SID index: `data/Amazon/index/Industrial_and_Scientific.index.json`
- Item info: `data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt`

模型与训练配置：

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset mode: `sid_only`
- Epochs: 3
- Batch size: 64
- Micro batch size: 4
- Learning rate: `5e-5`
- Cutoff length: 512
- GPU: RTX 4090D / RTX 5090 class single card

Baseline 指标：

| Metric | Value |
|---|---:|
| HR@10 | 0.1297 |
| NDCG@10 | 0.0913 |
| Coverage@10 | 0.1851 |
| Tail Ratio@10 | 0.0213 |
| Tail Recall@10 | 0.0288 |
| Invalid Rate | 0.0000 |

Baseline 结论：

普通 SFT 可以获得一定的准确率，但推荐列表明显偏向头部 item。Top-10 推荐中长尾 item 占比仅约 2.13%，真实长尾目标的 Top-10 召回约 2.88%，说明存在明显长尾召回不足问题。

---

## 3. 长尾划分方法

根据训练集中的 `item_id` 出现频次统计 item popularity，并按 item 数量比例划分：

- Head items: 频次排序前 20%
- Middle items: 中间 60%
- Tail items: 后 20%

Industrial_and_Scientific 训练集统计：

| Group | Item Count | Interaction Share |
|---|---:|---:|
| Head | 约 20% | 55.44% |
| Middle | 约 60% | 39.30% |
| Tail | 约 20% | 5.26% |

这说明少量头部 item 占据了过半交互，而尾部 item 交互非常稀疏，适合尝试 popularity-aware / long-tail-aware weighted loss。

---

## 4. 方法设计

### 4.1 Dataset 改造

新增 `LongTailSidSFTDataset`，继承自 `SidSFTDataset` 的 prompt 与 tokenization 逻辑，仅额外完成：

1. 读取训练数据中的 `item_id`
2. 统计 item frequency
3. 按 head / middle / tail 分组
4. 为每条训练样本返回 `tail_weight`

返回样本新增字段：

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,
    "tail_weight": ...
}