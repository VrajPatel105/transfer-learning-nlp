# ResNet-50 Transfer Learning vs From Scratch — Flowers102

## Results Summary

| Metric | Version A (Fine-tuned) | Version B (From Scratch) |
|---|---|---|
| **Val Accuracy** | **91.37%** | **30.69%** |
| Final Train Loss | 0.4220 | 41.5110 |
| Epochs | 30 | 30 |
| Pretrained Weights | Yes (ImageNet) | No |
| Frozen Layers | conv1, layer1, layer2, layer3 | None |
| Trainable Params | ~17M | ~25.5M |
| Optimizer | Adam (layer-wise LR) | Adam (flat LR = 1e-3) |
| Dataset | Flowers102 (1020 train, 1020 val) | Same |

---

## Epoch-wise Train Loss

| Epoch | Version A (Fine-tuned) | Version B (From Scratch) |
|---|---|---|
| 1 | 126.3965 | 172.3873 |
| 2 | 53.5333 | 150.5144 |
| 3 | 13.7404 | 136.3120 |
| 4 | 3.5289 | 127.9908 |
| 5 | 1.9181 | 122.6887 |
| 6 | 1.2669 | 116.7218 |
| 7 | 0.8905 | 114.4548 |
| 8 | 0.6217 | 107.2272 |
| 9 | 0.5861 | 100.6808 |
| 10 | 0.3737 | 98.6739 |
| 11 | 0.7494 | 95.1223 |
| 12 | 1.0097 | 88.2034 |
| 13 | 0.7863 | 88.6971 |
| 14 | 1.3903 | 87.2648 |
| 15 | 2.4821 | 84.1401 |
| 16 | 1.8778 | 77.4492 |
| 17 | 1.1461 | 76.6978 |
| 18 | 1.0896 | 72.5030 |
| 19 | 0.8736 | 73.2169 |
| 20 | 1.6739 | 69.3417 |
| 21 | 1.3675 | 65.3623 |
| 22 | 1.5276 | 62.5380 |
| 23 | 2.5055 | 61.7951 |
| 24 | 1.2244 | 56.9347 |
| 25 | 0.9866 | 53.9677 |
| 26 | 1.2197 | 53.3728 |
| 27 | 0.2708 | 51.0507 |
| 28 | 0.1893 | 45.7970 |
| 29 | 0.3750 | 46.4246 |
| 30 | 0.4220 | 41.5110 |

---

## Key Takeaways

- Fine-tuned model reached **~0.37 loss by epoch 10**. From scratch was still at **98.67**.
- Version B trained **~8.5M more parameters** and still achieved 3x worse accuracy.
- Version A's loss instability (epochs 11–26) reflects the layer-wise LR tension between frozen and unfrozen layers — but it recovered and converged.
- This directly confirms Yosinski (2014): early layers learn general features that transfer across tasks. With only 1020 training images, the from-scratch model never had enough data to learn those representations.