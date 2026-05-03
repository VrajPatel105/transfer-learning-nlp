**The core idea**

When you fine-tune a pretrained model, different layers need different treatment. Layer1 already knows how to detect edges and textures — that took millions of images to learn. If you hit it with lr=1e-3, you'll overwrite those features with noise from your small dataset. That's called catastrophic forgetting. You want to nudge it, not rewrite it.

The final layers (layer4, fc) are the opposite — they were specialized for ImageNet classes, so they need to actually change for your task. Higher LR, faster adaptation.

---

**What the code is actually doing**

```python
layer_groups = [model.layer1, model.layer2, model.layer3, model.layer4, model.fc]
```

You're defining the groups manually in order from earliest to latest.

```python
enumerate(reversed(layer_groups))
```

This is the key move. You reverse the list so that when i=0, you're at `fc` (latest layer, highest LR), and when i=4, you're at `layer1` (earliest layer, lowest LR). Then the decay formula does this:

```
fc     → 1e-3 * (0.3^0) = 1e-3
layer4 → 1e-3 * (0.3^1) = 3e-4
layer3 → 1e-3 * (0.3^2) = 9e-5
layer2 → 1e-3 * (0.3^3) = 2.7e-5
layer1 → 1e-3 * (0.3^4) = 8.1e-6
```

Earlier layers get progressively smaller LRs. That's the entire mechanic.

---

**What you actually build**

Take ResNet-50 (or even a small custom CNN for now) and do this:

1. Load the model, print its named children to see the actual layer names
2. Build the param groups loop programmatically with a decay factor you can tune
3. Pass it to Adam
4. Print each param group's LR to verify it looks right before any training happens
5. Add a comment on each group explaining *why* that LR makes sense

The verification step matters — don't assume it worked, print the LRs and confirm the numbers match your expectation.

---

**One thing to think about while building it**

What should the decay factor be? 0.3 is aggressive — layer1 ends up at ~8e-6. Some people use 0.1, some use 0.5. There's no universal answer, it's a hyperparameter. But you should be able to reason about what happens if decay is too small (layers get similar LRs, early layers change too fast) vs too large (early layers barely update even when they should). Build the loop so changing decay is one number change.

Go build it. Come back when you have something running or if you hit a wall.