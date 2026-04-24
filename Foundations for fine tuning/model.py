# 1. Load the model, print its named children to see the actual layer names
# 2. Build the param groups loop programmatically with a decay factor you can tune
# 3. Pass it to Adam
# 4. Print each param group's LR to verify it looks right before any training happens
# 5. Add a comment on each group explaining *why* that LR makes sense

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import Adam

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# print(model)


# print(model.layer1)


# for name, child in model.named_children():
#     print(name)

layer_groups = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer4, model.fc]


# writing the params loop

# lets firstly initizlize the lr
# The three things you need:
# - A `base_lr` — the highest LR, applied to the last layer
# - A `decay` factor — multiplied per step as you go earlier in the network
# - A loop that builds a list of dicts, each with `'params'` and `'lr'`

# The output you're trying to build looks like this:

# ```python
# [
#     {'params': model.conv1.parameters(), 'lr': <smallest>},
#     {'params': model.layer1.parameters(), 'lr': <a bit bigger>},
#     ...
#     {'params': model.fc.parameters(), 'lr': <base_lr>}
# ]
# ```

base_lr = 1e-5
decay_factor = 10
param_list = []
for layer in layer_groups:
    param_list.append({
        'params': layer.parameters(),
        'lr' : base_lr
    })
    base_lr = base_lr*decay_factor

for i in param_list:
    print(i)
# output : 
# {'params': <generator object Module.parameters at 0x000001E8C1453E60>, 'lr': 1e-05}
# {'params': <generator object Module.parameters at 0x000001E8C1453D80>, 'lr': 0.0001}
# {'params': <generator object Module.parameters at 0x000001E8C1453CA0>, 'lr': 0.001}
# {'params': <generator object Module.parameters at 0x000001E8C14C4040>, 'lr': 0.01}
# {'params': <generator object Module.parameters at 0x000001E8C14C4120>, 'lr': 0.1}
# {'params': <generator object Module.parameters at 0x000001E8C14C4200>, 'lr': 1.0}

optimizer = Adam(params=param_list)

for group in optimizer.param_groups:
    print(group['lr'])
# output for above loop :
# 1e-05 
# 0.0001
# 0.001
# 0.01
# 0.1
# 1.0


# 1e-05 : model.conv1 --> this is the one of the earliest layer and we need to make sure to have very very small lr,
#                         Large lr would lead to add alot of noise which leads to 'catastrophic forgetting'
# 0.0001 : model.layer1 --> since this is also early layers, it needs small lr, reason is 'catastrophic forgetting'
# 0.001 : model.layer2 --> now we are coming closer to the end layers, but this is still the middle part, so lr ,
#                           should be not too small not too large
# 0.01 : model.layer3 --> now we are finally near the end layers, this layers needs fine tuning so our lr is large
# 0.1 : model.layer4 --> same as above
# 1.0 : model.fc --> this is the final last layer, and since the resnet fc had 1000 classes, our head here will need,
#                   alot of fine tuning to fit with our custom dataset which is why the lr is high.

NOTE : # one thing to keep in mind is that even for fc, usually the lr is around 1e-3 but since this is for understanding purpose, we have not used small lr