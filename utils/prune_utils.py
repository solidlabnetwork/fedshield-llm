import torch
import torch.nn.utils.prune as prune
# ===== Pruning functions =====
def set_weight_by_mask(state_dict, mask):
    pruned_state_dict = {}
    for key in state_dict.keys():
        if key + "_mask" in mask:
            pruned_state_dict[key] = state_dict[key] * mask[key + "_mask"]
        else:
            pruned_state_dict[key] = state_dict[key]
    return pruned_state_dict

def generate_prune_param(state_dict):
    prune_params = []
    for key, value in state_dict.items():
        if "weight" in key and value.dim() >= 2:
            module = torch.nn.Module()
            setattr(module, "weight", torch.nn.Parameter(value.clone()))
            prune_params.append((module, "weight", key))
    return prune_params

def Prune_LLM(state_dict, sparsity):
    prune_params = generate_prune_param(state_dict)
    mask_dict = {}
    for module, name, key in prune_params:
        prune.l1_unstructured(module, name=name, amount=sparsity)
        mask_dict[key + "_mask"] = getattr(module, name + "_mask")
        prune.remove(module, name)
    return mask_dict
