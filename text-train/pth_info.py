import torch

sd = torch.load("fault_ratio_bert_modify_softmax.pt", map_location="cpu")
print(isinstance(sd, dict), type(sd))

has_enc = any(k.startswith("encoder.") for k in sd.keys())
has_reg = any(k.startswith("regressor.") for k in sd.keys())
print("contains encoder?:", has_enc)
print("contains regressor?:", has_reg)

# regressor 키 몇 개만 보기
print([k for k in sd.keys() if k.startswith("regressor.")][:10])


for k,v in sd.items():
    if k.startswith("regressor."):
        print(k, tuple(v.shape))