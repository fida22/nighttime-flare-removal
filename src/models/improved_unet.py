"""Configurable U-Net covering all four ablation experiments (Member 3).

  A  unet                  plain U-Net (use unet.UNet or build_model("unet"))
  B  attention             + attention gates on skip connections
  C  multiscale            + multi-scale dilated bottleneck
  D  attention_multiscale  both  (proposed model)

Same input/output interface as the baseline: [B,3,H,W] in [0,1] -> [B,3,H,W] in [0,1].
"""
import torch.nn as nn

from .attention import AttentionGate, MultiScaleDilatedBottleneck
from .unet import UNet


class ImprovedUNet(UNet):
    def __init__(self, in_ch=3, out_ch=3, base=32, depth=4,
                 use_attention=True, use_multiscale=True, dilations=(1, 2, 4)):
        self.use_multiscale = use_multiscale
        self.dilations = tuple(dilations)
        super().__init__(in_ch, out_ch, base, depth)
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.gates = None
        if use_attention:
            # gate at level i: decoder feature (after upsample) has chs[i] channels, skip has chs[i]
            self.gates = nn.ModuleList([AttentionGate(chs[i], chs[i]) for i in range(depth)])

    def make_bottleneck(self, cin, cout):
        if self.use_multiscale:
            # first conv block to change channels, then parallel dilated context
            return nn.Sequential(super().make_bottleneck(cin, cout),
                                 MultiScaleDilatedBottleneck(cout, cout, self.dilations))
        return super().make_bottleneck(cin, cout)

    def skip(self, level, enc_feat, dec_feat):
        if self.gates is None:
            return enc_feat
        return self.gates[level](dec_feat, enc_feat)


def build_model(name="unet", base=32, depth=4, dilations=(1, 2, 4)):
    name = name.lower()
    if name == "unet":
        return UNet(base=base, depth=depth)
    flags = {"attention": (True, False), "multiscale": (False, True),
             "attention_multiscale": (True, True)}
    if name not in flags:
        raise ValueError(f"unknown model '{name}'. Use unet|attention|multiscale|attention_multiscale")
    a, m = flags[name]
    return ImprovedUNet(base=base, depth=depth, use_attention=a, use_multiscale=m, dilations=dilations)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
