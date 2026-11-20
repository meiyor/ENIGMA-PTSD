"""
This module defines the Pytorch ResNet encoders
for representing the two encoders for checking (1) SSL and (2)
semi-supervised feature projections define first the 3D ResNet as
based on torchvision models.

This model defines PyTorch encoders definitions for ENIGMA-PTSD multimodal representation learning.

This module provides:
  1) A flexible 3D ResNet encoder wrapper (ResNet3DEncoder) built on top of
   torchvision's 'r3d_18' VideoResNet backbone, with extra knobs for:
     - custom input channels (e.g., 1-channel neuroimaging volumes)
     - swapping activation functions globally (ReLU, GELU, SiLU, etc.)
     - selectively disabling deeper stages (layer1..layer4) for shallow encoders
     - regex-based convolution overrides to change kernel/stride/padding inside
       the backbone (including conv sequences inside torchvision BasicBlocks)
     - optional projection head via either:
         a) MLP (LayerNorm -> Linear)
         b) MultiheadAttention-based projection (attention_projection=True)

  2) A late-fusion model (LateFusion4DResNet) that processes multiple 3D streams
   (e.g., timepoints from 4D resting-state fMRI) using independent ResNet3DEncoder
   instances and fuses their embeddings using concat/mean/gated fusion.

  Design goals:
  - Handle common neuroimaging tensor formats (B, C, D, H, W) robustly
  - Provide controllable receptive fields (via conv_overrides) to match anisotropic
    voxel spacing or modality characteristics
  - Keep the code simple enough to serve both SSL (contrastive pretraining) and
    downstream semi-supervised/transfer setups.

  Typical usage:
  - Instantiate ResNet3DEncoder per modality map (ALFF/fALFF/ReHo/structural maps)
  - For 4D fMRI, split (B, X, Y, Z, T) into a list of T tensors of shape (B,1,D,H,W)
    and feed into LateFusion4DResNet.

  Notes:
  - torchvision VideoResNet BasicBlocks store conv1/conv2 as nn.Sequential wrappers,
    so overriding them requires rebuilding those sequences; utilities in this module
    handle that safely.
"""

import re  # for detecing name patterns in the model's layers

import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18

# define external auxiliary after-the-fact patches adding knobs on each
# layer as a posterior dictionaries

# define here autopadding for avoid problems with dimension misalignments


def auto_padding(kernel, dilation=(1, 1, 1)):
    """
    Compute "same-ish" symmetric padding for 3D kernels under dilation.

    For odd-valued kernel sizes, this returns the padding that preserves spatial
    dimensions for stride=1 (approximately "same" convolution). The formula used:

      padding_i = dilation_i * (kernel_i // 2)

    Parameters
    ----------
    kernel : tuple[int, int, int]
      3D kernel size (kD, kH, kW). Expected to be odd in most cases.
    dilation : tuple[int, int, int], default=(1,1,1)
      Dilation factors along each dimension.

    Returns
    -------
    tuple[int, int, int]
      Padding values (pD, pH, pW) suitable for nn.Conv3d(..., padding=...).
    """
    return tuple(d * (k // 2) for k, d in zip(kernel, dilation, strict=False))


# use this to change the activation function if necessary..


def make_act(name: str, **kwargs) -> nn.Module:
    """
    Factory for activation functions used throughout the encoders.

    Parameters
    ----------
    name : str
      Activation identifier (case-insensitive). Supported:
      - "relu"
      - "leaky_relu"
      - "elu"
      - "gelu"
      - "silu" / "swish"
      - "mish"
    **kwargs
      Optional activation-specific keyword args (e.g., negative_slope for leaky_relu).

    Returns
    -------
    torch.nn.Module
       Instantiated activation module.

    Raises
    ------
    ValueError
      If `name` is not recognized.
    """

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=kwargs.get("inplace", True))
    if name == "leaky_relu":
        return nn.LeakyReLU(
            negative_slope=kwargs.get("negative_slope", 0.01), inplace=kwargs.get("inplace", True)
        )
    if name == "elu":
        return nn.ELU(alpha=kwargs.get("alpha", 1.0), inplace=kwargs.get("inplace", True))
    if name == "gelu":
        return nn.GELU()  # no inplace
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=kwargs.get("inplace", True))
    if name == "mish":
        return nn.Mish()  # no inplace
    raise ValueError(f"Unknown activation '{name}'")


def replace_videorn_activations(model: nn.Module, act_name: str, act_kwargs=None):
    """
     Replace activation layers inside a torchvision VideoResNet (e.g., r3d_18).

    This function traverses a VideoResNet-style model and replaces:
     - Stem ReLU(s) inside `stem` Sequential blocks (optional but typically desired)
     - Block-level activations:
        * block.conv1[2]  (activation inside conv1 Sequential)
        * block.relu      (post-residual activation)

    Parameters
    ----------
    model : torch.nn.Module
      The VideoResNet backbone (torchvision.models.video.r3d_18 output).
    act_name : str
        Name passed to `make_act` (e.g., "gelu", "silu", "relu").
    act_kwargs : dict | None
       Optional kwargs passed to the activation constructor.

    Returns
    -------
    None
      Operates in-place.

    """
    act_kwargs = act_kwargs or {}

    def new_act():
        return make_act(act_name, **act_kwargs)

    # Replace stem ReLU(s) (optional but usually desired)
    for name, m in model.named_modules():
        # stem is BasicStem: [conv, bn, relu, (maybe maxpool)]
        if name.endswith("stem") and isinstance(m, nn.Sequential):
            for i, child in enumerate(m):
                if isinstance(child, nn.ReLU):
                    m[i] = new_act()

    # Replace block activations
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        stage = getattr(model, stage_name, None)
        if stage is None or isinstance(stage, nn.Identity) or not isinstance(stage, nn.Sequential):
            continue

        for blk in stage:
            # 1) conv1 sequential activation at index 2 (conv,bn,act)
            if (
                hasattr(blk, "conv1")
                and isinstance(blk.conv1, nn.Sequential)
                and len(blk.conv1) >= 3
            ):
                if isinstance(blk.conv1[2], nn.ReLU):
                    blk.conv1[2] = new_act()
                else:
                    # sometimes it's already a different act; replace anyway
                    blk.conv1[2] = new_act()

            # 2) final activation after residual add
            if hasattr(blk, "relu") and isinstance(blk.relu, nn.Module):
                # often nn.ReLU; replace regardless
                blk.relu = new_act()


def replace_conv3d(conv: nn.Conv3d, **kwargs) -> nn.Conv3d:
    """
    Replace a Conv3d hyperparams; copy weights if shape matches.

    Create a new nn.Conv3d with updated hyperparameters and (if possible) copy weights.

    Parameters
    ----------
    conv : nn.Conv3d
      Existing convolution to replace.
    **kwargs
      Any Conv3d hyperparameter to override (kernel_size, stride, padding, dilation,
      groups, bias).

    Returns
    -------
    nn.Conv3d
      New convolution module.

    Weight handling
    ---------------
     - If the new weight tensor shape matches the original, weights are copied.
     - Otherwise, weights are initialized with Kaiming normal (fan_out, relu).
     - Bias is copied if shape matches; otherwise initialized to zeros.

    """

    new = nn.Conv3d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=kwargs.get("kernel_size", conv.kernel_size),
        stride=kwargs.get("stride", conv.stride),
        padding=kwargs.get("padding", conv.padding),
        dilation=kwargs.get("dilation", conv.dilation),
        groups=kwargs.get("groups", conv.groups),
        bias=kwargs.get("bias", (conv.bias is not None)),
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if new.weight.shape == conv.weight.shape:
            new.weight.copy_(conv.weight)
        else:
            nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        if new.bias is not None:
            if conv.bias is not None and new.bias.shape == conv.bias.shape:
                new.bias.copy_(conv.bias)
            else:
                nn.init.zeros_(new.bias)
    return new


def set_module_by_name(root: nn.Module, name: str, new_module: nn.Module):
    """
    Replace a nested submodule by its dotted path name.

    Supports both attribute traversal and integer indexing for Sequential / ModuleList,
    e.g.:
    - "stem.0"
    - "layer2.0.downsample.0"
    - "layer1.1.conv1"

    Parameters
    ----------
    root : nn.Module
      Root module containing the target.
    name : str
      Dotted path to the target module.
    new_module : nn.Module
      Replacement module.

    Returns
    -------
    None
      Operates in-place.

    """

    parent = root
    parts = name.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _get_by_name(root: nn.Module, name: str) -> nn.Module:
    """
    Get nested module by dotted name; supports numeric indices.
    Retrieve a nested submodule by dotted path name (attribute/index aware).

    Parameters
    ----------
    root : nn.Module
      Root module.
    name : str
      Dotted path.

    Returns
    -------
    nn.Module
      The located submodule.

    """
    cur = root
    for p in name.split("."):
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur


def _replace_block_conv_seq(base: nn.Module, seq_name: str, rule: dict):
    """
    Replace VideoResNet BasicBlock's conv1/conv2 which are nn.Sequential:
     conv1: [Conv3DSimple, BN, ReLU]
     conv2: [Conv3DSimple, BN]
    with:
     [Conv3d, BN, (ReLU if present)]

    Rebuild a torchvision VideoResNet BasicBlock conv sequence (conv1/conv2).

    In torchvision VideoResNet BasicBlocks:
     - conv1 is nn.Sequential([Conv3DSimple-like wrapper, BatchNorm3d, ReLU])
     - conv2 is nn.Sequential([Conv3DSimple-like wrapper, BatchNorm3d])

    Since the first element is often a lightweight wrapper rather than a raw Conv3d,
    this function reconstructs the sequence explicitly as:
       [nn.Conv3d(in_c, out_c, ...), BatchNorm3d(out_c), (optional ReLU)]

    Parameters
    ----------
    base : nn.Module
       The VideoResNet backbone.
    seq_name : str
       Dotted name of the conv sequence, e.g. "layer1.0.conv1".
    rule : dict
       Override settings. Supported keys:
         - kernel_size, stride, padding, dilation, groups

    Returns
    -------
    None
       Operates in-place

    """

    seq = _get_by_name(base, seq_name)
    if not isinstance(seq, nn.Sequential) or len(seq) < 2:
        raise ValueError(
            f"{seq_name} is not a conv-seq Sequential (expected conv+bn[+relu]). Got: {type(seq)}"
        )

    # conv-like wrapper sits at seq[0]; it has in/out planes as attributes in
    # VideoResNet
    conv_like = seq[0]
    if hasattr(conv_like, "in_planes") and hasattr(conv_like, "out_planes"):
        in_c = int(conv_like.in_planes)
        out_c = int(conv_like.out_planes)
    else:
        # fallback guess (rare): try common names
        in_c = int(conv_like.in_channels)
        out_c = int(conv_like.out_channels)

    bn = seq[1]
    if not isinstance(bn, nn.BatchNorm3d):
        # still allow other norm types, but must accept num_features
        bn_cls = bn.__class__
    else:
        bn_cls = nn.BatchNorm3d

    # Keep stride default to (1,1,1) unless rule explicitly sets it
    stride = rule.get("stride", (1, 1, 1))

    new_seq = nn.Sequential(
        nn.Conv3d(
            in_c,
            out_c,
            kernel_size=rule.get("kernel_size", (3, 3, 3)),
            stride=stride,
            padding=rule.get("padding", (1, 1, 1)),
            dilation=rule.get("dilation", (1, 1, 1)),
            groups=rule.get("groups", 1),
            bias=False,
        ),
        bn_cls(out_c),
    )

    # Preserve presence of relu if original seq had it (conv1 does)
    if len(seq) >= 3 and isinstance(seq[2], nn.ReLU):
        new_seq.add_module("2", nn.ReLU(inplace=True))

    set_module_by_name(base, seq_name, new_seq)


def apply_conv_overrides_videorn(base: nn.Module, conv_overrides: list[dict]):
    """
    Apply overrides to:
      1) real nn.Conv3d modules (stem/downsample etc) by name regex
      2) VideoResNet block conv sequences: layerX.Y.conv1 / layerX.Y.conv2 (Sequential)
         by rebuilding them as Conv3d+BN(+ReLU)

    Patterns supported:
      - '^stem\\.0$'
      - '^layer2\\.0\\.downsample\\.0$'
      - '^layer1\\.\\d+\\.conv1$'   (sequence)
      - '^layer1\\.\\d+\\.conv2$'   (sequence)


    Apply regex-driven convolution overrides to a torchvision VideoResNet backbone.

    Two-pass strategy:
      1) Replace *real* nn.Conv3d modules (stem, downsample convs, etc.) by name match
      2) Rebuild BasicBlock conv sequences (layerX.Y.conv1 / conv2) which are
         nn.Sequential wrappers rather than raw Conv3d modules

    Parameters
    ----------
    base : nn.Module
        torchvision VideoResNet backbone (e.g., returned by r3d_18()).
    conv_overrides : list[dict]
        Each rule must include:
          - "pattern": regex string matched against module names
        Optional keys:
          - kernel_size, stride, padding, dilation, groups

    Returns
    -------
    None
        Operates in-place.

    """
    # ---------
    # Pass 1: real Conv3d modules
    # ---------
    for name, module in list(base.named_modules()):
        if not isinstance(module, nn.Conv3d):
            continue
        for rule in conv_overrides:
            pat = rule.get("pattern")
            if pat is None:
                raise ValueError("Each conv_overrides rule must include a 'pattern' regex string.")
            if re.search(pat, name) is None:
                continue

            new_module = replace_conv3d(
                module,
                kernel_size=rule.get("kernel_size", None),
                stride=rule.get("stride", None),
                padding=rule.get("padding", None),
                dilation=rule.get("dilation", None),
                groups=rule.get("groups", None),
            )
            set_module_by_name(base, name, new_module)
            break

    # ---------
    # Pass 2: block conv sequences (conv1/conv2)
    # ---------
    # We match names like "layer1.0.conv1" which are Sequential, not Conv3d.
    for name, module in list(base.named_modules()):
        if not isinstance(module, nn.Sequential):
            continue
        for rule in conv_overrides:
            pat = rule.get("pattern")
            if pat is None:
                raise ValueError("Each conv_overrides rule must include a 'pattern' regex string.")
            if re.search(pat, name) is None:
                continue

            # Only rebuild if this looks like a block conv seq
            # (conv-like wrapper + BN [+ ReLU])
            if len(module) >= 2 and isinstance(module[1], nn.BatchNorm3d):
                # Optional: prevent stride edits on identity-skip blocks (layerX.1 etc.)
                # Safer: if you set stride != (1,1,1) on non-first block, it
                # will break residual add.
                if rule.get("stride", None) is not None:
                    # Determine if this is first block of a stage (layer?.0.*)
                    # or not
                    m = re.search(r"layer[1-4]\.(\d+)\.conv[12]$", name)
                    if m and int(m.group(1)) != 0:
                        raise ValueError(
                            f"Refusing stride override on {name} (block index != 0). "
                            "This will break residual addition because downsample=None in these blocks."
                        )

                _replace_block_conv_seq(base, name, rule)
                break


class ResNet3DEncoder(nn.Module):
    """
    Flexible 3D-ResNet encoder based on torchvision r3d_18.

    Key knobs:
      - Change stem kernel/stride/padding + optional maxpool
      - Use only the first K stages (skip deeper layers)
      - Optional adaptive pooling to handle varying spatial sizes

    Input:  (B, C, D, H, W)
    Output: (B, out_dim)

    Flexible 3D ResNet encoder built on torchvision's r3d_18 (VideoResNet).

    This encoder is designed for neuroimaging-style 3D tensors:
        Input : (B, C, D, H, W)
        Output: (B, out_dim)   (after optional neck + optional attention/MLP projection)

    Key features
    ------------
    1) Custom input channels
       - Replaces the stem conv (RGB->1ch, or other C) and optionally adapts pretrained
         weights by averaging RGB channels when C=1.

    2) Stage selection (shallow vs deep backbones)
       - You can disable layer1..layer4 using `use_stages=(...)`.
       - Useful for small volumes or when you want fast encoders for SSL pairing.

    3) Activation swapping
       - `activation_function` applied across the backbone using
         replace_videorn_activations().

    4) Regex-based Conv overrides
       - `conv_overrides` modifies internal convs by name pattern, including
         BasicBlock conv sequences (conv1/conv2) that are stored as nn.Sequential.

    5) Projection head options
       - If `attention_projection=False`:
           proj = LayerNorm -> Linear(feat_dim -> out_dim)
       - If `attention_projection=True`:
           proj = MultiheadAttention on the pooled feature tokens (batch_first=True),
           with residual + LayerNorm.

       Note: the current implementation expects the pooled representation `y`
       to be suitable for MHA. If you intend true token attention, you may want to
       reshape (B, C) -> (B, L, C) with L>1 (e.g., spatial tokens) before attention.

    Parameters
    ----------
    in_channels : int, default=1
        Number of input channels.
    out_dim : int, default=512
        Output embedding dimension produced by the projection head (when used).
    emb_dim : int | None, default=None
        Intermediate embedding size produced by the neck. If None, uses the backbone
        inferred feature dimension.
    pretrained : bool, default=False
        If True, loads torchvision default weights for r3d_18 and adapts the stem conv.

    stem_kernel, stem_stride, stem_padding : tuple[int,int,int]
        Hyperparameters used to rebuild the stem conv.
    use_maxpool : bool, default=True
        If False, replaces the stem maxpool with Identity to preserve detail.

    activation_function : str, default="relu"
        One of the names supported by make_act().

    attention_projection : bool, default=False
        If True, uses MultiheadAttention-based projection (see notes above).
    attn_heads : int, default=2
        Number of attention heads when attention_projection=True.

    use_stages : tuple[bool,bool,bool,bool], default=(True,True,True,True)
        Whether to keep (layer1, layer2, layer3, layer4). Disabled stages are replaced
        with Identity modules.

    conv_overrides : list[dict] | None
        Regex rules to override conv hyperparameters; see apply_conv_overrides_videorn().

    adaptive_pool : bool, default=True
        If True, applies AdaptiveAvgPool3d((1,1,1)) before flattening.
    freeze_backbone : bool, default=False
        If True, sets requires_grad=False for all backbone parameters.

    Attributes
    ----------
    backbone : nn.Module
        The modified r3d_18 backbone with selected stages and activation replacements.
    neck : nn.Module
        Optional mapping from backbone feature dimension -> emb_dim (LayerNorm+Linear+GELU).
    proj : nn.Module
        Final projection head producing out_dim (MLP or MHA-based depending on flags).

    Returns
    -------
    forward(x) -> torch.Tensor
        Tensor of shape (B, out_dim).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_dim: int = 512,
        emb_dim: int | None = None,  # if None -> uses backbone feature dim
        pretrained: bool = False,
        # ---- Stem customization # define this as default values based on the input data
        stem_kernel: tuple[int, int, int] = (3, 7, 7),
        stem_stride: tuple[int, int, int] = (1, 2, 2),
        stem_padding: tuple[int, int, int] | None = None,
        use_maxpool: bool = True,  # keep/remove the maxpool in stem
        activation_function: str = "relu",  # define the activation depending on the function
        attention_projection: bool = False,
        attn_heads: int = 2,
        # ---- Stage selection (skip deeper layers)
        # stages are: layer1, layer2, layer3, layer4
        use_stages: tuple[bool, bool, bool, bool] = (True, True, True, True),
        # define the stage strides here..
        stage_strides: dict[str, tuple[int, int, int]] | None = None,
        conv_overrides: list[dict] | None = None,
        # ---- Pooling / head
        adaptive_pool: bool = True,  # AdaptiveAvgPool3d((1,1,1)) before flatten
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        base = r3d_18(weights=weights, progress=True)

        # -------------------------
        # 1) Rebuild stem conv with custom kernel/stride/padding and in_channels
        # -------------------------
        old_conv = base.stem[0]  # Conv3d(3->64 by default)
        self.attention_projection = attention_projection
        self.activation_function = activation_function
        self.attn_heads = attn_heads

        if stem_padding is None:
            # "same-ish" padding for odd kernels
            stem_padding = tuple(k // 2 for k in stem_kernel)

        new_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=stem_kernel,
            stride=stem_stride,
            padding=stem_padding,
            bias=(old_conv.bias is not None),
        )

        # If pretrained, and converting RGB->1ch, initialize by averaging
        # channels
        if pretrained and old_conv.weight.shape[1] == 3 and in_channels == 1:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        base.stem[0] = new_conv

        # Optionally remove the maxpool in stem (helps preserve spatial detail)
        # torchvision VideoResNet stem is roughly: [conv, bn, relu, maxpool]
        # In r3d_18 it's stored as a Sequential; we can replace pool with
        # Identity. Do this optionally!!
        if not use_maxpool:
            # Usually maxpool is the last module in stem
            # (safe even if structure changes slightly; we check type)
            for i in range(len(base.stem) - 1, -1, -1):
                if isinstance(base.stem[i], nn.MaxPool3d):
                    base.stem[i] = nn.Identity()
                    break

        # -------------------------
        # 2) Keep only selected stages (skip deeper layers)
        # -------------------------
        s1, s2, s3, s4 = use_stages
        if not s1:
            base.layer1 = nn.Identity()
        if not s2:
            base.layer2 = nn.Identity()
        if not s3:
            base.layer3 = nn.Identity()
        if not s4:
            base.layer4 = nn.Identity()

        # -------------------------
        # 3) NEW: override any Conv3d by name regex pattern inside the network
        # -------------------------
        if conv_overrides:
            apply_conv_overrides_videorn(base, conv_overrides)

        # We'll ignore base.avgpool/base.fc and do our own pooling/head
        base.avgpool = nn.Identity()
        base.fc = nn.Identity()

        # be flexible with layers definition here
        self.backbone = base
        replace_videorn_activations(self.backbone, self.activation_function)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if adaptive_pool else nn.Identity()

        # -------------------------
        # 5) Determine feature dim dynamically (robust to skipping stages)
        # -------------------------
        # We forward a tiny dummy tensor through backbone parts to infer channels.
        # This avoids hardcoding 512 vs 256 etc.
        self._feat_dim = None
        self._infer_feat_dim(in_channels)

        feat_dim = self._feat_dim
        feat_dim if emb_dim is None else emb_dim

        # If user requested emb_dim != feat_dim, add a "neck" to map to emb_dim
        # first. **THIS NECK WILL BE DEFINED AS THE LAST LAYER OF THE 3D RESNET
        # ENCODER
        self.neck = nn.Identity()
        # need to define emb_dim anyways
        if emb_dim is not None and emb_dim != feat_dim:
            self.neck = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, emb_dim),
                nn.GELU(),
            )
            feat_dim = emb_dim

            if self.attention_projection is True:
                self.pre_norm = nn.LayerNorm(feat_dim)  # before attention
                self.proj = nn.MultiheadAttention(
                    embed_dim=feat_dim, num_heads=self.attn_heads, batch_first=True
                )
                self.norm = nn.Sequential(
                    nn.LayerNorm(feat_dim),
                    nn.Linear(feat_dim, out_dim),
                )
            else:
                self.proj = nn.Sequential(
                    nn.LayerNorm(feat_dim),
                    nn.Linear(feat_dim, out_dim),
                )

        # if we will like to parse it with freeze parameters
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # breakpoint()

    @torch.no_grad()
    def _infer_feat_dim(self, in_channels: int):
        # Minimal input; spatial sizes just need to be >= a few strides
        x = torch.zeros(1, in_channels, 16, 64, 64)
        y = self.backbone.stem(x)
        y = self.backbone.layer1(y)
        y = self.backbone.layer2(y)
        y = self.backbone.layer3(y)
        y = self.backbone.layer4(y)
        y = self.adaptive_pool(y)
        y = torch.flatten(y, 1)
        self._feat_dim = y.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = self.backbone.stem(x)
        y = self.backbone.layer1(y)
        y = self.backbone.layer2(y)
        y = self.backbone.layer3(y)
        y = self.backbone.layer4(y)
        y = self.adaptive_pool(y)
        y = torch.flatten(y, 1)  # (B, C)
        y = self.neck(y)  # optional (B, emb_dim)

        if self.attention_projection is True:
            # define here the residual connection as expection
            y = self.pre_norm(y).unsqueeze(1)       # (B, 1, features)
            y0, _ = self.proj(y, y, y, need_weights=False)
            z = self.norm(y0 + y).squeeze(1)        # (B, out_dim)
        else:
            # (B, out_dim) # this construct the final output and the final model!!
            z = self.proj(y)

        return z


# Define here the model for the 4D ResNet multiple branch model
class LateFusion4DResNet(nn.Module):
    """
    Late fusion over N streams: encode each stream with its own 3D-ResNet encoder,
    then fuse embeddings. Each for timepoint in the array

    Example streams:
      - T1w (B,1,D,H,W)
      - fMRI summary volume (B,1,D,H,W) or 4D reduced to 3D
      - ROI volume / derived maps (ReHo/fALFF) (B,1,D,H,W)

    ate-fusion multi-stream model for 4D data represented as a list of 3D volumes.

    Intended use: resting-state fMRI
      - Start with a 4D tensor (B, X, Y, Z, T) or equivalent
      - Reorder/slice into a list of T tensors each shaped (B, 1, D, H, W)
      - Feed that list into this model:
            z = model([x_t0, x_t1, ..., x_t(T-1)])

    Architecture
    ------------
    - `n_streams` independent ResNet3DEncoder instances (one per stream/timepoint)
    - Encode each stream to an embedding
    - Fuse embeddings using one of:
        * "concat": concatenate along feature dimension then reduce via MLP
        * "mean"  : average embeddings across streams then map via MLP
        * "gated" : learn a per-stream weight (gate) and compute a weighted average

    Parameters
    ----------
    n_streams : int
        Number of streams/timepoints expected in the forward input list.
    emb_dim : int, default=128
        Embedding dimension produced by each stream encoder's neck.
    fusion : str, default="concat"
        Fusion type: {"concat", "mean", "gated"}.
    out_dim : int, default=64
        Output dimension after fusion MLP.
    output_dim : int, default=32
        (Currently unused in this implementation; kept for compatibility/experiments.)
    pretrained : bool, default=False
        If True, loads pretrained r3d_18 weights for each stream encoder.

    Inputs
    ------
    xs : list[torch.Tensor]
        Length must equal n_streams. Each tensor shaped (B, 1, D, H, W).

    Returns
    -------
    torch.Tensor
        Fused embedding of shape (B, out_dim).

    """

    def __init__(
        self,
        n_streams: int,
        emb_dim: int = 128,
        fusion: str = "concat",  # "concat" | "mean" | "gated"
        out_dim: int = 64,
        pretrained: bool = False,
    ):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                ResNet3DEncoder(
                    in_channels=1,
                    emb_dim=emb_dim,
                    out_dim=out_dim,
                    use_stages=(True, False, False, False),
                    conv_overrides=[
                        # alway define this kernel size and padding  as odd values
                        # NOT event. Just work on the convolutional kernels
                        {
                            "pattern": r"^layer1\.\d+\.conv1$",
                            "kernel_size": (3, 3, 3),
                            "padding": auto_padding((3, 3, 3)),
                        },
                        {
                            "pattern": r"^layer1\.\d+\.conv2$",
                            "kernel_size": (3, 3, 3),
                            "padding": auto_padding((3, 3, 3)),
                        },
                    ],
                    pretrained=pretrained,
                )
                for i in range(n_streams)
            ]
        )

        # REMEMBER TO ADD A MHSA BLOCK IF THIS WILL BE NECESSARY!!
        self.fusion = fusion.lower()
        if self.fusion == "concat":
            fusion_in = n_streams * out_dim
            self.fuse = nn.Sequential(
                nn.LayerNorm(fusion_in),
                nn.Linear(fusion_in, out_dim),
                nn.ReLU(),
            )
        elif self.fusion == "mean":
            self.fuse = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, out_dim),
                nn.ReLU(),
            )
        elif self.fusion == "gated":
            # learn per-stream gates from embeddings (simple, effective)
            self.gate = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())
            self.fuse = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, out_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown fusion='{fusion}'")

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        assert len(xs) == len(self.encoders)

        zs = [enc(x) for enc, x in zip(self.encoders, xs, strict=False)]  # list of (B, emb_dim)

        if self.fusion == "concat":
            z = torch.cat(zs, dim=1)

        elif self.fusion == "mean":
            z = torch.stack(zs, dim=0).mean(dim=0)

        elif self.fusion == "gated":
            # weighted average of streams
            weights = torch.cat([self.gate(zi) for zi in zs], dim=1)  # (B, n_streams)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            z = 0.0
            for i, zi in enumerate(zs):
                z = z + zi * weights[:, i : i + 1]

        return self.fuse(z)


# Define the ResNetLSTM as choice to process the temporal dynamics in the last layer
class LateFusion4DResNet_LSTM(nn.Module):
    """
    Late-fusion RS model:
      - n_streams separate ResNet3DEncoder branches (one per timepoint/stream)
      - stack outputs into a sequence (B, T, out_dim)
      - LSTM over time -> final summary vector
      - small head -> (B, out_dim_final)

    Expected input:
      xs: list length n_streams, each tensor shaped (B, 1, D, H, W)
    """

    def __init__(
        self,
        n_streams: int,
        emb_dim: int = 128,
        fusion: str = "lstm",          # kept for compatibility; you can ignore
        out_dim: int = 64,             # each ResNet3DEncoder output dim
        pretrained: bool = False,

        # ---- LSTM config
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        lstm_dropout: float = 0.0,     # only used if lstm_layers > 1

        # ---- pooling over time after LSTM
        pool: str = "last",            # "last" | "mean"

        # ---- final projection
        out_dim_final: int = 64,
    ):
        super().__init__()

        self.n_streams = n_streams
        self.pool = pool

        # 1) Independent per-stream encoders (as you requested)
        self.encoders = nn.ModuleList(
            [
                ResNet3DEncoder(
                    in_channels=1,
                    emb_dim=emb_dim,
                    out_dim=out_dim,
                    use_stages=(True, False, False, False),
                    conv_overrides=[
                        {
                            "pattern": r"^layer1\.\d+\.conv1$",
                            "kernel_size": (3, 3, 3),
                            "padding": auto_padding((3, 3, 3)),
                        },
                        {
                            "pattern": r"^layer1\.\d+\.conv2$",
                            "kernel_size": (3, 3, 3),
                            "padding": auto_padding((3, 3, 3)),
                        },
                    ],
                    pretrained=pretrained,
                )
                for _ in range(n_streams)
            ]
        )

        # 2) LSTM over the per-stream feature sequence
        self.lstm = nn.LSTM(
            input_size=out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # 3) Final head
        self.fuse = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, out_dim_final),
            nn.ReLU(),
        )

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """
        xs: list length n_streams of (B,1,D,H,W)
        returns: (B, out_dim_final)
        """
        assert len(xs) == len(self.encoders), f"Expected {len(self.encoders)} streams, got {len(xs)}"

        # Encode each stream -> list of (B, out_dim)
        zs = [enc(x) for enc, x in zip(self.encoders, xs, strict=False)]
        # Stack into time sequence: (B, T, out_dim)
        seq = torch.stack(zs, dim=1)
        # LSTM
        out, _ = self.lstm(seq)  # out: (B, T, lstm_out_dim)
        # Pool over time
        if self.pool == "mean":
            pooled = out.mean(dim=1)   # (B, lstm_out_dim)
        else:
            pooled = out[:, -1, :]     # (B, lstm_out_dim)

        return self.fuse(pooled)

class LateFusion4DResNet_TemporalConv(nn.Module):
    """
    n_streams ResNet3D branches -> (B,T,C) -> temporal Conv1d (mixing channels) -> head
    """
    def __init__(
        self,
        n_streams: int,
        emb_dim: int = 128,
        out_dim: int = 64,            # per-branch output channels C
        pretrained: bool = False,

        # temporal conv config
        kernel_size: int = 5,
        n_blocks: int = 3,
        hidden_channels: int = 128,
        dilation_growth: int = 2,
        attention_projection: bool = False,
        attn_heads: int = 2,

        pool: str = "mean",
        out_dim_final: int = 64,
    ):
        super().__init__()
        self.pool = pool
        self.attention_projection = attention_projection
        self.attn_heads = attn_heads

        self.encoders = nn.ModuleList(
            [
                ResNet3DEncoder(
                    in_channels=1,
                    emb_dim=emb_dim,
                    out_dim=out_dim,
                    use_stages=(True, False, False, False),
                    conv_overrides=[
                        {"pattern": r"^layer1\.\d+\.conv1$", "kernel_size": (3,3,3), "padding": auto_padding((3,3,3))},
                        {"pattern": r"^layer1\.\d+\.conv2$", "kernel_size": (3,3,3), "padding": auto_padding((3,3,3))},
                    ],
                    pretrained=pretrained,
                )
                for _ in range(n_streams)
            ]
        )

        layers = []
        dilation = 1

        in_ch = out_dim
        for _ in range(n_blocks):
            pad = (kernel_size // 2) * dilation
            layers += [
                nn.Conv1d(in_ch, hidden_channels, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
            ]
            in_ch = hidden_channels
            dilation *= dilation_growth

        self.temporal = nn.Sequential(*layers)

        if self.attention_projection is True:
                self.pre_norm = nn.LayerNorm(hidden_channels)  # before attention
                self.head = nn.MultiheadAttention(
                    embed_dim=hidden_channels, num_heads=self.attn_heads, batch_first=True
                )
                self.norm = nn.Sequential(
                    nn.LayerNorm(hidden_channels),
                    nn.Linear(hidden_channels, out_dim_final),
                )
        else:
            self.head = nn.Sequential(
                  nn.LayerNorm(hidden_channels),
                  nn.Linear(hidden_channels, out_dim_final),
            )

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        # assert len(xs) == len(self.encoders), f"Expected {len(self.encoders)} streams, got {len(xs)}"

        zs = [enc(x) for enc, x in zip(self.encoders, xs, strict=False)]  # list (B,C)
        seq = torch.stack(zs, dim=1)                                      # (B,T,C)

        y = seq.transpose(1, 2)                                           # (B,C,T)
        y = self.temporal(y)                                              # (B,hidden,T)

        if self.pool == "mean":
            pooled = y.mean(dim=2)                                        # (B,hidden)
        else:
            pooled = y[:, :, -1]                                          # (B,hidden)

        if self.attention_projection is True:
            # define here the residual connection as expection
            y = self.pre_norm(pooled).unsqueeze(1)  # (B, 1, hidden)
            y0, _ = self.head(y, y, y, need_weights=False)
            o = self.norm(y0 + y).squeeze(1)        # (B, out_dim_final)
        else:
            o = self.head(pooled)

        return o

## define the variable length TCN attention pooling encoder for the RSData

class SharedResNet4D_TemporalConv(nn.Module):
    """
    Shared 3D ResNet encoder with temporal convolutions and variable-length
    pooling for resting-state fMRI (rs-fMRI) representation learning.

    Unlike LateFusion4DResNet_TemporalConv, which creates an independent
    ResNet3DEncoder for each temporal volume, this model uses a single
    shared spatial encoder across all timepoints.

    Architecture
    ------------
    1. Shared ResNet3DEncoder processes each valid 3D fMRI volume.
    2. Temporal Conv1d blocks model relationships across timepoints.
    3. Masked temporal pooling aggregates variable-length sequences.
    4. A final projection generates one fixed-dimensional embedding
       for each subject.

    Input
    -----
    x : torch.Tensor
        Padded rs-fMRI batch with shape (B, T, 1, D, H, W).

        B : number of subjects
        T : maximum temporal length in the batch
        D, H, W : spatial dimensions

        Alternatively, x can be a list of T tensors with shape
        (B, 1, D, H, W), provided all subjects have the same length
        or a separate lengths tensor is supplied.

    lengths : torch.Tensor, optional
        Actual number of valid timepoints for each subject, shape (B,).
        If None, all T timepoints are considered valid.

    Parameters
    ----------
    emb_dim : int, default=128
        Intermediate embedding dimension of ResNet3DEncoder.

    out_dim : int, default=64
        Feature dimension produced by the shared spatial encoder.

    pretrained : bool, default=False
        Whether to initialize ResNet3DEncoder using pretrained weights.

    kernel_size : int, default=5
        Temporal convolution kernel size. Must be odd.

    n_blocks : int, default=3
        Number of temporal convolution blocks.

    hidden_channels : int, default=128
        Number of output channels in each temporal convolution block.

    dilation_growth : int, default=2
        Multiplicative growth factor for temporal dilation.

    pool : str, default="mean_std"
        Temporal pooling strategy:
            "mean"     : masked temporal mean
            "mean_std" : concatenated masked mean and standard deviation
            "last"     : last valid temporal feature

    out_dim_final : int, default=64
        Final subject-level embedding dimension.

    frame_chunk : int, default=16
        Maximum number of valid 3D volumes processed per spatial
        encoder forward call. Limits activation memory per chunk.

    Returns
    -------
    torch.Tensor
        Subject-level embeddings with shape (B, out_dim_final).

    Optional returns
    ----------------
    When return_sequence=True, also returns:

        temporal_features : (B, T, hidden_channels)
        valid_mask        : (B, T)

    Notes
    -----
    - Temporal lengths can differ between subjects.
    - Padded timepoints are excluded from spatial encoding and pooling.
    - LayerNorm is applied independently at each temporal position.
    - Temporal convolution outputs are masked after every block.
    - Temporal mean/std pooling produces a fixed embedding size.
    - No CTQ scores, diagnostic labels, or cluster assignments are used.
    - This model does not perform temporal resampling; sequences should
      already be represented on the intended temporal sampling grid.
    """

    def __init__(
        self,
        emb_dim: int = 128,
        out_dim: int = 64,
        pretrained: bool = False,
        kernel_size: int = 5,
        n_blocks: int = 3,
        hidden_channels: int = 128,
        dilation_growth: int = 2,
        pool: str = "mean_std",
        out_dim_final: int = 64,
        frame_chunk: int = 16,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1.")

        if dilation_growth < 1:
            raise ValueError("dilation_growth must be >= 1.")

        if frame_chunk < 1:
            raise ValueError("frame_chunk must be >= 1.")

        if pool not in ("mean", "mean_std", "last", "attention"):
           raise ValueError(
             "pool must be one of: "
             "'mean', 'mean_std', 'last', 'attention'."
        )


        self.pool = pool
        self.out_dim = out_dim
        self.out_dim_final = out_dim_final
        self.hidden_channels = hidden_channels
        self.frame_chunk = frame_chunk

        # ---------------------------------------------------------
        # 1. Shared spatial encoder
        # ---------------------------------------------------------
        # One ResNet3DEncoder is reused for every temporal volume.
        # The spatial encoder parameters are independent of T.
        #
        # The configuration follows the existing RS-data encoder:
        #   - one input channel
        #   - shallow ResNet (layer1 only)
        #   - customized 3D convolution kernels
        # ---------------------------------------------------------

        ## define here the shared ResNet3DEncoder
        self.spatial_encoder = ResNet3DEncoder(
            in_channels=1,
            emb_dim=emb_dim,
            out_dim=out_dim,
            pretrained=pretrained,
            use_stages=(True, False, False, False),
            conv_overrides=[
                {
                    "pattern": r"^layer1\.\d+\.conv1$",
                    "kernel_size": (3, 3, 3),
                    "padding": auto_padding((3, 3, 3)),
                },
                {
                    "pattern": r"^layer1\.\d+\.conv2$",
                    "kernel_size": (3, 3, 3),
                    "padding": auto_padding((3, 3, 3)),
                },
            ],
        )

        # ---------------------------------------------------------
        # 2. Temporal convolution blocks
        # ---------------------------------------------------------
        # Input features:  (B, out_dim, T)
        # Output features: (B, hidden_channels, T)
        #
        # Dilations:
        #   block 1 -> 1
        #   block 2 -> dilation_growth
        #   block 3 -> dilation_growth ** 2
        #
        # LayerNorm replaces temporal BatchNorm1d to avoid mixing
        # normalization statistics across padded timepoints.
        # ---------------------------------------------------------

        self.temporal_blocks = nn.ModuleList()

        in_channels = out_dim

        for block_idx in range(n_blocks):

            ## use different 1d convolutions for each block inside
            dilation = dilation_growth ** block_idx
            padding = (kernel_size // 2) * dilation

            block = nn.ModuleDict(
                {
                    "conv": nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    "norm": nn.LayerNorm(hidden_channels),
                    "activation": nn.GELU(),
                }
            )

            self.temporal_blocks.append(block)
            in_channels = hidden_channels

        # ---------------------------------------------------------
        # 3. Temporal pooling definition
        # ---------------------------------------------------------
        # Supported pooling strategies:
        #
        # "mean"      : masked temporal mean
        # "mean_std"  : masked mean and standard deviation
        # "last"      : last valid temporal feature
        # "attention" : learnable query-based temporal attention
        #
        # Attention operates across timepoints WITHIN each subject,
        # never across subjects.
        # ---------------------------------------------------------

        if self.pool == "mean_std":
            pooled_dim = hidden_channels * 2
        else:
            pooled_dim = hidden_channels

        if self.pool == "attention":

            if hidden_channels % attention_heads != 0:
                raise ValueError(
                    "hidden_channels must be divisible "
                    "by attention_heads."
                )

        # One learnable query token, shared across subjects.
        # It learns which temporal features are important
        # for the final subject-level representation.
        self.attention_query = nn.Parameter(
                torch.randn(1, 1, hidden_channels) * 0.02
        )

        # Layer normalization before temporal attention.
        self.attention_norm = nn.LayerNorm(
                hidden_channels
        )

        # Cross-attention:
        # Query  : (B, 1, C)
        # Keys   : (B, T, C)
        # Values : (B, T, C)
        #
        # batch_first=True ensures that B remains
        # the independent subject dimension.
        self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_channels,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=True,
        )

        # ---------------------------------------------------------
        # 4. Final subject-level projection
        # ---------------------------------------------------------

        self.projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, out_dim_final),
        )


    def forward(
        self,
        x: torch.Tensor | list[torch.Tensor],
        lengths: torch.Tensor | None = None,
        return_sequence: bool = False,
    ):
        """
        Encode a batch of variable-length rs-fMRI acquisitions.

        Parameters
        ----------
        x : torch.Tensor or list[torch.Tensor]
            Tensor: (B, T, 1, D, H, W)
            List: T tensors of shape (B, 1, D, H, W).

        lengths : torch.Tensor, optional
            Valid temporal lengths, shape (B,).

        return_sequence : bool, default=False
            Whether to return temporal features and the validity mask.

        Returns
        -------
        embeddings : torch.Tensor
            Final subject embeddings: (B, out_dim_final).

        temporal_features : torch.Tensor, optional
            Temporal representations: (B, T, hidden_channels).

        mask : torch.Tensor, optional
            Boolean validity mask: (B, T).
        """

        # ---------------------------------------------------------
        # 1. Standardize input format
        # ---------------------------------------------------------

        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("Input sequence cannot be empty.")

            x = torch.stack(x, dim=1)

        if x.ndim != 6:
            raise ValueError(
                "Expected input shape (B, T, 1, D, H, W), "
                f"received {tuple(x.shape)}."
            )

        B, T, C, D, H, W = x.shape

        if B < 1 or T < 1:
            raise ValueError("Batch size and temporal length must be >= 1.")

        if C != 1:
            raise ValueError(
                f"Expected one rs-fMRI input channel, received {C}."
            )

        # ---------------------------------------------------------
        # 2. Construct temporal validity mask
        # ---------------------------------------------------------

        if lengths is None:
            lengths = torch.full(
                (B,),
                T,
                dtype=torch.long,
                device=x.device,
            )
        else:
            lengths = torch.as_tensor(
                lengths,
                dtype=torch.long,
                device=x.device,
            )

        if lengths.shape != (B,):
            raise ValueError(
                f"Expected lengths shape ({B},), "
                f"received {tuple(lengths.shape)}."
            )

        if torch.any(lengths < 1) or torch.any(lengths > T):
            raise ValueError(
                "All lengths must satisfy 1 <= length <= T."
            )

        time_index = torch.arange(
            T,
            device=x.device,
        ).unsqueeze(0)

        mask = time_index < lengths.unsqueeze(1)

        # mask: (B, T)

        # ---------------------------------------------------------
        # 3. Flatten batch and temporal dimensions
        # ---------------------------------------------------------
        # (B, T, 1, D, H, W)
        #             |
        #             v
        # (B*T, 1, D, H, W)
        # ---------------------------------------------------------

        frames = x.reshape(
            B * T,
            C,
            D,
            H,
            W,
        )

        flat_mask = mask.reshape(-1)

        valid_indices = torch.nonzero(
            flat_mask,
            as_tuple=False,
        ).flatten()

        # ---------------------------------------------------------
        # 4. Encode valid volumes with shared ResNet3D
        # ---------------------------------------------------------
        # Chunking reduces the number of 3D volumes processed
        # simultaneously, while preserving gradient propagation.
        # ---------------------------------------------------------

        spatial_embeddings = []

        for indices in valid_indices.split(self.frame_chunk):

            batch_frames = frames.index_select(
                0,
                indices,
            )

            z_chunk = self.spatial_encoder(batch_frames)

            spatial_embeddings.append(z_chunk)

        valid_embeddings = torch.cat(
            spatial_embeddings,
            dim=0,
        )

        # ---------------------------------------------------------
        # 5. Restore temporal sequence
        # ---------------------------------------------------------
        # Valid embeddings: (N_valid, out_dim)
        # Full sequence:    (B, T, out_dim)
        #
        # Padded positions receive zeros.
        # ---------------------------------------------------------

        seq_flat = valid_embeddings.new_zeros(
            (B * T, self.out_dim)
        )

        seq_flat = seq_flat.index_copy(
            0,
            valid_indices,
            valid_embeddings,
        )

        sequence = seq_flat.reshape(
            B,
            T,
            self.out_dim,
        )

        # ---------------------------------------------------------
        # 6. Temporal convolution
        # ---------------------------------------------------------
        # Conv1d expects:
        #     (B, channels, T)
        #
        # The validity mask is applied after each block so padded
        # features cannot propagate into later convolutions.
        # ---------------------------------------------------------

        y = sequence.transpose(1, 2)

        temporal_mask = mask.unsqueeze(1).to(
            dtype=y.dtype
        )

        for block in self.temporal_blocks:

            y = block["conv"](
                y * temporal_mask
            )

            # LayerNorm operates on the feature dimension.
            y = y.transpose(1, 2)

            y = block["norm"](y)
            y = block["activation"](y)

            y = y.transpose(1, 2)

            # Keep padded positions equal to zero.
            y = y * temporal_mask

        # y: (B, hidden_channels, T)

        # ---------------------------------------------------------
        # 7. Variable-length temporal pooling
        # ---------------------------------------------------------

        if self.pool == "mean":

            valid_counts = lengths.to(
                dtype=y.dtype
            ).unsqueeze(1)

            pooled = y.sum(dim=2) / valid_counts

        elif self.pool == "mean_std":

            valid_counts = lengths.to(
                dtype=y.dtype
            ).unsqueeze(1)

            # Masked mean
            mean = y.sum(dim=2) / valid_counts

            # Masked population variance
            centered = (
                y - mean.unsqueeze(-1)
            ) * temporal_mask

            variance = centered.square().sum(
                dim=2
            ) / valid_counts

            std = torch.sqrt(
                variance + 1e-6
            )

            pooled = torch.cat(
                [mean, std],
                dim=1,
            )

        elif self.pool == "attention":

            # -----------------------------------------------------
            # Learnable temporal attention pooling
            # -----------------------------------------------------
            # TCN output:
            #     y: (B, hidden_channels, T)
            #
            # MultiheadAttention expects:
            #     (B, T, hidden_channels)
            # -----------------------------------------------------

            temporal_features = y.transpose(1, 2)

            # Normalize temporal feature vectors.
            temporal_features = self.attention_norm(
                temporal_features
            )

            # Expand the learned query independently for
            # every subject in the batch.
            #
            # query: (B, 1, hidden_channels)
            query = self.attention_query.expand(
                B, -1, -1
            )

            # True indicates a padded position that
            # attention must ignore.
            #
            # key_padding_mask: (B, T)
            key_padding_mask = ~mask

            # -----------------------------------------------------
            # Cross-attention across temporal positions
            # -----------------------------------------------------
            # Query = learned subject-summary token
            # Key   = complete temporal feature sequence
            # Value = complete temporal feature sequence
            #
            # This is NOT attention over the batch.
            # -----------------------------------------------------

            pooled, _ = self.temporal_attention(
                query=query,
                key=temporal_features,
                value=temporal_features,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

            # Remove the single query-token dimension:
            #
            # (B, 1, hidden_channels)
            #             |
            #             v
            # (B, hidden_channels)
            pooled = pooled.squeeze(1)


        else:
            # Last valid timepoint, not the last padded position.
            last_indices = lengths - 1

            batch_indices = torch.arange(
                B,
                device=x.device,
            )

            pooled = y.transpose(1, 2)[
                batch_indices,
                last_indices,
            ]

        # ---------------------------------------------------------
        # 8. Final projection
        # ---------------------------------------------------------

        embeddings = self.projection(pooled)

        # embeddings: (B, out_dim_final)

        # ---------------------------------------------------------
        # 9. Optional temporal features
        # ---------------------------------------------------------

        if return_sequence:

            temporal_features = y.transpose(1, 2)

            return embeddings, temporal_features, mask

        return embeddings


if __name__ == "__main__":
    """
       ** Main section of the code here ** RUN THESE AS EXAMPLES**
    """
    enc_model_base = ResNet3DEncoder(
        in_channels=1,
        use_stages=(True, True, False, False),  # layer1+layer2 only
        out_dim=256,
    )

    ## define here the variable length ResNet TCN for RSData
    enc_4D_rsdata = SharedResNet4D_TemporalConv(
        emb_dim=256,
        out_dim=128,
        pretrained=False,
        kernel_size=5,
        n_blocks=3,
        hidden_channels=100,
        dilation_growth=2,
        pool="attention",
        attention_heads=4,
        attention_dropout=0.0,
        out_dim_final=128,
        frame_chunk=16,
    )

    enc_alt = ResNet3DEncoder(
        in_channels=1,
        # no depth mixing in the very first conv. This is for the first
        # convolution
        stem_kernel=(3, 7, 7),
        stem_stride=(2, 2, 2),
        # interrupting the action of some layers if conside or not
        use_stages=(True, True, False, False),
        out_dim=128,
        emb_dim=256,
        attn_heads=4,
        activation_function="silu",
        attention_projection=True,
        conv_overrides=[
            # alway define this kernel size and padding  as odd values NOT
            # event. Just work on the convolutional kernels
            {
                "pattern": r"^layer1\.\d+\.conv1$",
                "kernel_size": (1, 3, 3),
                "padding": auto_padding((1, 3, 3)),
            },
            {
                "pattern": r"^layer1\.\d+\.conv2$",
                "kernel_size": (5, 3, 3),
                "padding": auto_padding((5, 3, 3)),
            },
        ],
    )

    # define the 4D model
    enc_4D = LateFusion4DResNet(
        n_streams=50,
        emb_dim=128,
        fusion="concat",
        out_dim=64,
        output_dim=32,
        pretrained=False,
    )
