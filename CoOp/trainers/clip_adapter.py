import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX  # type: ignore
from dassl.metrics import compute_accuracy  # type: ignore
from dassl.utils import load_pretrained_weights, load_checkpoint  # type: ignore
from dassl.optim import build_optimizer, build_lr_scheduler  # type: ignore

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer  # type: ignore
from transformers import CLIPModel, CLIPTokenizerFast
from open_clip import get_tokenizer, create_model_from_pretrained
import conch.open_clip_custom


_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    "APACC": "a photo of a {}.",
    "BCFC": "a photo of a {}.",
    "BloodMNIST": "a photo of a {}.",
    "BMCD": "a photo of a {}.",
    "BMT": "a photo of a {}.",
    "EuroSAT": "a photo of a {}.",
    "FNAC2019": "a photo of a {}.",
    "Herlev": "a photo of a {}.",
    "HiCervix": "a photo of a {}.",
    "MLCC": "a photo of a {}.",
    "SiPakMed": "a photo of a {}.",
}


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_params_by_module(model: nn.Module, key="prompt_learner"):
    # utile pour vérifier que seul prompt_learner est entraîné
    sub = dict(model.named_modules()).get(key, None)
    if sub is None:
        return None
    total = sum(p.numel() for p in sub.parameters())
    trainable = sum(p.numel() for p in sub.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def vis_dim_from_encode_image_strict(model, image_size=224):
    was_training = model.training
    model.eval()

    p = next(model.parameters())
    orig_device, orig_dtype = p.device, p.dtype

    model = model.to("cpu")
    x = torch.zeros(1, 3, image_size, image_size, device="cpu", dtype=torch.float32)

    y = model.encode_image(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(y, dict):
        # au cas où
        y = y.get("image_features", next(iter(y.values())))
    if y.dim() != 2:
        y = y.reshape(y.size(0), -1)

    d = int(y.shape[-1])

    model.to(device=orig_device, dtype=orig_dtype)
    model.train(was_training)
    return d


def debug_sanity_check_adapter(trainer, batch):
    print("\n" + "=" * 100)
    print("🔍 SANITY CHECK CLIP-ADAPTER (RESIDUAL BOTTLENECK + UPDATES)")
    print("=" * 100)

    model = trainer.model
    img, label = trainer.parse_batch_train(batch)
    m = model.module if hasattr(model, "module") else model

    # ---------- helpers ----------
    def _get_backbone_image_features(m, img):
        """Compute image features WITHOUT adapter mixing (backbone-only)."""
        with torch.no_grad():
            if hasattr(m, "encode_image"):
                return m.encode_image(img)

            # fallback wrappers
            if hasattr(m, "biomed"):
                dt = next(m.biomed.parameters()).dtype
                return m.biomed.encode_image(img.to(dtype=dt))

            if hasattr(m, "clip_model") and hasattr(m.clip_model, "encode_image"):
                dt = next(m.clip_model.parameters()).dtype
                return m.clip_model.encode_image(img.to(dtype=dt))

            raise AttributeError(f"{type(m)} has no encode_image and no known fallback")

    def _adapter_out(m, feats):
        """Adapter(feats.float()) then cast back to feats.dtype (like forward)."""
        a = m.adapter(feats.float()).to(feats.dtype)
        return a

    def _mix(feats, a_out, ratio):
        return ratio * a_out + (1.0 - ratio) * feats

    # ---------- [0] Batch ----------
    print("\n[0] Batch")
    print("  img:", tuple(img.shape), img.dtype, img.device)
    print(
        "  label:",
        tuple(label.shape),
        label.dtype,
        label.device,
        "min/max:",
        int(label.min().item()),
        int(label.max().item()),
    )

    # ---------- [1] Trainable params ----------
    print("\n[1] Trainable parameters (should be ONLY adapter.*)")
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    print("  n_trainable:", len(trainable))
    for n, p in trainable:
        print(f"   ✔ {n:55s} shape={tuple(p.shape)} dtype={p.dtype} device={p.device}")

    names = [n for n, _ in trainable]
    assert len(names) > 0, "❌ No trainable parameters found"
    assert all(
        n.startswith("adapter.") or ".adapter." in n for n in names
    ), "❌ non-adapter param trainable"

    # ---------- [2] Is there a TEXT adapter? ----------
    # In your current code: NO. (text_encoder() returns features but no adapter applied)
    has_text_adapter = (
        hasattr(m, "text_adapter")
        or (hasattr(m, "adapter_text"))
        or (hasattr(m, "text_encoder") and hasattr(m.text_encoder, "adapter"))
    )
    print("\n[2] Text-adapter presence")
    print(
        "  has_text_adapter:",
        bool(has_text_adapter),
        " (expected: False with current implementation)",
    )

    # ---------- [3] Adapter architecture (bottleneck) ----------
    print("\n[3] Adapter architecture sanity")
    # Try to read Linear dims (bottleneck reduction)
    linears = [mod for mod in m.adapter.modules() if isinstance(mod, nn.Linear)]
    print("  n_linear_layers:", len(linears))
    if len(linears) >= 2:
        in0, out0 = linears[0].in_features, linears[0].out_features
        in1, out1 = linears[1].in_features, linears[1].out_features
        print("  linear1: in -> out =", in0, "->", out0)
        print("  linear2: in -> out =", in1, "->", out1)
        if out0 < in0:
            print("  ✔ bottleneck reduction detected (out0 < in0)")
        else:
            print("  ⚠ no reduction detected (check reduction factor)")
    else:
        print("  ⚠ could not infer bottleneck dims (unexpected adapter structure)")

    # ---------- [4] Forward eval -> logits ----------
    print("\n[4] Forward eval -> logits")
    was_training = m.training
    m.eval()
    with torch.no_grad():
        out = m(img)
    print(
        "  logits:",
        tuple(out.shape),
        out.dtype,
        out.device,
        "min/max:",
        float(out.min().item()),
        float(out.max().item()),
    )

    # ---------- [5] Verify residual mixing is actually used ----------
    print("\n[5] Residual mixing checks (image adapter is used)")
    m.eval()
    with torch.no_grad():
        feats = _get_backbone_image_features(m, img)  # backbone-only
        a_out = _adapter_out(m, feats)
        ratio = 0.2  # MUST match your forward
        mixed = _mix(feats, a_out, ratio)

        # sanity: mixed differs from feats if adapter is non-trivial
        d_mix = (mixed - feats).abs().mean().item()
        d_a = (a_out - feats).abs().mean().item()
        print("  image_features:", tuple(feats.shape), feats.dtype)
        print("  adapter_out:", tuple(a_out.shape), a_out.dtype)
        print("  mean|adapter_out - image_features|:", float(d_a))
        print("  mean|mixed - image_features|:", float(d_mix))

        assert d_mix > 0, "❌ mixed == feats (adapter not affecting features?)"
        # check exact formula numerically
        mixed_ref = ratio * a_out + (1.0 - ratio) * feats
        err = (mixed - mixed_ref).abs().max().item()
        print("  max|mixed - (ratio*a + (1-ratio)*feats)|:", float(err))
        assert err < 1e-6, "❌ residual mixing formula mismatch"

    # ---------- [7] Train forward -> backward: grads exist ----------
    print("\n[7] Train forward -> loss + backward (grads)")
    m.train()
    m.zero_grad(set_to_none=True)
    logits = m(img)
    loss = F.cross_entropy(logits, label)
    print("  loss:", float(loss.item()))
    loss.backward()

    # grads on adapter params only
    none_grads = []
    zeroish = []
    for n, p in m.adapter.named_parameters():
        if p.grad is None:
            none_grads.append(n)
        else:
            gm = p.grad.abs().mean().item()
            if gm == 0:
                zeroish.append(n)
    print("  adapter params with grad None:", none_grads)
    if len(none_grads) == 0:
        print("  ✔ all adapter params have grads")
    if len(zeroish) > 0:
        print("  ⚠ adapter params with grad mean == 0:", zeroish)

    assert len(none_grads) == 0, "❌ some adapter params have grad=None"

    # ---------- [8] Step check: adapter weights move ----------
    print("\n[8] Optim step check: adapter weights actually update")
    # snapshot
    w_before = {n: p.detach().clone() for n, p in m.adapter.named_parameters()}

    # do one tiny manual step with trainer's optimizer if exists,
    # otherwise fallback to a local SGD
    opt = getattr(trainer, "optim", None)
    if opt is None:
        opt = torch.optim.SGD(m.adapter.parameters(), lr=1e-3)

    opt.step()

    deltas = {}
    for n, p in m.adapter.named_parameters():
        deltas[n] = (p.detach() - w_before[n]).abs().mean().item()

    # print a few
    for n, d in list(deltas.items())[:6]:
        print(f"  Δ{n} (mean abs): {d:.3e}")

    moved = sum(d > 0 for d in deltas.values())
    print("  n_params_moved:", moved, "/", len(deltas))
    assert moved > 0, "❌ adapter weights did not update after step"

    m.train(was_training)

    print("\n✅ SANITY CHECK DONE")
    print("=" * 100 + "\n")


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME

    if backbone_name == "BiomedCLIP":
        model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

        model, _ = create_model_from_pretrained(model_id, device="cpu")

        return model

    if backbone_name == "Quilt-B/32":
        model_id = "hf-hub:wisdomik/QuiltNet-B-32"

        model, _ = create_model_from_pretrained(model_id, device="cpu")

        return model

    if backbone_name == "Quilt-B/16":
        model_id = "hf-hub:wisdomik/QuiltNet-B-16"

        model, _ = create_model_from_pretrained(model_id, device="cpu")

        return model

    if backbone_name == "PubMedCLIP-B/32":
        model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"

        model = CLIPModel.from_pretrained(model_id)

        return model

    if backbone_name == "PLIP-B/32":
        model_id = "vinid/plip"

        model = CLIPModel.from_pretrained(model_id)

        return model

    if backbone_name == "Conch":
        model, preprocess = conch.open_clip_custom.create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token="REMOVED_TOKEN",
        )

        return model

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self):
        device = next(self.clip_model.parameters()).device
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        vis_dim = int(self.image_encoder.output_dim)
        self.adapter = Adapter(vis_dim, 4)

    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        first_linear = next(
            m for m in self.adapter.modules() if isinstance(m, nn.Linear)
        )
        if first_linear.in_features != image_features.shape[-1]:
            raise RuntimeError(
                f"[DIM MISMATCH] adapter expects {first_linear.in_features} "
                f"but image_features is {image_features.shape[-1]}"
            )

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomBiomedCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomed_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.biomed = biomed_model
        self.tokenizer = tokenizer

        vis_dim = vis_dim_from_encode_image_strict(
            self.biomed, image_size=cfg.INPUT.SIZE[0]
        )
        self.adapter = Adapter(vis_dim, 4)

        self.logit_scale = getattr(self.biomed, "logit_scale", None)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        tok = tokenizer(prompts)
        if isinstance(tok, dict):
            tok = tok.get("input_ids", list(tok.values())[0])
        if hasattr(tok, "input_ids"):   # tokenizers.Encoding / BatchEncoding-like
            tok = tok.input_ids
        tok = torch.as_tensor(tok, dtype=torch.long)

        self.register_buffer("tokenized_prompts", tok, persistent=False)

    def encode_image(self, image):
        dt = next(self.biomed.parameters()).dtype
        return self.biomed.encode_image(image.to(dtype=dt))

    def forward(self, image):
        device = image.device

        dt = next(self.biomed.parameters()).dtype
        image_features = self.biomed.encode_image(image.to(dtype=dt))
        x = self.adapter(image_features.float()).to(image_features.dtype)

        first_linear = next(
            m for m in self.adapter.modules() if isinstance(m, nn.Linear)
        )
        if first_linear.in_features != image_features.shape[-1]:
            raise RuntimeError(
                f"[DIM MISMATCH] adapter expects {first_linear.in_features} "
                f"but image_features is {image_features.shape[-1]}"
            )

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        tokenized = self.tokenized_prompts.to(device)
        text_features = self.biomed.encode_text(tokenized)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(
                1 / 0.07, device=device, dtype=image_features.dtype
            )
        else:
            logit_scale = self.logit_scale.exp().to(
                device=device, dtype=image_features.dtype
            )

        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomQuiltCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.logit_scale = getattr(clip_model, "logit_scale", None)
        self.dtype = next(self.clip_model.parameters()).dtype

        vis_dim = vis_dim_from_encode_image_strict(
            self.clip_model, image_size=cfg.INPUT.SIZE[0]
        )
        self.adapter = Adapter(vis_dim, 4)

        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        tok = tokenizer(prompts)
        if isinstance(tok, dict):
            tok = torch.as_tensor(tok["input_ids"])
        else:
            tok = torch.as_tensor(tok)

        self.register_buffer("tokenized_prompts", tok, persistent=False)

    def encode_image(self, image):
        return self.image_encoder(image.to(dtype=self.dtype))

    def forward(self, image):
        device = image.device
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        first_linear = next(
            m for m in self.adapter.modules() if isinstance(m, nn.Linear)
        )
        if first_linear.in_features != image_features.shape[-1]:
            raise RuntimeError(
                f"[DIM MISMATCH] adapter expects {first_linear.in_features} "
                f"but image_features is {image_features.shape[-1]}"
            )

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features
        tokenized_prompts = self.tokenized_prompts.to(device)
        text_features = self.clip_model.encode_text(tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logits
        if self.logit_scale is None:
            logit_scale = torch.tensor(1 / 0.07, device=device)  # constant
        else:
            logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomPubMedCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.classnames = classnames
        self.image_encoder = clip_model.get_image_features

        self.logit_scale = getattr(clip_model, "logit_scale", None)
        self.vision_dtype = next(self.clip_model.vision_model.parameters()).dtype
        vis_dim = int(self.clip_model.config.projection_dim)
        self.adapter = Adapter(vis_dim, 4)

        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        tok = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        self.register_buffer("input_ids", tok["input_ids"], persistent=False)
        self.register_buffer("attention_mask", tok["attention_mask"], persistent=False)

    def encode_image(self, image):
        return self.clip_model.get_image_features(
            pixel_values=image.to(dtype=self.vision_dtype)
        )

    def forward(self, image):
        device = image.device
        image_features = self.clip_model.get_image_features(
            pixel_values=image.to(dtype=self.vision_dtype)
        )
        x = self.adapter(image_features)

        first_linear = next(
            m for m in self.adapter.modules() if isinstance(m, nn.Linear)
        )
        if first_linear.in_features != image_features.shape[-1]:
            raise RuntimeError(
                f"[DIM MISMATCH] adapter expects {first_linear.in_features} "
                f"but image_features is {image_features.shape[-1]}"
            )

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.clip_model.get_text_features(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # ---- logit scale ----
        if self.logit_scale is None:
            logit_scale = torch.tensor(
                1 / 0.07, device=device, dtype=text_features.dtype
            )
        else:
            logit_scale = self.logit_scale.exp().to(
                device=device, dtype=text_features.dtype
            )

        logits = logit_scale * image_features @ text_features.t()

        return logits


class CustomConchCLIP(nn.Module):
    def __init__(self, cfg, classnames, conch_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = conch_model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.image_encoder = conch_model.encode_image

        self.logit_scale = getattr(conch_model, "logit_scale", None)
        self.dtype = next(self.clip_model.parameters()).dtype
        vis_dim = None
        if hasattr(conch_model, "visual") and hasattr(conch_model.visual, "output_dim"):
            vis_dim = int(conch_model.visual.output_dim)
        if vis_dim is None:
            vis_dim = vis_dim_from_encode_image_strict(
                conch_model, image_size=cfg.INPUT.SIZE[0]
            )

        self.adapter = Adapter(vis_dim, 4)

        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        tok = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )[
            "input_ids"
        ]  # (n_cls, 77)
        tok = torch.as_tensor(tok, dtype=torch.long)
        self.register_buffer("tokenized_prompts", tok, persistent=False)

    def encode_image(self, image):
        return self.image_encoder(image.to(dtype=self.dtype))

    def forward(self, image):
        device = image.device
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features.float()).to(image_features.dtype)

        first_linear = next(
            m for m in self.adapter.modules() if isinstance(m, nn.Linear)
        )
        if first_linear.in_features != image_features.shape[-1]:
            raise RuntimeError(
                f"[DIM MISMATCH] adapter expects {first_linear.in_features} "
                f"but image_features is {image_features.shape[-1]}"
            )

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        tokenized_prompts = self.tokenized_prompts.to(device)
        text_features = self.clip_model.encode_text(tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # ---- logit scale ----
        if self.logit_scale is None:
            logit_scale = torch.tensor(
                1 / 0.07, device=device, dtype=text_features.dtype
            )
        else:
            logit_scale = self.logit_scale.exp().to(
                device=device, dtype=text_features.dtype
            )

        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CLIP_Adapter(TrainerX):
    """CLIP-Adapter"""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print("Building custom CLIP")
        if cfg.MODEL.BACKBONE.NAME == "BiomedCLIP":
            model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            tokenizer = get_tokenizer(model_id)
            self.model = CustomBiomedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Quilt-B/32":
            tokenizer = get_tokenizer("hf-hub:wisdomik/QuiltNet-B-32")
            self.model = CustomQuiltCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Quilt-B/16":
            tokenizer = get_tokenizer("hf-hub:wisdomik/QuiltNet-B-16")
            self.model = CustomQuiltCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "PubMedCLIP-B/32":
            tokenizer = CLIPTokenizerFast.from_pretrained(
                "flaviagiammarino/pubmed-clip-vit-base-patch32"
            )
            self.model = CustomPubMedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "PLIP-B/32":
            tokenizer = CLIPTokenizerFast.from_pretrained("vinid/plip")
            self.model = CustomPubMedCLIP(cfg, classnames, clip_model, tokenizer)

        elif cfg.MODEL.BACKBONE.NAME == "Conch":
            tokenizer = conch.open_clip_custom.get_tokenizer()
            self.model = CustomConchCLIP(cfg, classnames, clip_model, tokenizer)

        else:
            self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # Count of parameters
        m = self.model.module if hasattr(self.model, "module") else self.model

        first = next(mod for mod in m.adapter.modules() if isinstance(mod, nn.Linear))
        print("[DEBUG] USING FILE:", __file__)
        print("[DEBUG] adapter.in_features =", first.in_features)
        print("[DEBUG] model class =", type(m))

        tot, tr = count_params(m)
        print(
            f"[PARAMS] total={tot:,} | trainable={tr:,} | trainable%={100*tr/tot:.4f}%"
        )

        pl = count_params_by_module(m, "adapter")
        if pl is not None:
            print(f"[PARAMS] adapter total/trainable = {pl[0]:,} / {pl[1]:,}")

        # Debug - sanity check
        batch_debug = next(iter(self.train_loader_x))
        debug_sanity_check_adapter(self, batch_debug)

        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("clip_adapter", self.model.adapter, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
