import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from transformers import CLIPModel, CLIPTokenizerFast
from open_clip import get_tokenizer, create_model_from_pretrained
import conch.open_clip_custom


_tokenizer = _Tokenizer()


def _build_causal_mask(seq_len: int, device, dtype):
    # mask additive (seq_len, seq_len) with -inf above diagonal
    # works for attention implementations that expect additive mask
    m = torch.empty(seq_len, seq_len, device=device, dtype=dtype)
    m.fill_(float("-inf"))
    m.triu_(1)
    return m


def _get_resblocks(module):
    # open_clip / CLIP style
    if hasattr(module, "resblocks"):
        return module.resblocks
    # some vit impls use "blocks"
    if hasattr(module, "blocks"):
        return module.blocks
    raise AttributeError(f"Could not find resblocks/blocks in {type(module)}")


def _infer_visual_width(visual) -> int:
    """
    Renvoie la *token dim* du ViT (celle des patch tokens / cls token),
    pas l'output_dim CLIP après projection.
    """
    # 1) open_clip timm wrapper: visual.trunk est un timm VisionTransformer
    if hasattr(visual, "trunk"):
        trunk = visual.trunk

        # timm ViT: cls_token est (1,1,D)
        if hasattr(trunk, "cls_token") and trunk.cls_token is not None:
            return int(trunk.cls_token.shape[-1])

        # sinon: patch_embed.proj.out_channels = D
        if hasattr(trunk, "patch_embed") and hasattr(trunk.patch_embed, "proj"):
            proj = trunk.patch_embed.proj
            if hasattr(proj, "out_channels"):
                return int(proj.out_channels)

        # fallback timm attrs
        for attr in ["embed_dim", "num_features"]:
            if hasattr(trunk, attr):
                v = getattr(trunk, attr)
                if isinstance(v, int) and v > 0:
                    return int(v)

    # 2) open_clip VisionTransformer direct
    # class_embedding est (D,) => token dim
    if hasattr(visual, "class_embedding") and visual.class_embedding is not None:
        return int(visual.class_embedding.shape[0])

    # positional_embedding est (1+n_patches, D) ou (1,1+n_patches,D)
    pe = getattr(visual, "positional_embedding", None)
    if pe is not None and torch.is_tensor(pe):
        return int(pe.shape[-1])

    # conv1 patchify (B,D,grid,grid)
    conv1 = getattr(visual, "conv1", None)
    if conv1 is not None and hasattr(conv1, "out_channels"):
        return int(conv1.out_channels)

    raise AttributeError(f"Could not infer visual token dim from {type(visual)}")


def _get_openclip_text_module(model):
    # cas "classique"
    if hasattr(model, "text"):
        return model.text

    # beaucoup de modèles open_clip ont juste transformer + embeddings au top-level
    # (CustomTextCLIP / CLIP-like)
    if hasattr(model, "transformer") and hasattr(model, "token_embedding"):
        return model  # on traitera model comme "text_mod"

    # parfois c'est text_encoder (selon versions / wrappers)
    if hasattr(model, "text_encoder"):
        return model.text_encoder

    raise AttributeError(
        f"open_clip model has no .text (type={type(model)}). "
        f"Available attrs sample: {list(vars(model).keys())[:40]}"
    )


def _get_openclip_token_embedding(model):
    # token_embedding souvent au top-level
    if hasattr(model, "token_embedding"):
        return model.token_embedding

    # sinon dans model.text ou model.text_encoder
    for attr in ["text", "text_encoder"]:
        if hasattr(model, attr):
            tm = getattr(model, attr)
            if hasattr(tm, "token_embedding"):
                return tm.token_embedding
            if hasattr(tm, "transformer") and hasattr(
                tm.transformer, "token_embedding"
            ):
                return tm.transformer.token_embedding

    raise AttributeError("Could not find token_embedding in open_clip model")


def _get_openclip_positional_embedding(text_mod):
    if hasattr(text_mod, "positional_embedding"):
        return text_mod.positional_embedding
    if hasattr(text_mod, "transformer") and hasattr(
        text_mod.transformer, "positional_embedding"
    ):
        return text_mod.transformer.positional_embedding
    return None


def _get_openclip_text_transformer(text_mod):
    # CLIP-like: transformer au même niveau
    if hasattr(text_mod, "transformer"):
        return text_mod.transformer
    # parfois appelé text_transformer
    if hasattr(text_mod, "text_transformer"):
        return text_mod.text_transformer
    raise AttributeError(f"Could not find text transformer in {type(text_mod)}")


def _get_openclip_ln_final(text_mod):
    if hasattr(text_mod, "ln_final"):
        return text_mod.ln_final
    if hasattr(text_mod, "transformer") and hasattr(text_mod.transformer, "ln_final"):
        return text_mod.transformer.ln_final
    raise AttributeError("Could not find ln_final for open_clip text")


def _get_openclip_text_projection(model, text_mod):
    # ordre important: model d'abord (souvent là)
    for obj in [model, text_mod]:
        for attr in ["text_projection", "proj", "projection"]:
            if hasattr(obj, attr):
                return getattr(obj, attr)
    return None


def _to_device_tokenized(tokenized, device):
    # transformers BatchEncoding has .to()
    if hasattr(tokenized, "to") and not isinstance(tokenized, torch.Tensor):
        try:
            return tokenized.to(device)
        except Exception:
            pass

    if isinstance(tokenized, torch.Tensor):
        return tokenized.to(device)

    if isinstance(tokenized, dict):
        return {
            k: (v.to(device) if torch.is_tensor(v) else torch.as_tensor(v).to(device))
            for k, v in tokenized.items()
        }

    if hasattr(tokenized, "items"):  # dict-like fallback
        d = dict(tokenized)
        return {
            k: (v.to(device) if torch.is_tensor(v) else torch.as_tensor(v).to(device))
            for k, v in d.items()
        }

    raise TypeError(f"Unexpected tokenized type: {type(tokenized)}")


def _count_trainable(model):
    n = 0
    names = []
    for k, p in model.named_parameters():
        if p.requires_grad:
            n += p.numel()
            names.append(k)
    return n, names


def safe_logit_scale_exp(logit_scale_param, device, dtype=torch.float32):
    """
    Clamp en log-space pour éviter overflow exp().
    """
    if logit_scale_param is None:
        return torch.tensor(1 / 0.07, device=device, dtype=dtype)
    x = logit_scale_param
    if torch.is_tensor(x):
        x = x.to(device=device, dtype=torch.float32).clamp(-10.0, 10.0)
        return x.exp().to(dtype=dtype)
    return torch.tensor(1 / 0.07, device=device, dtype=dtype)


def sanity_check_maple_unified(trainer, batch, print_trainables=True, check_backward=True):
    print("\n" + "=" * 100)
    print("🔍 SANITY CHECK MaPLe (UNIFIED)")
    print("=" * 100)

    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    model.eval()
    print("  grad enabled at entry:", torch.is_grad_enabled())

    img, label = trainer.parse_batch_train(batch)
    B = img.size(0)
    n_cls = len(trainer.dm.dataset.classnames)

    print("\n[0] Batch")
    print("  img:", tuple(img.shape), img.dtype, img.device)
    print("  label:", tuple(label.shape), label.dtype, label.device,
          "min/max:", int(label.min()), int(label.max()))

    assert label.min() >= 0 and label.max() < n_cls, "❌ labels out of range"

    print("\n[1] Trainables scope")
    n_trainable, trainable_names = _count_trainable(model)
    assert n_trainable > 0, "❌ nothing is trainable"

    outside = [n for n in trainable_names if "prompt_learner" not in n]
    if len(outside) > 0:
        print("❌ Trainable params outside prompt_learner:")
        for n in outside:
            print("  -", n)
        raise AssertionError("trainables must be in prompt_learner only")

    print("  n_trainable params:", n_trainable)
    if print_trainables:
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"   ✔ {n:70s} {tuple(p.shape)} {p.dtype} {p.device}")

    print("\n[2] Prompt learner outputs")
    assert hasattr(model, "prompt_learner"), "❌ model has no prompt_learner"
    pl = model.prompt_learner

    # Forward prompt learner only
    prompts, shared_ctx, deep_text, deep_vision = pl()

    # prompts: (n_cls, L, ctx_dim_text)
    assert torch.is_tensor(prompts), "❌ prompts must be a tensor"
    print("  prompts:", tuple(prompts.shape), prompts.dtype, prompts.device)

    if shared_ctx is not None:
        if torch.is_tensor(shared_ctx):
            print("  shared_ctx:", tuple(shared_ctx.shape), shared_ctx.dtype, shared_ctx.device)
        else:
            print("  shared_ctx: type=", type(shared_ctx))
    else:
        print("  shared_ctx: None")

    # deep prompts lists
    if deep_text is not None:
        print("  deep_text: len =", len(deep_text), "| elem shape =", tuple(deep_text[0].shape))
        for i, p in enumerate(deep_text):
            assert torch.is_tensor(p), "❌ deep_text entries must be tensors"
    else:
        print("  deep_text: None")

    if deep_vision is not None:
        print("  deep_vision: len =", len(deep_vision), "| elem shape =", tuple(deep_vision[0].shape))
        for i, p in enumerate(deep_vision):
            assert torch.is_tensor(p), "❌ deep_vision entries must be tensors"
    else:
        print("  deep_vision: None")

    # tokenized prompts
    tok = getattr(pl, "tokenized_prompts", None)
    assert tok is not None, "❌ prompt_learner has no tokenized_prompts"
    tok = tok.to(img.device)
    assert tok.dtype in (torch.int64, torch.long), f"❌ tokenized_prompts must be long, got {tok.dtype}"
    print("  tokenized_prompts:", tuple(tok.shape), tok.dtype, tok.device)

    print("\n[3] Eval forward -> logits")
    out = model(img)  # eval path should return logits
    if isinstance(out, (tuple, list)):
        out = out[0]
    print("  logits:", tuple(out.shape), out.dtype, out.device)

    assert out.ndim == 2, "❌ logits must be (B, n_cls)"
    assert out.shape == (B, n_cls), f"❌ logits shape mismatch: got {tuple(out.shape)} expected {(B, n_cls)}"
    assert torch.isfinite(out).all(), "❌ logits contains NaN/Inf"

    # check logit_scale if present
    if hasattr(model, "logit_scale") and torch.is_tensor(model.logit_scale):
        ls = float(model.logit_scale.detach().cpu())
        print("  logit_scale (raw):", ls, "| exp(clamped):", float(safe_logit_scale_exp(model.logit_scale, img.device)))

    if not check_backward:
        print("\n✅ SANITY CHECK PASSED (no backward)")
        print("=" * 100 + "\n")
        return True

    print("\n[4] Train forward -> loss + backward")
    model.train()
    model.zero_grad(set_to_none=True)

    print("  grad enabled BEFORE loss:", torch.is_grad_enabled())

    with torch.enable_grad():  # ✅ force grad même si caller est en no_grad
        loss = model(img, label)  # train path should return CE loss
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        print("  loss:", float(loss.item()), loss.dtype, loss.device, "| requires_grad:", loss.requires_grad)

        assert torch.isfinite(loss).all(), "❌ loss is NaN/Inf"
        assert loss.requires_grad, "❌ loss has no grad (graph broken?)"

        loss.backward()

    tot = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            tot += float(p.grad.abs().sum().item())
    print("  total grad abs sum =", tot)

    print("\n[5] Grad check (trainables only)")
    bad = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            print(f"   ❌ {n}: grad None")
            bad += 1
            continue
        g = p.grad
        if not torch.isfinite(g).all():
            print(f"   ❌ {n}: grad has NaN/Inf")
            bad += 1
            continue
        gmean = g.abs().mean().item()
        gmax = g.abs().max().item()
        print(f"   ✔ {n:70s} grad mean={gmean:.3e} max={gmax:.3e}")
        if gmean == 0.0:
            print(f"     ⚠️ {n}: grad mean is 0 (maybe frozen path / no signal)")

    assert bad == 0, "❌ some trainables have missing/non-finite grads"

    print("\n✅ SANITY CHECK PASSED")
    print("=" * 100 + "\n")
    return True


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

    design_details = {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.MAPLE.N_CTX,
    }

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def maple_text_forward_clip_like(
    text_transformer, x, deep_prompts, counter_start=0, causal=True
):
    """
    text_transformer: module containing resblocks (ResidualAttentionBlock list)
    x: (seq_len, batch, dim)  (CLIP convention)
    deep_prompts: list/ParameterList length = DEPTH-1, each (n_ctx, dim)
    counter_start: usually 0
    causal: True for text
    """
    resblocks = _get_resblocks(text_transformer)
    device, dtype = x.device, x.dtype

    counter = counter_start
    for i, blk in enumerate(resblocks):
        # inject deep prompt for layers 1..depth-1 (like MaPLe)
        if (deep_prompts is not None) and (counter < len(deep_prompts)):
            p = deep_prompts[counter]  # (n_ctx, dim)
            # expand to (n_ctx, batch, dim)
            p = p.unsqueeze(1).expand(-1, x.size(1), -1).to(device=device, dtype=dtype)

            # insert after first token (SOS/CLS) -> positions: [0] [PROMPTS...] [1..]
            x = torch.cat([x[:1], p, x[1:]], dim=0)

            # build causal mask for new length if needed
            if causal:
                attn_mask = _build_causal_mask(x.size(0), device, dtype=torch.float32)
            else:
                attn_mask = None

            # forward
            try:
                x = blk(x, attn_mask=attn_mask)
            except TypeError:
                # some impls use (x, attn_mask)
                x = blk(x, attn_mask)

            # remove injected tokens (keep original length)
            n_ctx = p.size(0)
            x = torch.cat([x[:1], x[1 + n_ctx :]], dim=0)

            counter += 1
        else:
            # no deep prompts
            if causal:
                attn_mask = _build_causal_mask(x.size(0), device, dtype=torch.float32)
            else:
                attn_mask = None

            try:
                x = blk(x, attn_mask=attn_mask)
            except TypeError:
                x = blk(x, attn_mask)

    return x


def maple_text_forward_clip_like_batch_first(
    text_transformer, x, deep_prompts, counter_start=0, causal=True
):
    """
    x: (N, L, D)  batch_first=True
    deep_prompts: list of (n_ctx, D)
    """
    resblocks = _get_resblocks(text_transformer)
    device, dtype = x.device, x.dtype

    counter = counter_start
    for blk in resblocks:

        if (deep_prompts is not None) and (counter < len(deep_prompts)):
            p = deep_prompts[counter]  # (n_ctx, D)
            p = (
                p.unsqueeze(0).expand(x.size(0), -1, -1).to(device=device, dtype=dtype)
            )  # (N,n_ctx,D)

            # insert after token 0
            x = torch.cat([x[:, :1], p, x[:, 1:]], dim=1)

            if causal:
                attn_mask = _build_causal_mask(
                    x.size(1), device, dtype=torch.float32
                )  # (L,L)
            else:
                attn_mask = None

            try:
                x = blk(x, attn_mask=attn_mask)
            except TypeError:
                x = blk(x, attn_mask)

            # strip prompts
            n_ctx = p.size(1)
            x = torch.cat([x[:, :1], x[:, 1 + n_ctx :]], dim=1)

            counter += 1

        else:
            if causal:
                attn_mask = _build_causal_mask(x.size(1), device, dtype=torch.float32)
            else:
                attn_mask = None

            try:
                x = blk(x, attn_mask=attn_mask)
            except TypeError:
                x = blk(x, attn_mask)

    return x


def maple_vision_forward_vit_like(vit_transformer, x, deep_prompts, counter_start=0):
    """
    vit_transformer: module containing blocks/resblocks
    x: (seq_len, batch, dim)  (CLIP ViT convention after patch embedding)
    deep_prompts: list length DEPTH-1, each (n_ctx, dim_vis)
    """
    blocks = _get_resblocks(vit_transformer)
    device, dtype = x.device, x.dtype
    counter = counter_start

    for i, blk in enumerate(blocks):
        if (deep_prompts is not None) and (counter < len(deep_prompts)):
            p = deep_prompts[counter]  # (n_ctx, dim)
            p = p.unsqueeze(1).expand(-1, x.size(1), -1).to(device=device, dtype=dtype)

            x = torch.cat([x[:1], p, x[1:]], dim=0)

            # forward (no mask)
            x = blk(x)

            # strip prompts
            n_ctx = p.size(0)
            x = torch.cat([x[:1], x[1 + n_ctx :]], dim=0)

            counter += 1
        else:
            x = blk(x)

    return x


def maple_text_forward_bert_like(
    bert_model, hidden_states, attention_mask, deep_prompts
):
    """
    bert_model: HF BertModel (ou similaire) -> bert_model.encoder.layer list
    hidden_states: (B,L,D)
    attention_mask: (B,L) 1=keep
    deep_prompts: list length DEPTH-1 of (n_ctx, D)
    """
    device, dtype = hidden_states.device, hidden_states.dtype
    layers = bert_model.encoder.layer

    counter = 0
    for layer in layers:
        if deep_prompts is not None and counter < len(deep_prompts):
            p = deep_prompts[counter].to(device=device, dtype=dtype)  # (n_ctx, D)
            p = p.unsqueeze(0).expand(hidden_states.size(0), -1, -1)  # (B,n_ctx,D)

            # insert after CLS
            hidden_states = torch.cat(
                [hidden_states[:, :1], p, hidden_states[:, 1:]], dim=1
            )

            # extend mask
            am = torch.cat(
                [
                    attention_mask[:, :1],
                    torch.ones(
                        attention_mask.size(0),
                        p.size(1),
                        device=device,
                        dtype=attention_mask.dtype,
                    ),
                    attention_mask[:, 1:],
                ],
                dim=1,
            )

            # HF expects extended attention mask (B,1,1,L)
            ext = bert_model.get_extended_attention_mask(am, am.shape, device=device)

            out = layer(hidden_states, attention_mask=ext)
            hidden_states = out[0]

            # remove prompts
            hidden_states = torch.cat(
                [hidden_states[:, :1], hidden_states[:, 1 + p.size(1) :]], dim=1
            )

            counter += 1
        else:
            ext = bert_model.get_extended_attention_mask(
                attention_mask, attention_mask.shape, device=device
            )
            out = layer(hidden_states, attention_mask=ext)
            hidden_states = out[0]

    return hidden_states


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [
            x,
            compound_prompts_deeper_text,
            0,
        ]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


def _apply_projection(x, proj):
    """
    proj peut être:
      - None
      - torch.Tensor (matrice)
      - nn.Module (Linear/Sequential/...)
    """
    if proj is None:
        return x
    if torch.is_tensor(proj):
        return x @ proj
    if isinstance(proj, nn.Module):
        return proj(x)
    raise TypeError(f"Unsupported projection type: {type(proj)}")


class TextEncoderOpenCLIP(nn.Module):
    def __init__(self, openclip_model):
        super().__init__()
        self.model = openclip_model
        self.text = _get_openclip_text_module(openclip_model)

        self.token_embedding = _get_openclip_token_embedding(openclip_model)
        self.positional_embedding = _get_openclip_positional_embedding(self.text)
        self.transformer = _get_openclip_text_transformer(self.text)
        self.ln_final = _get_openclip_ln_final(self.text)
        self.text_projection = _get_openclip_text_projection(openclip_model, self.text)

        self.dtype = next(openclip_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts, deep_prompts):
        device = tokenized_prompts.device

        x = prompts.to(device=device, dtype=self.dtype)  # (N,L,D)

        if self.positional_embedding is not None:
            pe = self.positional_embedding
            # pe peut être (L,D) ou (1,L,D)
            if pe.dim() == 2:
                pe = pe.unsqueeze(0)
            x = x + pe[:, : x.shape[1], :].to(device=device, dtype=self.dtype)

        # transformer (batch_first ou pas)
        try:
            x = maple_text_forward_clip_like_batch_first(
                self.transformer, x, deep_prompts, 0, causal=True
            )
        except Exception:
            x2 = x.permute(1, 0, 2)  # (L,N,D)
            x2 = maple_text_forward_clip_like(
                self.transformer, x2, deep_prompts, 0, causal=True
            )
            x = x2.permute(1, 0, 2)

        x = self.ln_final(x).to(self.dtype)

        # --- EOT pooling robuste = dernière position non-pad
        pad_id = getattr(self.model, "pad_id", 0)
        pad_id = int(pad_id) if pad_id is not None else 0

        nonpad = (tokenized_prompts != pad_id).long()  # (N,L)
        lengths = nonpad.sum(dim=-1)  # (N,)
        eot = (lengths - 1).clamp(min=0)  # (N,)

        x = x[torch.arange(x.size(0), device=device), eot]  # (N,D)

        x = _apply_projection(x, self.text_projection)
        return x


class TextEncoderBiomed(nn.Module):
    """
    Pour BiomedCLIP open_clip HF-text (PubMedBERT).
    On récupère CLS en sortie, puis projection si elle existe.
    """

    def __init__(self, biomed_model):
        super().__init__()
        self.model = biomed_model
        self.text = biomed_model.text
        self.bert = self.text.transformer  # HF BertModel
        self.dtype = next(biomed_model.parameters()).dtype

        # embeddings
        self.word_embeddings = self.bert.embeddings.word_embeddings
        self.position_embeddings = self.bert.embeddings.position_embeddings
        self.token_type_embeddings = self.bert.embeddings.token_type_embeddings
        self.layernorm = self.bert.embeddings.LayerNorm
        self.dropout = self.bert.embeddings.dropout

        # projection (selon wrapper)
        self.text_projection = _get_openclip_text_projection(biomed_model, self.text)

    def forward(self, prompts, tokenized_prompts, deep_prompts):
        # prompts: (N,L,D) construit via word_embeddings + concat ctx etc.
        # tokenized_prompts: (N,L) ids
        device = prompts.device
        x = prompts.to(self.dtype)

        # attention mask = padding != 0 (classique HF). Si ton tokenizer a pad_id=0 c’est OK.
        attention_mask = (tokenized_prompts != 0).to(device=device)

        # positions + token types (0)
        seq_len = x.size(1)
        pos_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(x.size(0), -1)
        )
        tok_type = torch.zeros_like(tokenized_prompts, device=device)

        x = (
            x
            + self.position_embeddings(pos_ids).to(self.dtype)
            + self.token_type_embeddings(tok_type).to(self.dtype)
        )
        x = self.layernorm(x)
        x = self.dropout(x)

        x = maple_text_forward_bert_like(self.bert, x, attention_mask, deep_prompts)

        # CLS pooling
        x = x[:, 0]  # (N,D)

        x = _apply_projection(x, self.text_projection)

        return x


class TextEncoderHF(nn.Module):
    """
    Text encoder pour transformers.CLIPModel (PLIP / PubMedCLIP).
    Important: hf_model.text_model est un CLIPTextTransformer qui n'accepte PAS inputs_embeds,
    donc on appelle directement .encoder(...) avec nos embeddings.
    """

    def __init__(self, hf_model: CLIPModel):
        super().__init__()
        self.model = hf_model

        # CLIPTextTransformer
        self.text = hf_model.text_model

        # modules internes (noms HF standard)
        self.embeddings = self.text.embeddings  # CLIPTextEmbeddings
        self.encoder = self.text.encoder  # CLIPEncoder
        self.final_layer_norm = self.text.final_layer_norm
        self.text_projection = hf_model.text_projection

        self.dtype = next(hf_model.parameters()).dtype

        cfg = hf_model.config.text_config
        self.eos_token_id = cfg.eos_token_id
        self.pad_token_id = cfg.pad_token_id if cfg.pad_token_id is not None else 0

    def _build_additive_padding_mask(self, attn2d, device):
        # attn2d: (B,L) 1=keep 0=pad
        # HF attend souvent (B,1,1,L) additive mask en float
        mask = (1.0 - attn2d.float()) * torch.finfo(torch.float32).min
        return mask.to(device=device, dtype=torch.float32)[
            :, None, None, :
        ]  # (B,1,1,L)

    def _build_causal_mask(self, L, B, device):
        # causal additive mask (B,1,L,L) avec -inf au-dessus de la diagonale
        m = torch.full(
            (L, L), torch.finfo(torch.float32).min, device=device, dtype=torch.float32
        )
        m = torch.triu(m, diagonal=1)  # upper triangle = -inf
        return m[None, None, :, :].expand(B, 1, L, L)  # (B,1,L,L)

    def _unwrap_last_hidden(enc_out):
        # 1) HF output object
        if hasattr(enc_out, "last_hidden_state") and enc_out.last_hidden_state is not None:
            return enc_out.last_hidden_state
        # 2) some HF outputs use .hidden_states or .logits etc (rare here)
        if hasattr(enc_out, "hidden_states") and enc_out.hidden_states is not None:
            # hidden_states is a tuple (layer0,...,last)
            return enc_out.hidden_states[-1]
        # 3) tuple/list
        if isinstance(enc_out, (tuple, list)):
            return enc_out[0]
        # 4) already a tensor
        if torch.is_tensor(enc_out):
            return enc_out
        raise TypeError(f"Cannot unwrap encoder output of type {type(enc_out)}")

    def forward(self, prompts, tokenized_prompts, deep_prompts=None):
        """
        prompts: (B,L,D) embeddings construits par ton prompt learner
        tokenized_prompts: (B,L) ids pour pad/eos positions
        deep_prompts: pas utilisé ici (tu peux l'ajouter plus tard si tu veux)
        """
        device = tokenized_prompts.device
        x = prompts.to(device=device, dtype=self.dtype)  # (B,L,D)

        B, L, _ = x.shape

        # attention mask 2D
        attn2d = (tokenized_prompts != self.pad_token_id).to(
            device=device, dtype=torch.long
        )

        # HF CLIP ajoute position embeddings via CLIPTextEmbeddings
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos = self.embeddings.position_embedding(pos_ids).to(dtype=self.dtype)
        x = x + pos

        # masks 4D
        pad4d = self._build_additive_padding_mask(attn2d, device)  # (B,1,1,L)
        causal4d = self._build_causal_mask(L, B, device)  # (B,1,L,L)

        enc_kwargs = dict(inputs_embeds=x, attention_mask=pad4d, causal_attention_mask=causal4d)

        # Certaines versions n'aiment pas causal_attention_mask -> fallback
        try:
            enc_out = self.encoder(**enc_kwargs)
        except TypeError:
            enc_kwargs.pop("causal_attention_mask", None)
            enc_out = self.encoder(**enc_kwargs)

        # unwrap -> Tensor
        if hasattr(enc_out, "last_hidden_state") and enc_out.last_hidden_state is not None:
            last_hidden = enc_out.last_hidden_state
        elif isinstance(enc_out, (tuple, list)):
            last_hidden = enc_out[0]
        elif torch.is_tensor(enc_out):
            last_hidden = enc_out
        else:
            raise TypeError(f"Unexpected encoder output type: {type(enc_out)}")

        # LN
        last_hidden = self.final_layer_norm(last_hidden)

        # EOS pooling (CLIP-like)
        if self.eos_token_id is not None:
            eos_pos = (tokenized_prompts == self.eos_token_id).int().argmax(dim=-1)
        else:
            # fallback: last non-pad
            lengths = attn2d.sum(dim=-1)
            eos_pos = (lengths - 1).clamp(min=0)

        pooled = last_hidden[torch.arange(B, device=device), eos_pos]  # (B,D)
        txt_feat = self.text_projection(pooled)  # (B,proj_dim)
        return txt_feat


class VisionEncoderHF(nn.Module):
    def __init__(self, hf_model: CLIPModel):
        super().__init__()
        self.model = hf_model
        self.vision_model = hf_model.vision_model
        self.visual_projection = hf_model.visual_projection
        self.dtype = next(hf_model.parameters()).dtype

    def forward(self, pixel_values, shared_ctx=None, deep_prompts_vision=None, target_dim=None):
        vm = self.vision_model
        device = pixel_values.device
        x = vm.embeddings(pixel_values.to(dtype=self.dtype))  # (B, 1+P, D)
        B, L, D = x.shape

        def _hf_additive_padding_mask(attention_mask_2d, dtype, device):
            # attention_mask_2d: (B,L) 1=keep 0=pad
            # HF CLIP expects additive mask shape (B, 1, 1, L) in float
            additive = (1.0 - attention_mask_2d.float()) * torch.finfo(
                torch.float32
            ).min
            return additive.to(device=device, dtype=torch.float32)[:, None, None, :]

        attn2d = torch.ones(
            (B, L), device=device, dtype=torch.long
        )  # pas de padding en vision
        pad4d = _hf_additive_padding_mask(attn2d, self.dtype, device)
        causal4d = None

        # --- inject shallow ctx after CLS
        if shared_ctx is not None:
            if shared_ctx.dim() == 2:  # (n_ctx, Dv)
                ctx = shared_ctx.unsqueeze(0).expand(B, -1, -1)
            else:  # (B, n_ctx, Dv)
                ctx = shared_ctx
            ctx = ctx.to(device=device, dtype=self.dtype)
            x = torch.cat([x[:, :1], ctx, x[:, 1:]], dim=1)

        print("[HF VISION] injected shallow ctx, new seq len =", x.size(1))
        # --- forward encoder layers with deep prompts injection
        layers = vm.encoder.layers
        counter = 0

        def make_masks(cur_L):
            attn2d = torch.ones((B, cur_L), device=device, dtype=torch.long)
            pad4d = (1.0 - attn2d.float()) * torch.finfo(torch.float32).min
            pad4d = pad4d[:, None, None, :]  # (B,1,1,L)
            causal4d = None  # vision non-causal
            return pad4d, causal4d

        for layer in layers:
            if deep_prompts_vision is not None and counter < len(deep_prompts_vision):
                p = deep_prompts_vision[counter].to(
                    device=device, dtype=self.dtype
                )  # (n_ctx, D)
                p = p.unsqueeze(0).expand(B, -1, -1)  # (B, n_ctx, D)

                x = torch.cat([x[:, :1], p, x[:, 1:]], dim=1)

                pad4d, causal4d = make_masks(x.size(1))
                out = layer(
                    x,
                    attention_mask=pad4d,
                    causal_attention_mask=causal4d,
                )
                x = out[0] if isinstance(out, (tuple, list)) else out

                n_ctx = p.size(1)
                x = torch.cat([x[:, :1], x[:, 1 + n_ctx :]], dim=1)

                counter += 1
            else:
                pad4d, causal4d = make_masks(x.size(1))
                try:
                    out = layer(x, attention_mask=pad4d, causal_attention_mask=causal4d)
                except TypeError:
                    out = layer(x, attention_mask=pad4d)
                x = out[0] if isinstance(out, (tuple, list)) else out

        # --- pooled CLS + post norm (HF CLIP convention)
        pooled = x[:, 0]
        if hasattr(vm, "post_layernorm") and vm.post_layernorm is not None:
            pooled = vm.post_layernorm(pooled)

        img_feat = self.visual_projection(pooled)
        return img_feat


class VisionEncoderOpenCLIPTimm(nn.Module):
    def __init__(self, openclip_model):
        super().__init__()
        self.model = openclip_model
        self.visual = openclip_model.visual
        assert hasattr(self.visual, "trunk")
        self.trunk = self.visual.trunk
        self.dtype = next(openclip_model.parameters()).dtype

    def _apply_proj_to_target_dim(self, x, target_dim: int):
        """Try hard to map x: (B,in_dim) -> (B,target_dim) using any projection found in the model."""
        if target_dim is None or x.shape[-1] == target_dim:
            return x

        in_dim = x.shape[-1]

        # 1) obvious candidates
        candidates = []
        for obj in [self.visual, self.model, getattr(self.model, "visual", None)]:
            if obj is None:
                continue
            for attr in ["proj", "projection", "visual_projection", "head"]:
                if hasattr(obj, attr):
                    candidates.append(getattr(obj, attr))

        for proj in candidates:
            if proj is None:
                continue
            if (
                torch.is_tensor(proj)
                and proj.ndim == 2
                and proj.shape == (in_dim, target_dim)
            ):
                return x @ proj
            if (
                isinstance(proj, nn.Linear)
                and proj.in_features == in_dim
                and proj.out_features == target_dim
            ):
                return proj(x)
            if isinstance(proj, nn.Module):
                try:
                    y = proj(x)
                    if y.shape[-1] == target_dim:
                        return y
                except Exception:
                    pass

        # 2) scan modules: Linear(in_dim -> target_dim)
        for _, mod in self.model.named_modules():
            if (
                isinstance(mod, nn.Linear)
                and mod.in_features == in_dim
                and mod.out_features == target_dim
            ):
                return mod(x)

        # 3) scan params: matrix (in_dim, target_dim)
        for _, p in self.model.named_parameters():
            if p.ndim == 2 and p.shape == (in_dim, target_dim):
                return x @ p

        # nothing found -> keep as-is (will crash later, but better than silent wrong)
        return x

    def forward(
        self, image, shared_ctx=None, deep_prompts_vision=None, target_dim=None
    ):
        trunk = self.trunk
        B = image.size(0)
        device = image.device

        x = trunk.patch_embed(image.to(dtype=self.dtype))
        if x.dim() == 4:
            embed_dim = (
                getattr(trunk, "embed_dim", None)
                or getattr(trunk, "num_features", None)
                or trunk.pos_embed.shape[-1]
            )
            if x.shape[1] == embed_dim:
                x = x.flatten(2).transpose(1, 2)
            elif x.shape[-1] == embed_dim:
                x = x.contiguous().view(B, -1, embed_dim)
            elif x.shape[2] == embed_dim:
                x = x.permute(0, 1, 3, 2).contiguous().view(B, -1, embed_dim)
            else:
                raise RuntimeError(
                    f"patch_embed returned 4D unexpected: {tuple(x.shape)} embed_dim={embed_dim}"
                )

        cls = trunk.cls_token.expand(B, -1, -1).to(device=device, dtype=self.dtype)

        # shallow ctx
        ctx = None
        if shared_ctx is not None:
            ctx = (
                shared_ctx.unsqueeze(0).expand(B, -1, -1)
                if shared_ctx.dim() == 2
                else shared_ctx
            )
            ctx = ctx.to(device=device, dtype=self.dtype)

        # pos embed (only for CLS+patches)
        if hasattr(trunk, "pos_embed") and trunk.pos_embed is not None:
            pos = trunk.pos_embed.to(
                device=device, dtype=self.dtype
            )  # (1, 1+n_patches, D)
            cls = cls + pos[:, :1, :]
            patch_pos = pos[:, 1:, :][:, : x.shape[1], :]
            x = x + patch_pos

        # concat
        x = (
            torch.cat([cls, ctx, x], dim=1)
            if ctx is not None
            else torch.cat([cls, x], dim=1)
        )

        if hasattr(trunk, "pos_drop") and trunk.pos_drop is not None:
            x = trunk.pos_drop(x)
        if hasattr(trunk, "norm_pre") and trunk.norm_pre is not None:
            x = trunk.norm_pre(x)

        # blocks + deep prompts
        counter = 0
        for blk in trunk.blocks:
            if deep_prompts_vision is not None and counter < len(deep_prompts_vision):
                p = deep_prompts_vision[counter].to(
                    device=device, dtype=self.dtype
                )  # (n_ctx, D)
                p = p.unsqueeze(0).expand(B, -1, -1)
                x = torch.cat([x[:, :1], p, x[:, 1:]], dim=1)
                x = blk(x)
                x = torch.cat([x[:, :1], x[:, 1 + p.size(1) :]], dim=1)
                counter += 1
            else:
                x = blk(x)

        if hasattr(trunk, "norm") and trunk.norm is not None:
            x = trunk.norm(x)

        x = x[:, 0]  # CLS
        if hasattr(trunk, "fc_norm") and trunk.fc_norm is not None:
            x = trunk.fc_norm(x)

        # ✅ CRUCIAL: force projection to match text space (e.g., 512)
        x = self._apply_proj_to_target_dim(x, target_dim)
        return x


class VisionEncoderOpenCLIPViT(nn.Module):
    def __init__(self, openclip_model):
        super().__init__()
        self.model = openclip_model
        self.visual = openclip_model.visual
        self.dtype = next(openclip_model.parameters()).dtype

        self.visual_proj = getattr(self.visual, "proj", None)

    def _apply_proj_if_any(self, x, target_dim=None):
        """
        Essaie d'appliquer une projection pour matcher target_dim (ex: 512).
        Cherche:
        - attributs classiques
        - puis une projection cachée (Parameter ou Linear) de shape (in_dim, target_dim)
        """
        in_dim = x.shape[-1]
        if target_dim is None or in_dim == target_dim:
            return x

        # 1) noms classiques (wrapper + modèle)
        candidates = [
            (self.visual, "proj"),
            (self.visual, "projection"),
            (self.visual, "visual_projection"),
            (self.model, "visual_projection"),
            (self.model, "proj"),
            (getattr(self.model, "visual", None), "proj"),
            (getattr(self.model, "visual", None), "projection"),
        ]

        for obj, attr in candidates:
            if obj is None or not hasattr(obj, attr):
                continue
            proj = getattr(obj, attr)
            if proj is None:
                continue

            if (
                torch.is_tensor(proj)
                and proj.ndim == 2
                and proj.shape[0] == in_dim
                and proj.shape[1] == target_dim
            ):
                return x @ proj
            if (
                isinstance(proj, nn.Linear)
                and proj.in_features == in_dim
                and proj.out_features == target_dim
            ):
                return proj(x)
            if isinstance(proj, nn.Module):
                # parfois c'est un Sequential etc.
                try:
                    y = proj(x)
                    if y.shape[-1] == target_dim:
                        return y
                except Exception:
                    pass

        # 2) recherche "cachée" dans les sous-modules
        #    - nn.Linear(in_dim -> target_dim)
        for name, mod in self.model.named_modules():
            if (
                isinstance(mod, nn.Linear)
                and mod.in_features == in_dim
                and mod.out_features == target_dim
            ):
                # print(f"[FOUND PROJ Linear] {name}")
                return mod(x)

        #    - Parameter/Tensor de shape (in_dim, target_dim)
        for name, p in self.model.named_parameters():
            if p.ndim == 2 and p.shape[0] == in_dim and p.shape[1] == target_dim:
                # print(f"[FOUND PROJ Param] {name}")
                return x @ p

        # rien trouvé
        return x

    def _apply_proj_param_scan(self, x, target_dim):
        in_dim = x.shape[-1]
        # Linear in modules
        for name, mod in self.model.named_modules():
            if (
                isinstance(mod, nn.Linear)
                and mod.in_features == in_dim
                and mod.out_features == target_dim
            ):
                return mod(x)
        # Param matrix
        for name, p in self.model.named_parameters():
            if p.ndim == 2 and p.shape[0] == in_dim and p.shape[1] == target_dim:
                return x @ p
        return x

    def forward(
        self, image, shared_ctx=None, deep_prompts_vision=None, target_dim=None
    ):
        v = self.visual
        B = image.size(0)
        device = image.device
        dtype = self.dtype

        # -------------------------
        # 1) patchify: conv1 ou patch_embed
        # open_clip ViT : conv1 est souvent la conv patchify
        if hasattr(v, "conv1") and v.conv1 is not None:
            x = v.conv1(image.to(dtype=dtype))  # (B, width, grid, grid)
            x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)  # (B, n_patches, width)
        elif hasattr(v, "patch_embed"):
            x = v.patch_embed(image.to(dtype=dtype))  # souvent (B, n_patches, width)
        else:
            raise AttributeError(f"Unknown patch embedding for visual={type(v)}")

        # -------------------------
        # 2) CLS token
        if hasattr(v, "class_embedding") and v.class_embedding is not None:
            cls = v.class_embedding.to(device=device, dtype=dtype)
            cls = cls.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B,1,width)
        elif hasattr(v, "cls_token") and v.cls_token is not None:
            cls = v.cls_token.expand(B, -1, -1).to(device=device, dtype=dtype)
        else:
            raise AttributeError(
                "Could not find class token (class_embedding/cls_token)"
            )

        # -------------------------
        # 3) shallow ctx (VPT-style) : insert AFTER CLS
        ctx = None
        if shared_ctx is not None:
            if shared_ctx.dim() == 2:
                ctx = shared_ctx.unsqueeze(0).expand(B, -1, -1)
            else:
                ctx = shared_ctx
            ctx = ctx.to(device=device, dtype=dtype)

        # -------------------------
        # 4) positional embedding
        # open_clip : positional_embedding est souvent (1+n_patches, width) ou (1,1+n_patches,width)
        pos = getattr(v, "positional_embedding", None)
        if pos is not None:
            pos = pos.to(device=device, dtype=dtype)
            if pos.dim() == 2:  # (1+n_patches, width)
                pos = pos.unsqueeze(0)  # (1,1+n_patches,width)
            # pos couvre [CLS + patches], pas les ctx
            cls = cls + pos[:, :1, :]
            x = x + pos[:, 1 : 1 + x.size(1), :]

        # -------------------------
        # 5) concat tokens: [CLS, (CTX), PATCHES]
        if ctx is not None:
            x = torch.cat([cls, ctx, x], dim=1)
        else:
            x = torch.cat([cls, x], dim=1)

        # -------------------------
        # 6) ln_pre
        if hasattr(v, "ln_pre") and v.ln_pre is not None:
            x = v.ln_pre(x)

        # -------------------------
        # 7) transformer expects (L,B,D)
        x = x.permute(1, 0, 2)

        # deep prompts layer-wise (insert after CLS)
        blocks = _get_resblocks(getattr(v, "transformer", v))
        counter = 0
        for blk in blocks:
            if deep_prompts_vision is not None and counter < len(deep_prompts_vision):
                p = deep_prompts_vision[counter].to(
                    device=device, dtype=dtype
                )  # (n_ctx, D)
                p = p.unsqueeze(1).expand(-1, x.size(1), -1)  # (n_ctx,B,D)

                x = torch.cat([x[:1], p, x[1:]], dim=0)
                x = blk(x)
                x = torch.cat([x[:1], x[1 + p.size(0) :]], dim=0)

                counter += 1
            else:
                x = blk(x)

        x = x.permute(1, 0, 2)  # (B,L,D)

        # -------------------------
        # 8) ln_post + take CLS
        if hasattr(v, "ln_post") and v.ln_post is not None:
            x = v.ln_post(x)

        # CLS token
        x = x[:, 0]

        if hasattr(v, "fc_norm") and v.fc_norm is not None:
            x = v.fc_norm(x)

        # ✅ si on veut matcher l'espace texte
        if target_dim is not None and x.shape[-1] != target_dim:
            # essaie proj du wrapper / modèle
            x2 = self._apply_proj_if_any(x)
            if x2.shape[-1] == target_dim:
                x = x2
            else:
                # fallback: cherche une proj compatible dans le modèle (comme tu fais déjà dans ViT)
                x = self._apply_proj_param_scan(
                    x, target_dim
                )  # à ajouter (cf ci-dessous)
        else:
            # sinon garde ton comportement historique
            x = self._apply_proj_if_any(x)

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = getattr(clip_model, "dtype", None)
        if dtype is None:
            dtype = next(clip_model.parameters()).dtype

        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert (
            cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1
        ), "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = (
            cfg.TRAINER.MAPLE.PROMPT_DEPTH
        )  # max=12, but will create 11 such shared prompts
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print("MaPLe design: Multi-modal Prompt Learning")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        vis_width = _infer_visual_width(clip_model.visual)

        self.proj = nn.Linear(ctx_dim, vis_width)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, 512))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, vis_width)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]
        )  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        return (
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )


class MultiModalPromptLearnerQuilt(nn.Module):
    def __init__(self, cfg, classnames, openclip_model, tokenizer):
        super().__init__()

        def _tok_to_input_ids(tok_out, tokenizer, texts, device):
            """
            Normalize tokenizer output to torch.LongTensor (B, L) on device.
            Handles:
            - open_clip tokenizer: returns Tensor
            - dict / BatchEncoding: contains 'input_ids'
            - list/np: numeric ids
            - BAD CASE: strings -> fallback using tokenizer.encode / tokenizer.tokenize if available
            """
            # transformers BatchEncoding
            if hasattr(tok_out, "data") and isinstance(tok_out.data, dict):
                tok_out = tok_out.data

            # dict-like
            if isinstance(tok_out, dict):
                if "input_ids" in tok_out:
                    tok_out = tok_out["input_ids"]
                else:
                    tok_out = tok_out[next(iter(tok_out.keys()))]

            # tensor
            if torch.is_tensor(tok_out):
                return tok_out.to(device=device, dtype=torch.long)

            # ✅ BAD CASE: strings / list of strings
            if isinstance(tok_out, str) or (
                isinstance(tok_out, (list, tuple))
                and len(tok_out) > 0
                and isinstance(tok_out[0], str)
            ):
                # Try common tokenizer APIs
                if hasattr(tokenizer, "encode"):
                    # encode each text -> list[int]
                    ids = [tokenizer.encode(t) for t in texts]
                    return torch.as_tensor(ids, dtype=torch.long, device=device)
                if hasattr(tokenizer, "tokenize") and hasattr(
                    tokenizer, "convert_tokens_to_ids"
                ):
                    ids = [
                        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t))
                        for t in texts
                    ]
                    return torch.as_tensor(ids, dtype=torch.long, device=device)

                raise TypeError(
                    f"Tokenizer returned strings (type={type(tok_out)}) but no encode/tokenize API available. "
                    f"Tokenizer={type(tokenizer)}"
                )

            # list / numpy / other numeric -> tensor
            return torch.as_tensor(tok_out, dtype=torch.long, device=device)

        def _check_vocab(ids, te, tag):
            vocab = te.weight.shape[0]
            mx = int(ids.max().item())
            if mx >= vocab:
                raise RuntimeError(
                    f"[{tag}] tokenizer produced id {mx} but vocab_size={vocab}. "
                    f"Tokenizer/model mismatch. ids shape={tuple(ids.shape)}"
                )

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT

        dtype = next(openclip_model.parameters()).dtype
        device = next(openclip_model.parameters()).device

        te = _get_openclip_token_embedding(openclip_model)
        ctx_dim = te.weight.shape[1]

        # ---- init ctx (shallow)
        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            tok_ids = _tok_to_input_ids(
                tokenizer([ctx_init]), tokenizer, [ctx_init], device
            )
            _check_vocab(tok_ids, te, "ctx_init")
            embedding = te(tok_ids).to(dtype)

            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # ---- tokenized prompts
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = _tok_to_input_ids(
            tokenizer(prompts), tokenizer, prompts, device
        )

        _check_vocab(tokenized_prompts, te, "prompts")

        with torch.no_grad():
            embedding = te(tokenized_prompts).to(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx :, :], persistent=False
        )

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.register_buffer("tokenized_prompts", tokenized_prompts, persistent=False)

        # -----------------------------
        # ✅ Trainable parameters (like MaPLe)
        # -----------------------------
        # vision width (token dim côté ViT, pas forcément l'embed CLIP final)
        vis_width = _infer_visual_width(openclip_model.visual)

        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH

        # shallow ctx (trainable)
        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, ctx_dim)

        # project shallow ctx -> vision token dim
        self.proj = nn.Linear(ctx_dim, vis_width).to(device=device)
        if dtype == torch.float16:
            self.proj.half()

        # deep prompts (text side) in ctx_dim
        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, ctx_dim, device=device, dtype=dtype))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)

        # projections deep prompts -> vision dim
        single_layer = nn.Linear(ctx_dim, vis_width).to(device=device)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )
        if dtype == torch.float16:
            for li in self.compound_prompt_projections:
                li.half()

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        visual_deep_prompts = []
        for i, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[i]))

        return (
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )


class MultiModalPromptLearnerPubMedCLIP(nn.Module):
    def __init__(
        self, cfg, classnames, hf_model: CLIPModel, hf_tokenizer: CLIPTokenizerFast
    ):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT

        dtype = next(hf_model.parameters()).dtype
        device = next(hf_model.parameters()).device

        ctx_dim = hf_model.text_model.config.hidden_size
        vis_width = hf_model.vision_model.config.hidden_size

        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH

        # init ctx
        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            tok = hf_tokenizer(
                [ctx_init],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            with torch.no_grad():
                emb = hf_model.text_model.embeddings.token_embedding(
                    tok["input_ids"]
                ).to(dtype)
            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print("MaPLe design: Multi-modal Prompt Learning (HF)")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.proj = nn.Linear(ctx_dim, vis_width).to(device=device)
        if dtype == torch.float16:
            self.proj.half()

        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, ctx_dim, device=device, dtype=dtype))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)

        single_layer = nn.Linear(ctx_dim, vis_width).to(device=device)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )
        if dtype == torch.float16:
            for li in self.compound_prompt_projections:
                li.half()

        # tokenized prompts
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tok = hf_tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        tokenized_prompts = tok["input_ids"]

        with torch.no_grad():
            embedding = hf_model.text_model.embeddings.token_embedding(
                tokenized_prompts
            ).to(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx :, :], persistent=False
        )

        self.n_cls = n_cls
        self.n_ctx = n_ctx

        self.register_buffer("tokenized_prompts", tokenized_prompts, persistent=False)

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        visual_deep_prompts = []
        for i, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[i]))

        return (
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )


class MultiModalPromptLearnerBiomed(nn.Module):
    def __init__(self, cfg, classnames, biomed_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT

        dtype = next(biomed_model.parameters()).dtype
        device = next(biomed_model.parameters()).device

        text = biomed_model.text
        vis_width = _infer_visual_width(biomed_model.visual)

        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH

        we = text.transformer.embeddings.word_embeddings
        ctx_dim = we.weight.shape[1]

        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            tok = tokenizer([ctx_init])
            tok = _to_device_tokenized(tok, device)
            if isinstance(tok, dict):
                tok_ids = tok["input_ids"]
            else:
                tok_ids = tok
            with torch.no_grad():
                emb = we(tok_ids).to(dtype)
            ctx_vectors = emb[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print("MaPLe design: Multi-modal Prompt Learning (BiomedCLIP)")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.proj = nn.Linear(ctx_dim, vis_width).to(device=device)
        if dtype == torch.float16:
            self.proj.half()

        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, ctx_dim, device=device, dtype=dtype))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)

        single_layer = nn.Linear(ctx_dim, vis_width).to(device=device)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )
        if dtype == torch.float16:
            for li in self.compound_prompt_projections:
                li.half()

        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tok = tokenizer(prompts)
        tok = _to_device_tokenized(tok, device)
        if isinstance(tok, dict):
            tokenized_prompts = tok["input_ids"]
        else:
            tokenized_prompts = tok

        with torch.no_grad():
            embedding = we(tokenized_prompts).to(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx :, :], persistent=False
        )

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.register_buffer("tokenized_prompts", tokenized_prompts, persistent=False)

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        visual_deep_prompts = []
        for i, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[i]))

        return (
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )


def build_openclip_image_encoder(openclip_model):
    visual = openclip_model.visual
    # wrapper timm
    if hasattr(visual, "trunk"):
        return VisionEncoderOpenCLIPTimm(openclip_model)
    # ViT direct open_clip
    return VisionEncoderOpenCLIPViT(openclip_model)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tok_openclip=None, tok_hf=None):
        super().__init__()
        self.cfg = cfg
        backbone = cfg.MODEL.BACKBONE.NAME

        # -------- Prompt learner
        if backbone in ["Quilt-B/32", "Quilt-B/16", "Conch"]:
            assert tok_openclip is not None
            self.prompt_learner = MultiModalPromptLearnerQuilt(
                cfg, classnames, clip_model, tok_openclip
            )

            self.text_encoder = TextEncoderOpenCLIP(clip_model)
            self.image_encoder = build_openclip_image_encoder(clip_model)

            self.dtype = next(clip_model.parameters()).dtype

        elif backbone == "BiomedCLIP":
            assert tok_openclip is not None
            self.prompt_learner = MultiModalPromptLearnerBiomed(
                cfg, classnames, clip_model, tok_openclip
            )

            self.text_encoder = TextEncoderBiomed(clip_model)
            self.image_encoder = build_openclip_image_encoder(clip_model)

            self.dtype = next(clip_model.parameters()).dtype

        elif backbone in ["PubMedCLIP-B/32", "PLIP-B/32"]:
            assert tok_hf is not None

            self.prompt_learner = MultiModalPromptLearnerPubMedCLIP(
                cfg, classnames, clip_model, tok_hf
            )

            # IMPORTANT: il te faut aussi un text encoder HF + vision encoder HF
            self.text_encoder = TextEncoderHF(clip_model)
            self.image_encoder = VisionEncoderHF(clip_model)

            self.dtype = next(clip_model.parameters()).dtype

        else:
            # OpenAI CLIP (ViT only) - MaPLe original
            self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)

            self.text_encoder = TextEncoder(clip_model)
            self.image_encoder = clip_model.visual

            self.dtype = (
                getattr(clip_model, "dtype", None)
                or next(clip_model.parameters()).dtype
            )

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.logit_scale = clip_model.logit_scale

    def forward(self, image, label=None):
        device = image.device

        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)

        prompts, shared_ctx, deep_text, deep_vision = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts, deep_text)
        target_dim = text_features.shape[-1]

        img_in = image.to(dtype=self.dtype)

        # image encoder (tes wrappers)
        if isinstance(self.image_encoder, (VisionEncoderOpenCLIPTimm, VisionEncoderOpenCLIPViT, VisionEncoderHF)):
            image_features = self.image_encoder(img_in, shared_ctx, deep_vision, target_dim=target_dim)
        else:
            # OpenAI CLIP visual
            try:
                image_features = self.image_encoder(img_in, shared_ctx, deep_vision)
            except TypeError:
                image_features = self.image_encoder(img_in)

        # ✅ normalisation robuste en fp32 + eps
        image_features = F.normalize(image_features.float(), dim=-1, eps=1e-6)
        text_features = F.normalize(text_features.float(), dim=-1, eps=1e-6)

        # ✅ logit_scale safe
        logit_scale = safe_logit_scale_exp(self.logit_scale, device=device, dtype=torch.float32)

        # ✅ matmul fp32
        logits = logit_scale * (image_features @ text_features.t())

        print("shared_ctx:", None if shared_ctx is None else shared_ctx.shape)
        print("deep_text:", None if deep_text is None else len(deep_text))
        print("deep_vision:", None if deep_vision is None else len(deep_vision))

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        tok_openclip = None
        tok_hf = None

        if cfg.MODEL.BACKBONE.NAME in [
            "Quilt-B/32",
            "Quilt-B/16",
            "BiomedCLIP",
            "Conch",
        ]:
            # open_clip tokenizer (get_tokenizer)
            model_id = {
                "Quilt-B/32": "hf-hub:wisdomik/QuiltNet-B-32",
                "Quilt-B/16": "hf-hub:wisdomik/QuiltNet-B-16",
                "BiomedCLIP": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                "Conch": "Conch",
            }[cfg.MODEL.BACKBONE.NAME]

            def get_openclip_tokenizer_for_conch(clip_model):
                """
                Retourne un callable f(texts: List[str]) -> Tensor[int64] (B,L)
                compatible avec le reste du code.
                """
                # 1) si conch expose un get_tokenizer() utilisable directement
                if hasattr(conch.open_clip_custom, "get_tokenizer"):
                    try:
                        tok = conch.open_clip_custom.get_tokenizer()
                        # tok est souvent callable(texts)->ids
                        return tok
                    except TypeError:
                        pass

                # 2) si conch expose tokenize(...) mais avec signature spéciale,
                # on essaie plusieurs appels possibles.
                if hasattr(conch.open_clip_custom, "tokenize"):
                    fn = conch.open_clip_custom.tokenize

                    def _wrapped(texts):
                        # essais signature: tokenize(texts) / tokenize(model, texts) / tokenize(tokenizer, texts)
                        try:
                            return fn(texts)
                        except TypeError:
                            pass
                        try:
                            return fn(clip_model, texts)
                        except TypeError:
                            pass
                        # certains projets ont tokenize(tokenizer, texts) où tokenizer est dans le modèle
                        tok_obj = getattr(clip_model, "tokenizer", None)
                        if tok_obj is not None:
                            try:
                                return fn(tok_obj, texts)
                            except TypeError:
                                pass

                        raise TypeError(
                            f"Conch tokenize signature not supported. "
                            f"tokenize={fn} model_type={type(clip_model)}"
                        )

                    return _wrapped

                # 3) fallback open_clip
                return get_tokenizer("ViT-B-16")

            if cfg.MODEL.BACKBONE.NAME == "Conch":
                tok_openclip = get_openclip_tokenizer_for_conch(clip_model)
            else:
                tok_openclip = get_tokenizer(model_id)

        if cfg.MODEL.BACKBONE.NAME in ["PubMedCLIP-B/32", "PLIP-B/32"]:
            model_id = (
                "flaviagiammarino/pubmed-clip-vit-base-patch32"
                if cfg.MODEL.BACKBONE.NAME == "PubMedCLIP-B/32"
                else "vinid/plip"
            )
            tok_hf = CLIPTokenizerFast.from_pretrained(model_id)

        self.model = CustomCLIP(
            cfg, classnames, clip_model, tok_openclip=tok_openclip, tok_hf=tok_hf
        )

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        batch = next(iter(self.train_loader_x))
        sanity_check_maple_unified(self, batch, print_trainables=True, check_backward=True)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "MultiModalPromptLearner", self.model, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
