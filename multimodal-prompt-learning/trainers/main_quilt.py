import torch
import torch.nn as nn
# from torch.nn import functional as F
import open_clip


# -------------------------
# Helpers
# -------------------------
@torch.no_grad()
def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def tinfo(x: torch.Tensor, name: str):
    print(
        f"[T] {name:18s} shape={tuple(x.shape)} dtype={x.dtype} device={x.device} finite={torch.isfinite(x).all().item()}"
    )


def diff01(x, name):
    if x.size(0) >= 2:
        print(f"[D] {name:18s} max|x0-x1| = {(x[0]-x[1]).abs().max().item()}")


def describe_openclip_quilt(model):
    print("\n" + "=" * 100)
    print("[MODEL] type =", type(model))
    # open_clip models usually have .visual, .text, .logit_scale
    print("[ATTR] has visual:", hasattr(model, "visual"))
    print("[ATTR] has text  :", hasattr(model, "text"))
    print("[ATTR] has encode_image:", hasattr(model, "encode_image"))
    print("[ATTR] has encode_text :", hasattr(model, "encode_text"))
    print("[ATTR] has logit_scale :", hasattr(model, "logit_scale"))

    vis = model.visual
    print("\n[VISUAL] type =", type(vis))
    for k in [
        "conv1",
        "class_embedding",
        "positional_embedding",
        "ln_pre",
        "ln_post",
        "proj",
        "transformer",
    ]:
        print(f"[VISUAL] has {k:16s}:", hasattr(vis, k))

    if hasattr(vis, "conv1"):
        print("[VISUAL] conv1:", vis.conv1)
    if hasattr(vis, "positional_embedding") and torch.is_tensor(
        vis.positional_embedding
    ):
        print(
            "[VISUAL] positional_embedding:",
            tuple(vis.positional_embedding.shape),
            vis.positional_embedding.dtype,
        )
    if hasattr(vis, "transformer"):
        tr = vis.transformer
        print("[VISUAL] transformer type:", type(tr))
        if hasattr(tr, "resblocks"):
            print("[VISUAL] #resblocks:", len(tr.resblocks))
            if len(tr.resblocks) > 0:
                print("[VISUAL] block0 type:", type(tr.resblocks[0]))

    txt = getattr(model, "text", None)
    print("\n[TEXT] type =", type(txt))
    if txt is not None:
        for k in [
            "token_embedding",
            "positional_embedding",
            "transformer",
            "ln_final",
            "text_projection",
            "attn_mask",
        ]:
            print(f"[TEXT] has {k:18s}:", hasattr(txt, k))
        if hasattr(txt, "token_embedding"):
            print("[TEXT] token_embedding:", txt.token_embedding)
        if hasattr(txt, "positional_embedding") and torch.is_tensor(
            txt.positional_embedding
        ):
            print(
                "[TEXT] positional_embedding:",
                tuple(txt.positional_embedding.shape),
                txt.positional_embedding.dtype,
            )
        if hasattr(txt, "transformer"):
            tr = txt.transformer
            print("[TEXT] transformer type:", type(tr))
            if hasattr(tr, "resblocks"):
                print("[TEXT] #resblocks:", len(tr.resblocks))
    print("=" * 100 + "\n")


# -------------------------
# VPT wrapper (your style) — with switch expand/repeat
# -------------------------
class OpenClipVisionVPT(nn.Module):
    def __init__(self, visual_vit: nn.Module, n_ctx: int):
        super().__init__()
        self.base = visual_vit
        self.n_ctx = int(n_ctx)
        self.use = self.n_ctx > 0

        width = self.base.conv1.out_channels
        p0 = torch.empty(self.n_ctx, width)
        nn.init.normal_(p0, std=0.02)
        self.VPT0 = nn.Parameter(p0)

    def _append_vpt(self, x):  # (B,L,D)
        vpt = (
            self.VPT0.to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1, -1)
        )
        return torch.cat([x, vpt], dim=1)  # (B, L+n_ctx, D)

    def forward(self, x):
        # 1) conv -> patch tokens (B,N,D)
        x = self.base.conv1(x)  # (B,D,gh,gw)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # 2) CLS + pos (open_clip ViT uses pos (L,D))
        cls = (
            self.base.class_embedding.to(x.dtype)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(x.size(0), 1, -1)
        )
        x = torch.cat([cls, x], dim=1)  # (B,197,D)

        pos = self.base.positional_embedding.to(x.dtype)
        if pos.dim() == 2:
            x = x + pos[: x.size(1), :].unsqueeze(0)
        else:
            x = x + pos[:, : x.size(1), :]

        # 3) append VPT tokens (à la fin)
        if self.use:
            x = self._append_vpt(x)

        # 4) patch_dropout si présent (Quilt l’a)
        if hasattr(self.base, "patch_dropout") and self.base.patch_dropout is not None:
            x = self.base.patch_dropout(x)

        # 5) ln_pre
        if getattr(self.base, "ln_pre", None) is not None:
            x = self.base.ln_pre(x)

        # ✅ 6) transformer en (B,L,D) (PAS de permute)
        x = self.base.transformer(x)

        # 7) pool CLS
        feat = x[:, 0, :]
        if getattr(self.base, "ln_post", None) is not None:
            feat = self.base.ln_post(feat)

        # 8) proj
        proj = getattr(self.base, "proj", None)
        if proj is not None:
            feat = feat @ proj

        return feat


class QuiltTextEncoder(nn.Module):
    def __init__(self, quilt_model, n_ctx: int):
        super().__init__()
        self.model = quilt_model
        self.n_ctx = int(n_ctx)

        # open_clip: parfois tout est au top-level (pas de model.text)
        root = quilt_model
        text = getattr(quilt_model, "text", None)
        self.root = text if text is not None else root  # <- fallback IMPORTANT

        # token embedding: parfois token_embedding est au top-level
        self.token_embedding = getattr(self.root, "token_embedding", None)
        if self.token_embedding is None:
            self.token_embedding = getattr(root, "token_embedding", None)
        assert self.token_embedding is not None, "No token_embedding found"

        self.positional_embedding = getattr(self.root, "positional_embedding", None)
        if self.positional_embedding is None:
            self.positional_embedding = getattr(root, "positional_embedding", None)

        self.transformer = getattr(self.root, "transformer", None)
        if self.transformer is None:
            self.transformer = getattr(root, "transformer", None)

        self.ln_final = getattr(self.root, "ln_final", None)
        if self.ln_final is None:
            self.ln_final = getattr(root, "ln_final", None)

        self.text_projection = getattr(self.root, "text_projection", None)
        if self.text_projection is None:
            self.text_projection = getattr(root, "text_projection", None)

        # attn_mask: parfois sur transformer
        self.attn_mask = getattr(self.root, "attn_mask", None)
        if self.attn_mask is None:
            self.attn_mask = getattr(self.transformer, "attn_mask", None)

        assert self.positional_embedding is not None, "No positional_embedding found"
        assert self.transformer is not None, "No transformer found"
        assert self.ln_final is not None, "No ln_final found"

    def forward(self, tokenized_prompts: torch.Tensor, ctx_for_classes: torch.Tensor):
        """
        tokenized_prompts: (C, L) (une séquence par classe)
        ctx_for_classes:   (C, n_ctx, D)
        """
        device = tokenized_prompts.device
        dtype = self.token_embedding.weight.dtype

        x = self.token_embedding(tokenized_prompts).to(dtype=dtype)  # (C,L,D)

        # inject ctx dans positions 1..1+n_ctx
        x = x.clone()
        x[:, 1:1 + self.n_ctx, :] = ctx_for_classes.to(device=device, dtype=dtype)

        # add pos
        pe = self.positional_embedding
        if pe.dim() == 2:
            pe = pe[:x.size(1), :].to(device=device, dtype=dtype)     # (L,D)
            x = x + pe.unsqueeze(0)                                   # (C,L,D)
        else:
            pe = pe[:, :x.size(1), :].to(device=device, dtype=dtype)  # (1,L,D)
            x = x + pe

        # open_clip: transformer attend souvent (L,B,D), pas (B,L,D)
        # On gère les deux cas simplement via un try.
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=device)

        try:
            # certains open_clip acceptent (B,L,D) + attn_mask kwarg
            x = self.transformer(x, attn_mask=attn_mask)
        except TypeError:
            # fallback (L,B,D)
            x = x.permute(1, 0, 2)     # (L,C,D)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)     # (C,L,D)

        x = self.ln_final(x)

        # pool au EOT
        eot_pos = tokenized_prompts.argmax(dim=-1)  # CLIP-style EOT == max token id
        x = x[torch.arange(x.shape[0], device=device), eot_pos]  # (C,D)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x


def detect_layout(visual, device):
    layout = {"shape": None}

    blk0 = visual.transformer.resblocks[0]

    def hook(module, inp, out):
        layout["shape"] = tuple(inp[0].shape)

    h = blk0.register_forward_hook(hook)

    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224, device=device)
        _ = visual(x)

    h.remove()
    return layout["shape"]


# -------------------------
# Main
# -------------------------
# def main():
#     import open_clip

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     torch.manual_seed(0)

#     # ⚠️ adapte ces noms à ta config Quilt exacte
#     print("[LOAD] model_name =", "Quilt-B/16")

#     model_id = "hf-hub:wisdomik/QuiltNet-B-16"
#     model, preprocess = open_clip.create_model_from_pretrained(model_id, device="cpu")
#     # tokenizer = open_clip.get_tokenizer(model_id)

#     model = model.to(device)
#     model.eval()

#     B, H, W = 2, 224, 224
#     img = torch.randn(B, 3, H, W, device=device)
#     img2 = img.clone()
#     img2[1] = img2[1] * 0.0 + 5.0  # force différence énorme
#     tinfo(img2, "img2")
#     diff01(img2.flatten(1), "img2")

#     # 1) VPT forward
#     vpt = OpenClipVisionVPT(model.visual, n_ctx=4).to(img2.device).eval()

#     with torch.no_grad():
#         f_base = model.encode_image(img2)
#         f_base = f_base / f_base.norm(dim=-1, keepdim=True)

#         f_vpt = vpt(img2)
#         f_vpt = f_vpt / f_vpt.norm(dim=-1, keepdim=True)

#     print("\n[CHECK] base")
#     diff01(f_base, "f_base(normed)")

#     print("\n[CHECK] vpt")
#     diff01(f_vpt, "f_vpt(normed)")

#     print("\n[COMPARE] base vs vpt (normed)")
#     print("maxdiff(f_base, f_vpt) =", maxdiff(f_base, f_vpt))

#     # 2) Gradient sanity: does VPT get gradient?
#     vpt.train()
#     for p in vpt.parameters():
#         if p.grad is not None:
#             p.grad = None

#     # simple classification head just for grad signal
#     head = torch.nn.Linear(f_vpt.shape[-1], 3, bias=False).to(img2.device).train()
#     opt = torch.optim.SGD(list(vpt.parameters()) + list(head.parameters()), lr=0.1)

#     lab = torch.tensor([0, 1], device=img2.device)
#     feat = vpt(img2)  # (B,512)
#     logits = head(feat)
#     loss = F.cross_entropy(logits, lab)

#     opt.zero_grad(set_to_none=True)
#     loss.backward()

#     g = vpt.VPT0.grad
#     print("\n[GRAD] loss =", float(loss.item()))
#     print("[GRAD] VPT0 grad is None?", g is None)
#     if g is not None:
#         print("[GRAD] VPT0 grad finite:", torch.isfinite(g).all().item())
#         print("[GRAD] VPT0 grad mean abs:", g.abs().mean().item())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_id = "hf-hub:wisdomik/QuiltNet-B-16"
    model, preprocess = open_clip.create_model_from_pretrained(model_id, device="cpu")
    tokenizer = open_clip.get_tokenizer(model_id)
    model = model.to(device).eval()

    prompts = ["a photo of a cat.", "a photo of a dog."]
    tok = tokenizer(prompts)
    if isinstance(tok, dict):
        tok = tok.get("input_ids", tok[list(tok.keys())[0]])
    tok = torch.as_tensor(tok, device=device).long()  # (2,L)

    # baseline
    with torch.no_grad():
        t_base = model.encode_text(tok)
        t_base = t_base / t_base.norm(dim=-1, keepdim=True)

    # ---- test wrapper n_ctx=0 (doit matcher)
    enc0 = QuiltTextEncoder(model, n_ctx=0).to(device).eval()
    ctx0 = torch.empty(tok.size(0), 0, model.token_embedding.weight.shape[1], device=device)
    with torch.no_grad():
        t0 = enc0(tok, ctx0)
        t0 = t0 / t0.norm(dim=-1, keepdim=True)
    print("[TEXT] maxdiff(base, wrapper n_ctx=0) =", maxdiff(t_base, t0))

    # ---- test wrapper n_ctx=4 (doit différer) + grads
    n_ctx = 4
    enc = QuiltTextEncoder(model, n_ctx=n_ctx).to(device).train()

    # ctx trainable
    D = model.token_embedding.weight.shape[1]
    ctx = nn.Parameter(torch.randn(tok.size(0), n_ctx, D, device=device) * 0.02)

    # simple loss: rendre cat plus proche de dog (ou n’importe) -> juste pour générer grad
    t = enc(tok, ctx)
    t = t / t.norm(dim=-1, keepdim=True)
    loss = (t[0] @ t[1]).neg()  # veut diminuer la similarité
    loss.backward()

    with torch.no_grad():
        t = t.detach()
        t = t / t.norm(dim=-1, keepdim=True)

    print("[TEXT] maxdiff(base, wrapper n_ctx=4) =", maxdiff(t_base, t))
    print("[TEXT][GRAD] ctx grad None?", ctx.grad is None)
    if ctx.grad is not None:
        print("[TEXT][GRAD] ctx grad finite:", torch.isfinite(ctx.grad).all().item())
        print("[TEXT][GRAD] ctx grad mean abs:", ctx.grad.abs().mean().item())


if __name__ == "__main__":
    main()
