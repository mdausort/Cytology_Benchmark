import clip
import torch
from tqdm import tqdm
from open_clip import get_tokenizer
import sys
import torch.nn as nn
from pathlib import Path
import conch.open_clip_custom
from torch.nn.utils.rnn import pad_sequence


def setup_logging(args):
    if getattr(args, "log_path", None):
        log_dir = Path(args.log_path)
    else:
        if args.task == "lora-vision":
            log_dir = (
                Path("./Cytology_Benchmark/output")
                / str(args.dataset)
                / "CLIP-LoRA"
                / f"{args.encoder}"
                / str(args.model_name)
                / f"rank{args.r}"
                / f"{args.shots}shots"
                / f"seed{args.seed}"
            )
        elif args.task == "feature_extract":
            log_dir = (
                Path("./Cytology_Benchmark/output")
                / str(args.dataset)
                / "CLIP-LoRA"
                / f"{args.encoder}_feat"
                / str(args.model_name)
                / f"{args.shots}shots"
                / f"seed{args.seed}"
            )
        else:
            log_dir = (
                Path("./Cytology_Benchmark/output")
                / str(args.dataset)
                / "CLIP-LoRA"
                / f"{args.encoder}_vlm"
                / str(args.model_name)
                / f"rank{args.r}"
                / f"{args.shots}shots"
                / f"seed{args.seed}"
            )

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.txt"

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    f = open(log_file, "w")
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)

    print(f"[LOGGING] {log_file}")
    return log_dir


def _model_device_dtype(m):
    for p in m.parameters():
        return p.device, p.dtype
    return torch.device("cpu"), torch.float32


def _get_input_ids_any(tokens, *, pad_token_id=0, max_len=77):
    if hasattr(tokens, "data") and isinstance(tokens.data, dict):
        if "input_ids" in tokens.data:
            tokens = tokens.data["input_ids"]
    elif hasattr(tokens, "__contains__") and "input_ids" in tokens:
        tokens = tokens["input_ids"]

    if torch.is_tensor(tokens):
        ids = tokens.long()
        if max_len is not None and ids.dim() == 2:
            ids = ids[:, :max_len]
            if ids.shape[1] < max_len:
                pad = torch.full((ids.shape[0], max_len - ids.shape[1]),
                                 pad_token_id, dtype=ids.dtype, device=ids.device)
                ids = torch.cat([ids, pad], dim=1)
        return ids

    if isinstance(tokens, (list, tuple)) and len(tokens) > 0 and isinstance(tokens[0], (list, tuple)):
        seqs = [torch.tensor(x[:max_len], dtype=torch.long) for x in tokens]
        ids = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)
        if max_len is not None:
            if ids.shape[1] < max_len:
                pad = torch.full((ids.shape[0], max_len - ids.shape[1]),
                                 pad_token_id, dtype=torch.long)
                ids = torch.cat([ids, pad], dim=1)
            else:
                ids = ids[:, :max_len]
        return ids

    if isinstance(tokens, (list, tuple)) and (len(tokens) == 0 or isinstance(tokens[0], int)):
        ids = torch.tensor(tokens[:max_len], dtype=torch.long).unsqueeze(0)
        if max_len is not None and ids.shape[1] < max_len:
            pad = torch.full((1, max_len - ids.shape[1]), pad_token_id, dtype=torch.long)
            ids = torch.cat([ids, pad], dim=1)
        return ids

    raise TypeError(f"Unsupported tokens type for input_ids: {type(tokens)}")


def get_openclip_tokenizer_for_conch(clip_model):
    if hasattr(conch.open_clip_custom, "get_tokenizer"):
        try:
            return conch.open_clip_custom.get_tokenizer()
        except TypeError:
            pass

    if hasattr(conch.open_clip_custom, "tokenize"):
        fn = conch.open_clip_custom.tokenize

        def _wrapped(texts):
            try:
                return fn(texts)
            except TypeError:
                pass
            try:
                return fn(clip_model, texts)
            except TypeError:
                pass
            tok_obj = getattr(clip_model, "tokenizer", None)
            if tok_obj is not None:
                try:
                    return fn(tok_obj, texts)
                except TypeError:
                    pass
            raise TypeError(f"Unsupported Conch tokenize signature: {fn}")

        return _wrapped

    return get_tokenizer("ViT-B-16")


class DinoEncodeImage(nn.Module):
    def __init__(self, dino, pool="cls"):
        super().__init__()
        self.dino = dino
        self.pool = pool

    def forward(self, x):
        out = self.dino(x)

        if torch.is_tensor(out) and out.dim() == 2:
            return out

        if torch.is_tensor(out) and out.dim() == 3:
            if self.pool == "cls":
                return out[:, 0]
            return out.mean(dim=1)

        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            if "x_norm_patchtokens" in out:
                pt = out["x_norm_patchtokens"]
                return pt.mean(dim=1) if self.pool == "mean" else pt[:, 0]
            if "last_hidden_state" in out:
                t = out["last_hidden_state"]
                return t.mean(dim=1) if self.pool == "mean" else t[:, 0]

        if isinstance(out, (tuple, list)):
            t = out[0]
            if torch.is_tensor(t):
                if t.dim() == 2:
                    return t
                if t.dim() == 3:
                    return t.mean(dim=1) if self.pool == "mean" else t[:, 0]

        raise RuntimeError(f"Unsupported DINO output type: {type(out)}")


def get_function(args, model, hf_processor=None):

    name = str(args.model_name).lower()

    if name in ["clip-b16", "clip-b32", "clip-l14"]:
        return model.encode_image, model.encode_text, clip.tokenize

    if name == "quilt-b16":
        tok = get_tokenizer("hf-hub:wisdomik/QuiltNet-B-16")
        return model.encode_image, model.encode_text, tok

    if name == "quilt-b32":
        tok = get_tokenizer("hf-hub:wisdomik/QuiltNet-B-32")
        return model.encode_image, model.encode_text, tok

    if name == "biomedclip":
        tok = get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

        vision_fn = (
            model.encode_image if hasattr(model, "encode_image") else model.visual
        )
        text_fn = model.encode_text if hasattr(model, "encode_text") else model.text

        return vision_fn, text_fn, tok

    if name == "conch":
        tok = get_openclip_tokenizer_for_conch(model)

        def vision_fn(images):
            dev, dt = _model_device_dtype(model)
            return model.encode_image(images.to(device=dev, dtype=dt))

        def text_fn(tokens):
            dev, _ = _model_device_dtype(model)

            if hasattr(tokens, "data") and isinstance(tokens.data, dict):
                t = {k: v.to(dev) for k, v in tokens.data.items() if torch.is_tensor(v)}
                try:
                    return model.encode_text(**t)
                except TypeError:
                    pass

            input_ids = _get_input_ids_any(tokens).to(dev)
            return model.encode_text(input_ids)

        def token_fn(texts, **kwargs):
            if len(kwargs) == 0:
                kwargs = dict(padding="max_length", truncation=True, max_length=77, return_tensors="pt")
            try:
                return tok(texts, **kwargs)
            except TypeError:
                return tok(texts)

        return vision_fn, text_fn, token_fn

    if name in ["pubmedclip", "plip"]:
        if hf_processor is None:
            hf_processor = getattr(args, "_hf_processor", None)
        if hf_processor is None:
            raise RuntimeError("HF processor missing (set args._hf_processor in main)")

        def vision_fn(images):
            return model.get_image_features(pixel_values=images)

        def text_fn(tokens):
            if hasattr(tokens, "to_dict"):
                tokens = tokens.data
            return model.get_text_features(**tokens)

        def token_fn(texts):
            return hf_processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        return vision_fn, text_fn, token_fn

    if "dinobloom" in name:
        encoder = DinoEncodeImage(model, pool="mean")
        return encoder, None, None

    if name in ["vit_google-b16", "vit_google-b32"]:
        return model, None, None

    if name == "uni":
        return model, None, None


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]

    return acc


def _device_of_callable(fn):
    if hasattr(fn, "parameters"):
        try:
            return next(fn.parameters()).device
        except StopIteration:
            pass
    mod = getattr(fn, "__self__", None)
    if mod is not None and hasattr(mod, "parameters"):
        try:
            return next(mod.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_tokens(tokens, device):
    """Move HF BatchEncoding / dict / tensor to device."""
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if torch.is_tensor(tokens):
        return tokens.to(device)
    if isinstance(tokens, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tokens.items()}
    raise TypeError(f"Unsupported token type: {type(tokens)}")


@torch.no_grad()
def clip_classifier(args, classnames, template, clip_model):
    vision_fn, text_fn, token_fn = get_function(args, clip_model)

    text_device = _device_of_callable(text_fn)

    clip_weights = []

    for classname in classnames:
        classname = classname.replace("_", " ")
        prompts = [t.format(classname) for t in template]

        tokens = token_fn(prompts)
        tokens = _move_tokens(tokens, text_device)

        use_amp = (text_device.type == "cuda")
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.autocast("cpu", enabled=False)

        with amp_ctx:
            class_embeddings = text_fn(tokens)

        class_embeddings = class_embeddings.float()
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()

        clip_weights.append(class_embedding)

    clip_weights = torch.stack(clip_weights, dim=1).to(text_device)
    return clip_weights


def pre_load_features(args, clip_model, loader):
    vision, text, token = get_function(args, clip_model)

    if hasattr(vision, "eval"):
        vision.eval()
    else:
        parent = getattr(vision, "__self__", None)
        if parent is not None and hasattr(parent, "eval"):
            parent.eval()

    device = _device_of_callable(vision)

    features, labels = [], []

    for batch in tqdm(loader):
        if isinstance(batch, dict):
            images = batch["img"]
            target = batch["label"]
        else:
            images, target = batch[0], batch[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = vision(images)
        else:
            image_features = vision(images)

        image_features = image_features.float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        features.append(image_features.cpu())
        labels.append(target.cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)
