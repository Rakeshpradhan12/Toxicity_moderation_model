import os, sys, time, random
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np, pandas as pd, torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

QUICK = "--quick" in sys.argv          # small test run to check the setup
MODEL_DIR = "models/original"
OUT_DIR = "models/ft_best_quick" if QUICK else "models/ft_best"
LANGS = ["en", "hi", "hin"]
EXTRA_CSVS = []                        # e.g. ["odia_train.csv"]  (columns: text, lang, label)
UPSAMPLE = {"or": 3}
MAX_LEN = 128
BATCH = 16                             # lower to 8 if you get a memory error
EPOCHS = 1 if QUICK else 2
LR = 2e-5
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, "| quick run:", QUICK)

# ---- data ----
frames = []
for lang in LANGS:
    path = hf_hub_download(repo_id="textdetox/multilingual_toxicity_dataset",
                           filename=f"data/{lang}-00000-of-00001.parquet", repo_type="dataset")
    d = pd.read_parquet(path).rename(columns={"toxic": "label"})[["text", "label"]]
    d["lang"] = lang
    frames.append(d)
for p in EXTRA_CSVS:
    frames.append(pd.read_csv(p)[["text", "lang", "label"]])

data = pd.concat(frames, ignore_index=True).dropna(subset=["text", "label"])
data["label"] = data["label"].astype(int)
data = data.drop_duplicates(subset="text").reset_index(drop=True)
data["strat"] = data["lang"] + "_" + data["label"].astype(str)

train, rest = train_test_split(data, test_size=0.2, stratify=data["strat"], random_state=SEED)
val, test = train_test_split(rest, test_size=0.5, stratify=rest["strat"], random_state=SEED)
if QUICK:
    train = train.sample(min(1500, len(train)), random_state=SEED)
    val = val.sample(min(300, len(val)), random_state=SEED)
    test = test.sample(min(600, len(test)), random_state=SEED)
extra = [pd.concat([train[train.lang == l]] * (k - 1)) for l, k in UPSAMPLE.items() if (train.lang == l).any()]
train = pd.concat([train] + extra, ignore_index=True)
print(train.groupby(["lang", "label"]).size().unstack())
print("val:", len(val), "test:", len(test))

# ---- model ----
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
TOX = model.config.label2id["toxicity"]

def collate(batch):
    y = torch.tensor([b[1] for b in batch], dtype=torch.float)
    enc = tok([b[0] for b in batch], truncation=True, max_length=MAX_LEN, padding=True,
              return_tensors="pt", return_token_type_ids=False)
    return enc, y

def make_loader(df, shuffle, bs):
    return DataLoader(list(zip(df["text"].tolist(), df["label"].tolist())),
                      batch_size=bs, shuffle=shuffle, collate_fn=collate)

@torch.no_grad()
def predict(df, bs=32):
    model.eval()
    out = []
    for enc, _ in make_loader(df, False, bs):
        enc = {k: v.to(device) for k, v in enc.items()}
        out.append(torch.sigmoid(model(**enc).logits[:, TOX].float()).cpu())
    return torch.cat(out).numpy()

def report(df, name):
    d = df.copy()
    d["tox"] = predict(d)
    aucs = {l: roc_auc_score(g["label"], g["tox"]) for l, g in d.groupby("lang")}
    print(name, {k: round(v, 3) for k, v in aucs.items()}, "mean:", round(float(np.mean(list(aucs.values()))), 3))
    return d, float(np.mean(list(aucs.values())))

report(test, "BEFORE fine-tuning, test AUC")

# ---- train ----
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
loader = make_loader(train, True, BATCH)
total = EPOCHS * len(loader)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
lossf = torch.nn.BCEWithLogitsLoss()
best, step, t0 = -1.0, 0, time.time()
for ep in range(EPOCHS):
    model.train()
    for enc, y in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        y = y.to(device)
        loss = lossf(model(**enc).logits[:, TOX].float(), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        step += 1
        if step % 50 == 0:
            el = time.time() - t0
            eta = el / step * (total - step)
            print(f"epoch {ep+1} step {step}/{total} loss {loss.item():.4f} | {el/60:.1f} min elapsed, ~{eta/60:.1f} min left")
    _, score = report(val, f"epoch {ep+1} validation AUC")
    if score > best:
        best = score
        model.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)
        print("  saved best so far to", OUT_DIR)

# ---- final check ----
model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)
final, _ = report(test, "AFTER fine-tuning, test AUC")
print("\nthreshold table (fine-tuned model, test split)")
print("lang  thr  good_flagged_%  bad_missed_%")
for lang, g in final.groupby("lang"):
    for th in [0.3, 0.5, 0.7, 0.9]:
        pred = g["tox"] >= th
        gf = 100 * (pred & (g["label"] == 0)).sum() / max((g["label"] == 0).sum(), 1)
        bm = 100 * ((~pred) & (g["label"] == 1)).sum() / max((g["label"] == 1).sum(), 1)
        print(f"{lang:5} {th:.1f}  {gf:13.1f}  {bm:11.1f}")
print("done. model saved in", OUT_DIR)
