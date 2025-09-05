import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import gensim.downloader as api
from tqdm.auto import tqdm, trange


TOKEN_RE = re.compile(r"[a-z]+")


def toks(s: str) -> List[str]:
    """Split text into lowercase tokens using regex"""
    return TOKEN_RE.findall((s or "").lower())


def build_vocab(df_list: List[pd.DataFrame], min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from query and passage texts, keeping words that appear at least min_freq times"""
    c = Counter()
    for df in df_list:
        for txt in pd.concat([df["query_text"], df["passage_text"]]).fillna(""):
            c.update(toks(txt))
    # special tokens
    vocab = {"<pad>": 0, "<oov>": 1}
    for w, f in c.items():
        if f >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def load_keyed_vectors(name_or_none: Optional[str] = "glove-wiki-gigaword-200"):
    """Load pre-trained gensim embeddings"""
    try:
        kv = api.load(name_or_none)  # glove-wiki-gigaword-200 or fasttext-wiki-news-subwords-300
        return kv, kv.vector_size
    except Exception as e:
        print(f"Could not load gensim vectors: {e}")
        return None, 300


def build_embedding_matrix(vocab, kv=None, dim=300, pad_token="<pad>", oov_token="<oov>", seed=798):
    """
    Build an embedding matrix
    Args:
        vocab: dict token -> index
        kv: pre-trained embeddings
        dim: embedding dimensionality
    """
    if kv is not None:
        dim = kv.vector_size

    pad_id = vocab[pad_token]
    oov_id = vocab[oov_token]

    M = np.zeros((len(vocab), dim), dtype=np.float32)

    # single shared out-of-vocabulary vector
    rs = np.random.RandomState(seed)
    oov_vec = rs.uniform(-0.2, 0.2, size=dim).astype(np.float32)
    n = np.linalg.norm(oov_vec)
    if n > 0:
        oov_vec /= n

    M[pad_id] = 0.0
    M[oov_id] = oov_vec

    # fill the rest
    for token, idx in vocab.items():
        if idx in (pad_id, oov_id):
            continue

        vec = None
        if kv is not None:
            try:
                v = kv.get_vector(token).astype(np.float32)
                vn = np.linalg.norm(v)
                vec = v / vn if vn > 0 else v
            except KeyError:
                pass

        M[idx] = vec if vec is not None else oov_vec

    return M


def encode_tokens(text: str, vocab: Dict[str, int], max_len: int, oov_token="<oov>") -> List[int]:
    """Convert text to list of token IDs, truncate to max_len, use OOV for unknown words"""
    ids = [vocab.get(t, vocab[oov_token]) for t in toks(text)]
    return ids[:max_len] if max_len > 0 else ids


class TripletDataset(torch.utils.data.Dataset):
    """Holds (qid, pos_pid, neg_pid) triplets"""
    def __init__(self, triplets: List[Tuple[str, str, str]], q_text: Dict[str, str], p_text: Dict[str, str],
                 vocab: Dict[str, int], max_q_len: int = 32, max_p_len: int = 256):
        self.tri = triplets
        self.q_text = q_text
        self.p_text = p_text
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len
        self.vocab = vocab

    def __len__(self):
        return len(self.tri)

    def __getitem__(self, i):
        qid, pos_pid, neg_pid = self.tri[i]
        q = encode_tokens(self.q_text[qid], self.vocab, self.max_q_len)
        p_pos = encode_tokens(self.p_text[pos_pid], self.vocab, self.max_p_len)
        p_neg = encode_tokens(self.p_text[neg_pid], self.vocab, self.max_p_len)
        return {"q": q, "p_pos": p_pos, "p_neg": p_neg}


def collate_triplet(batch):
    """Pad query/positive/negative passages to same length"""
    qs, pos, neg = [], [], []
    # compute max lens
    max_q = max(len(b["q"]) for b in batch) if batch else 0
    max_p = max(max(len(b["p_pos"]), len(b["p_neg"])) for b in batch) if batch else 0

    for b in batch:
        q = b["q"] + [0]*(max_q - len(b["q"]))
        ppos = b["p_pos"] + [0]*(max_p - len(b["p_pos"]))
        pneg = b["p_neg"] + [0]*(max_p - len(b["p_neg"]))
        qs.append(q)
        pos.append(ppos)
        neg.append(pneg)

    return (torch.tensor(qs, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long))
    

def build_id_maps(df: pd.DataFrame):
    """Create dictionaries mapping query IDs to text and passage IDs to text"""
    qid2q = df[["qid", "query_text"]].drop_duplicates().set_index("qid")["query_text"].to_dict()
    pid2p = df[["pid", "passage_text"]].drop_duplicates().set_index("pid")["passage_text"].to_dict()
    return qid2q, pid2p


def make_triplets(pairs: pd.DataFrame, seed: int = 77, per_pos_neg: int = 2) -> List[Tuple[str, str, str]]:
    """Create training triplets: query ID, positive passage ID, negative passage ID"""
    assert per_pos_neg > 0

    rng = np.random.default_rng(seed)
    triplets = []
    groups = pairs.groupby("qid", sort=False)

    for qid, g in groups:

        pos = g.loc[g["label"] == 1, "pid"].tolist()

        neg_all = g.loc[g["label"] == 0, ["pid", "pas_type"]].copy()
        order = ["imp_hard", "bm25_hard", "imp_med", "easy"]
        neg_pid_ordered = []

        for tag in order:
            neg_pid_ordered += neg_all.loc[neg_all["pas_type"] == tag, "pid"].tolist()

        if not pos or not neg_pid_ordered:
            continue

        # for each positive sample up to per_pos_neg negatives
        for ppos in pos:
            n_choices = min(per_pos_neg, len(neg_pid_ordered))
            chosen = rng.choice(neg_pid_ordered, size=n_choices, replace=False)
            for pneg in chosen:
                triplets.append((qid, ppos, pneg))

    return triplets


class GaussianKernel(nn.Module):
    """Gaussian kernel function for similarity scoring"""
    def __init__(self, mu: float, sigma: float):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=False)
    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2))


class KNRM(nn.Module):
    """Kernel-based Neural Ranking Model for passage ranking"""
    def __init__(self, emb_matrix: np.ndarray,
                 freeze: bool = True,
                 K: int = 11,
                 sigma: float = 0.1,
                 hidden: List[int] = [10, 5],
                 dropout: float = 0.4,
                 pad_id: int = 0):
        super().__init__()

        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=freeze, padding_idx=pad_id)

        mus, sig = self._kernel_settings(K=K, sigma=sigma)
        self.kernels = nn.ModuleList([GaussianKernel(float(m), float(s)) for m, s in zip(mus, sig)])

        layers = []
        dims = [K] + hidden + [1]
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if b != 1:
                layers += [nn.ReLU(), nn.Dropout(p=dropout)]
        self.mlp = nn.Sequential(*layers)

    def _kernel_settings(self, K: int = 11, sigma: float = 0.1):
        """Configure kernel positions and widths"""
        mus = np.linspace(-1.0, 1.0, K, dtype=np.float32)
        sig = np.full(K, sigma, dtype=np.float32)
        sig[-1] = 1e-3  # sharpen exact match near 1
        return mus, sig

    def _sim_matrix(self, Qids: torch.LongTensor, Pids: torch.LongTensor) -> torch.FloatTensor:
        """Compute cosine similarity matrix between query and passage tokens"""
        # [B, Lq, D]  [B, Lp, D]
        Q = self.emb(Qids)
        P = self.emb(Pids)
        Q = F.normalize(Q, p=2, dim=-1)
        P = F.normalize(P, p=2, dim=-1)
        # [B, Lq, Lp]
        S = torch.einsum("bld,brd->blr", Q, P)
        return S.clamp_(-1.0, 1.0)

    def _kernel_pool(self, S: torch.FloatTensor, Qids: torch.LongTensor, Pids: torch.LongTensor) -> torch.FloatTensor:
        """Apply Gaussian kernels to similarity matrix and pool features"""
        # mask zeroes (padding tokens)
        qmask = (Qids != self.emb.padding_idx).float().unsqueeze(-1) # [B, Lq, 1]
        pmask = (Pids != self.emb.padding_idx).float().unsqueeze(-2) # [B, 1, Lp]
        mask = qmask * pmask # [B, Lq, Lp]

        pooled = []
        for kernel in self.kernels:
            Kmat = kernel(S) * mask
            sum_j = Kmat.sum(dim=-1) # [B, Lq]
            feat = torch.log1p(sum_j).sum(dim=-1) # [B]
            pooled.append(feat)
        return torch.stack(pooled, dim=1) # [B, K]

    def score(self, Qids: torch.LongTensor, Pids: torch.LongTensor) -> torch.FloatTensor:
        """Compute relevance score between query and passage"""
        S = self._sim_matrix(Qids, Pids)
        Fk = self._kernel_pool(S, Qids, Pids)
        return self.mlp(Fk).squeeze(-1)  # [B]


def count_parameters(model):
    """
    Count number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_val_loss(model, val_dl, device, criterion) -> float:
    """
    Evaluate KNRM model using Pairwise Ranking Loss
    """
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for Q, Ppos, Pneg in tqdm(val_dl, desc="Validation loss", leave=False):
            Q, Ppos, Pneg = Q.to(device), Ppos.to(device), Pneg.to(device)
            s_pos = model.score(Q, Ppos)
            s_neg = model.score(Q, Pneg)
            logits = s_pos - s_neg
            loss = criterion(logits, torch.ones_like(logits))
            bs = Q.size(0)
            total += loss.item() * bs
            n += bs
    return total / max(1, n)


def train_knrm_model(model, train_dl, val_dl, criterion, optimizer, sched, device,
                    epochs=10, patience=3,
                    model_save_path="knrm_model.pt"):
    """
    Train KNRM model with early stopping and val monitoring
    Returns:
        dict: training history with losses
    """

    best_val = float("inf")
    improvement_th = 1e-4
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'best_epoch': 0}

    for epoch in trange(1, epochs + 1, desc="Epochs", leave=True):

        model.train()
        total_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Train | epoch {epoch}", leave=False)
        for Q, Ppos, Pneg in pbar:

            Q, Ppos, Pneg = Q.to(device), Ppos.to(device), Pneg.to(device)
            s_pos = model.score(Q, Ppos)
            s_neg = model.score(Q, Pneg)
            logits = s_pos - s_neg
            loss = criterion(logits, torch.ones_like(logits))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            sched.step()

            total_loss += loss.item() * Q.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / len(train_dl.dataset)
        history['train_loss'].append(avg_train_loss)

        # validation
        val_loss = evaluate_val_loss(model, val_dl, device, criterion)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f}")

        # check for improvement and save best model
        if best_val - val_loss > improvement_th:
            best_val = val_loss
            epochs_no_improve = 0
            history['best_epoch'] = epoch
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch,
                 "train_loss": avg_train_loss, "val_loss": val_loss}, model_save_path)
            print(f"New best model saved with val_loss={val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print("Early stopping triggered...")
                break

    return history


def predict_knrm_scores(df, model, device, vocab, max_q_len=32, max_p_len=256, batch_size=1024, pad_id=0):
    """
    Get knrm scores row by row for a given dataframe
    """
    model.eval()
    N = len(df)
    scores = np.empty(N, dtype=np.float32)

    q_tok = [encode_tokens(q, vocab, max_q_len) for q in df["query_text"]]
    p_tok = [encode_tokens(p, vocab, max_p_len) for p in df["passage_text"]]

    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            q_batch = q_tok[i:j]
            p_batch = p_tok[i:j]

            # dynamic padding within batch
            max_q = max((len(x) for x in q_batch), default=0)
            max_p = max((len(x) for x in p_batch), default=0)

            Q = torch.tensor([x + [pad_id]*(max_q - len(x)) for x in q_batch], dtype=torch.long, device=device)
            P = torch.tensor([x + [pad_id]*(max_p - len(x)) for x in p_batch], dtype=torch.long, device=device)

            s = model.score(Q, P).detach().cpu().numpy().astype(np.float32)
            scores[i:j] = s

    return scores


def plot_loss(history):
    """
    Function to plot train and validation losses of a neural network
    """
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, history['train_loss'], 'o-', linewidth=2, markersize=6, label='Training Loss', color='#2E86AB')
    plt.plot(epochs, history['val_loss'], 's-', linewidth=2, markersize=6, label='Validation Loss', color='#A23B72')

    best_epoch = history['best_epoch']
    best_val_loss = history['val_loss'][best_epoch-1]
    plt.axvline(x=best_epoch, color='#F18F01', linestyle='--', alpha=0.8,  linewidth=1.5, label=f'Best Epoch ({best_epoch})')
    plt.plot(best_epoch, best_val_loss, 'o', markersize=10, color='#F18F01', markeredgecolor='black')

    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.title('Training and Validation Loss', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.xticks(epochs)

    plt.annotate(f'Best: {best_val_loss:.4f}', 
                xy=(best_epoch, best_val_loss),
                xytext=(best_epoch+0.3, best_val_loss+0.005),
                fontweight='bold',
                color='#A23B72')

    plt.tight_layout()
    plt.show()
