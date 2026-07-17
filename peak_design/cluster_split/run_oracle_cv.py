"""Cross-cluster CV for the Oracle (cnn-concatstd-d1). Populates a per-fold cache so the
companion notebook reads results instantly. Nothing existing is modified."""
import csv, os, sys, numpy as np, torch, hashlib, time
# This script lives in peak_design/cluster_split/; peak_models.py + the shared
# oracle_cv_cache/ live one level up in peak_design/, the dataset two levels up.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PEAK_DESIGN = os.path.dirname(_HERE)
sys.path.insert(0, _PEAK_DESIGN)
import peak_models as pm
from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit

CUR=os.path.join(_HERE, "..", "..", "dataset_pipeline", "data", "peak", "curated")
CACHE=os.path.join(_PEAK_DESIGN, "oracle_cv_cache"); os.makedirs(CACHE, exist_ok=True)
dev=torch.device("mps") if (getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()) \
    else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

rows=list(csv.DictReader(open(os.path.join(CUR,"peaks_assignments.csv")))); N=len(rows)
peaks=np.load(os.path.join(CUR,"peaks.npy")).astype(np.float32)
H=np.load(os.path.join(CUR,"esm_residue_fp16.npy")); Ls=np.load(os.path.join(CUR,"esm_residue_len.npy"))
Lmax=int(Ls.max()); H=H[:,:Lmax]; Ht=torch.tensor(H); ar=torch.arange(Lmax)
Pk=torch.tensor(peaks, dtype=torch.float32, device=dev)

ORACLE_SPEC=dict(arch="cnn", pool="concatstd", conv_ch=128, k=5, n_conv=1, hidden=256, nl=2, sid="cnn-concatstd-d1")
EPOCHS=150; LR=1e-3; WD=1e-4; DROP=0.2; NSPLITS=5; SEED=0

def batches(idx,bs=32,shuffle=False):
    idx=np.array(idx)
    if shuffle: np.random.shuffle(idx)
    for i in range(0,len(idx),bs):
        b=idx[i:i+bs]
        yield Ht[b].float().to(dev),(ar.unsqueeze(0)<torch.tensor(Ls[b]).unsqueeze(1)).to(dev),b

def predict(net, idx):
    net.eval(); ps=[]
    with torch.no_grad():
        for Hb,mk,b in batches(idx): ps.append(net(Hb,mk).cpu().numpy())
    return np.concatenate(ps)

def _mae(P,T): return 0.5*(np.abs(P[:,0]-T[:,0]).mean()+np.abs(P[:,1]-T[:,1]).mean())

def train_oracle(tr, va, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    mean=peaks[tr].mean(0).astype("float32"); std=(peaks[tr].std(0)+1e-6).astype("float32")
    base=pm.build_base(ORACLE_SPEC, dev, drop=DROP); net=pm.wrap(base, mean, std, dev)
    sd=torch.tensor(std,device=dev)
    opt=torch.optim.Adam(net.parameters(), LR, weight_decay=WD); best=1e9; bst=None
    for ep in range(EPOCHS):
        net.train()
        for Hb,mk,b in batches(tr,shuffle=True):
            opt.zero_grad(); pred=net(Hb,mk); tgt=Pk[torch.as_tensor(b,device=dev)]
            loss=(((pred-tgt)/sd)**2).mean(); loss.backward(); opt.step()
        v=_mae(predict(net,va), peaks[va])
        if v<best: best=v; bst={k:vv.cpu().clone() for k,vv in net.base.state_dict().items()}
    net.base.load_state_dict(bst); return net

def inner_val(train_idx, groups, fold):
    """Carve ~15% of the fold's training pool as an early-stopping val set (cluster-disjoint when grouped)."""
    if groups is None:
        rng=np.random.default_rng(SEED+fold); perm=rng.permutation(train_idx)
        nv=int(round(0.15*len(train_idx))); return perm[nv:], perm[:nv]
    gss=GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tri,vai=next(gss.split(train_idx, groups=groups[train_idx]))
    return train_idx[tri], train_idx[vai]

def run_cv(scheme, groups):
    """5-fold OOF predictions. groups=None -> random KFold (interpolation); else GroupKFold (extrapolation)."""
    idx=np.arange(N)
    if groups is None:
        splitter=KFold(n_splits=NSPLITS, shuffle=True, random_state=SEED).split(idx)
    else:
        splitter=GroupKFold(n_splits=NSPLITS).split(idx, groups=groups)
    oof=np.full((N,2), np.nan, dtype=np.float32); foldid=np.full(N,-1,dtype=int)
    for k,(train_idx,test_idx) in enumerate(splitter):
        fc=os.path.join(CACHE, f"fold_{scheme}_{k}.npz")
        if os.path.exists(fc):
            d=np.load(fc); ti=d["test_idx"]; oof[ti]=d["pred"]; foldid[ti]=k
            print(f"[{scheme}] fold {k}: cached (test n={len(ti)})", flush=True); continue
        tr,va=inner_val(train_idx, groups, k)
        t0=time.time(); net=train_oracle(tr, va, seed=SEED)
        pred=predict(net, test_idx); oof[test_idx]=pred; foldid[test_idx]=k
        np.savez(fc, test_idx=test_idx, pred=pred)
        print(f"[{scheme}] fold {k}: trained n_tr={len(tr)} n_va={len(va)} n_te={len(test_idx)} "
              f"| test MAE {_mae(pred,peaks[test_idx]):.1f} nm | {time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(CACHE, f"oof_{scheme}.npz"), oof=oof, foldid=foldid)
    return oof, foldid

def load_clusters(pct):
    p=os.path.join(CUR, f"seqid{pct}_clusters.csv")
    r=sorted(csv.DictReader(open(p)), key=lambda x:int(x["index"]))
    return np.array([int(x["cluster_id"]) for x in r])

if __name__=="__main__":
    clab70=load_clusters(70); clab85=load_clusters(85)
    print("groups: 70% ->", len(set(clab70)), "clusters | 85% ->", len(set(clab85)), "clusters", flush=True)
    run_cv("random", None)       # interpolation baseline
    run_cv("group70", clab70)    # extrapolation @70%
    run_cv("group85", clab85)    # extrapolation @85%
    print("ALL CV DONE", flush=True)
