#!/usr/bin/env python3
"""
Sequence design from a masked sequence with ESM-2 (650M).

Given a protein sequence containing masked positions, this script uses Facebook
Research's ESM-2 650M model (``esm2_t33_650M_UR50D``, ~650 MB) as a masked
language model to predict the most plausible amino acids at the masked
positions. This is the standard "fill-in-the-blank" / sequence-design use of
ESM-2.

Note on ESMFold: the ESMFold entry in the ESM repo predicts 3D *structure* from
a sequence. It does not design sequence. For designing (in-painting) residues we
use the ESM-2 language model and its masked-LM head, as done here.

Masking
-------
Mark positions to design with the mask token. By default the mask character is
``#`` on the command line (so you don't have to escape ESM's ``<mask>`` token in
a shell), e.g.::

    MKTAYIAKQR####GFTLLILVDDDEK

You can also use ``_`` or any single character via ``--mask-char``.

Examples
--------
Greedy fill-in of a sequence with masked positions::

    python design.py --sequence "MKTAYIAKQR####GFTLLILVDDDEK"

Generate 5 designs by temperature sampling::

    python design.py --sequence "MKT##IAKQR" --num-samples 5 --temperature 1.0

Read from a FASTA file (mask positions with '#')::

    python design.py --fasta masked.fasta --num-samples 3

References
----------
ESM repo: https://github.com/facebookresearch/esm
Model:    esm2_t33_650M_UR50D (33 layers, 650M params)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch


MODEL_NAME = "esm2_t33_650M_UR50D"


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(device: torch.device):
    """Load the ESM-2 650M model, its alphabet, and a batch converter.

    Downloads weights (~650 MB) to the torch hub cache on first run.
    """
    try:
        import esm  # fair-esm
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "The 'fair-esm' package is required. Install it with:\n"
            "    pip install fair-esm torch\n"
            f"(import error: {exc})"
        )

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def read_fasta_first(path: str) -> tuple[str, str]:
    """Return (label, sequence) for the first record in a FASTA file."""
    label, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    break
                label = line[1:].strip() or "query"
            else:
                chunks.append(line)
    if not chunks:
        raise SystemExit(f"No sequence found in FASTA file: {path}")
    return label or "query", "".join(chunks)


def to_esm_sequence(raw: str, mask_char: str, mask_token: str) -> str:
    """Convert a user sequence (with mask_char) into an ESM input string.

    ESM expects masked positions as the literal ``<mask>`` token in the string
    passed to the batch converter.
    """
    raw = raw.strip().upper()
    mask_char = mask_char.upper()
    if mask_char not in raw and mask_token not in raw:
        raise SystemExit(
            f"No masked positions found. Mark positions to design with "
            f"'{mask_char}' (e.g. MKT{mask_char}{mask_char}IAKQR)."
        )
    # Replace single-char masks with the ESM mask token.
    return raw.replace(mask_char, mask_token)


# --------------------------------------------------------------------------- #
# Design / decoding
# --------------------------------------------------------------------------- #
@dataclass
class DesignResult:
    sequence: str          # full designed sequence (residues only, no specials)
    filled: list[tuple]    # (position_0based, amino_acid, probability)
    score: float           # mean log-prob of the filled residues


def _standard_aa_mask(alphabet) -> torch.Tensor:
    """Boolean mask over the alphabet selecting the 20 standard amino acids."""
    standard = "ACDEFGHIKLMNPQRSTVWY"
    keep = torch.zeros(len(alphabet.all_toks), dtype=torch.bool)
    for aa in standard:
        keep[alphabet.get_idx(aa)] = True
    return keep


@torch.no_grad()
def design_once(
    model,
    alphabet,
    batch_converter,
    esm_seq: str,
    device: torch.device,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    restrict_to_standard: bool = True,
    seed: int | None = None,
) -> DesignResult:
    """Fill all masked positions in a single forward pass.

    temperature == 0  -> greedy (argmax) decoding.
    temperature  > 0  -> sample from the (optionally top-k) softmax.
    """
    if seed is not None:
        torch.manual_seed(seed)

    _, _, tokens = batch_converter([("query", esm_seq)])
    tokens = tokens.to(device)

    logits = model(tokens)["logits"][0]  # (L, vocab)

    if restrict_to_standard:
        keep = _standard_aa_mask(alphabet).to(device)
        logits = logits.masked_fill(~keep, float("-inf"))

    mask_idx = alphabet.mask_idx
    mask_positions = (tokens[0] == mask_idx).nonzero(as_tuple=True)[0]

    full = tokens[0].clone()
    filled: list[tuple] = []
    log_probs_sum = 0.0

    for pos in mask_positions:
        pos_logits = logits[pos]
        probs = torch.softmax(pos_logits, dim=-1)

        if temperature and temperature > 0:
            scaled = pos_logits / temperature
            if top_k:
                topv, topi = torch.topk(scaled, k=min(top_k, scaled.numel()))
                filt = torch.full_like(scaled, float("-inf"))
                filt[topi] = topv
                scaled = filt
            sample_probs = torch.softmax(scaled, dim=-1)
            choice = torch.multinomial(sample_probs, num_samples=1).item()
        else:
            choice = int(torch.argmax(pos_logits).item())

        aa = alphabet.get_tok(choice)
        p = float(probs[choice].item())
        full[pos] = choice
        # position relative to residues: subtract 1 for the prepended BOS/cls
        filled.append((int(pos.item()) - 1, aa, p))
        log_probs_sum += float(torch.log(probs[choice] + 1e-9).item())

    # Reconstruct the residue-only sequence (drop BOS/EOS and any specials).
    special = {
        alphabet.cls_idx,
        alphabet.eos_idx,
        alphabet.padding_idx,
        alphabet.mask_idx,
    }
    residues = [
        alphabet.get_tok(int(t)) for t in full if int(t) not in special
    ]
    designed = "".join(residues)

    score = log_probs_sum / max(len(filled), 1)
    return DesignResult(sequence=designed, filled=filled, score=score)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Design masked positions in a protein sequence with ESM-2 650M.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", help="Masked sequence string (use --mask-char for masks).")
    src.add_argument("--fasta", help="FASTA file; first record is used.")

    p.add_argument("--mask-char", default="#",
                   help="Character marking positions to design (default: '#').")
    p.add_argument("--num-samples", type=int, default=1,
                   help="Number of designs to produce (default: 1).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy argmax (default: 0.0).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Restrict sampling to top-k tokens (only with temperature>0).")
    p.add_argument("--allow-nonstandard", action="store_true",
                   help="Allow non-standard tokens (B, U, Z, O, X, ...).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    p.add_argument("--device", default=None,
                   help="torch device, e.g. cpu / cuda / mps (default: auto).")
    return p


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.num_samples > 1 and args.temperature == 0.0:
        print(
            "[warn] --num-samples > 1 with temperature 0 gives identical greedy "
            "results; using temperature 1.0 for sampling.",
            file=sys.stderr,
        )
        args.temperature = 1.0

    device = pick_device(args.device)
    print(f"[info] device: {device}", file=sys.stderr)

    if args.fasta:
        label, raw = read_fasta_first(args.fasta)
    else:
        label, raw = "query", args.sequence

    print(f"[info] loading {MODEL_NAME} (~650 MB on first run)...", file=sys.stderr)
    model, alphabet, batch_converter = load_model(device)

    esm_seq = to_esm_sequence(raw, args.mask_char, "<mask>")
    n_masked = esm_seq.count("<mask>")
    print(f"[info] query '{label}': {n_masked} masked position(s)\n", file=sys.stderr)

    for i in range(args.num_samples):
        seed = None if args.seed is None else args.seed + i
        res = design_once(
            model, alphabet, batch_converter, esm_seq, device,
            temperature=args.temperature,
            top_k=args.top_k,
            restrict_to_standard=not args.allow_nonstandard,
            seed=seed,
        )
        tag = f"design_{i + 1}" if args.num_samples > 1 else "design"
        print(f">{label}|{tag}|mean_logp={res.score:.3f}")
        print(res.sequence)
        fills = "  ".join(f"{pos + 1}{aa}({p:.2f})" for pos, aa, p in res.filled)
        print(f"# filled: {fills}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
