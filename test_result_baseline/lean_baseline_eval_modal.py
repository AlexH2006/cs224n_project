"""
Lean Baseline Evaluation Pipeline (Modal + Kimina)

What this file does (as requested):

Function 1 (generate):
    - Takes pass@k and n_problems
    - If n_problems >= 244: uses all test problems in order. Else: random sample (distinct, seed)
    - For each selected problem, generates k independent proof attempts
    - Runs generation in parallel on Modal (each attempt is a separate remote call)
    - Writes outputs locally to a JSONL file (and also optionally to /output volume if enabled)

Function 2 (verify):
    - Reads the JSONL outputs from Function 1
    - For each problem, verifies attempts serially using Kimina (stop early on first success)
    - Computes pass@k accuracy over the selected problems
    - Prints per-problem results + final summary

How to run (from your repo root):
    modal run baseline/lean_baseline_eval_modal.py --n-problems 100 --pass-k 4

Or from inside baseline/:
    cd baseline
    modal run lean_baseline_eval_modal.py --n-problems 100 --pass-k 4

Notes:
    - This uses Goedel-Prover-V2-8B by default.
    - Dataset defaults to HaimingW/minif2f-lean4 (test split, paper-style MiniF2F).
    - Kimina verification is done in series as requested.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# Verification robustness: timeout (seconds), retries on server error/timeout
VERIFY_TIMEOUT_S = 30
VERIFY_RETRIES = 3
VERIFY_RETRY_WAIT_S = 3

# -----------------------------------------------------------------------------
# Modal setup
# -----------------------------------------------------------------------------

try:
    import modal

    app = modal.App("lean-baseline-eval-kimina")

    # Optional persistent volumes (helpful but not required for "local writes")
    hf_cache_volume = modal.Volume.from_name("baseline-hf-cache", create_if_missing=True)
    output_volume = modal.Volume.from_name("baseline-output", create_if_missing=True)

    # Inference image (vLLM + transformers + datasets)
    inference_image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "vllm>=0.6.0",
            "transformers==4.53.2",
            "accelerate==1.9.0",
            "datasets",
            "sentencepiece",
            "protobuf",
            "httpx",
        )
    )

    # Kimina Lean Server image
    kimina_image = modal.Image.from_registry(
        "projectnumina/kimina-lean-server:2.0.0",
    ).pip_install("httpx")

except ImportError:
    modal = None
    app = None


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # Model
    model_name: str = "Goedel-LM/Goedel-Prover-V2-8B"

    # Dataset (official MiniF2F Lean4 for paper-style eval)
    dataset_name: str = "HaimingW/minif2f-lean4"
    dataset_subset: Optional[str] = None
    dataset_split: str = "test"
    
    seed: int = 42

    # Field mapping
    theorem_fields: list = field(default_factory=lambda: [
        "lean4_code", "formal_statement", "lean4_statement",
        "statement", "code", "theorem", "problem_statement",
    ])
    header_fields: list = field(default_factory=lambda: [
        "header", "imports", "preamble", "prefix",
    ])
    id_fields: list = field(default_factory=lambda: [
        "problem_id", "name", "id", "idx", "index",
    ])
    # Natural-language problem text for minif2f paper prompt (# Problem: ...)
    problem_fields: list = field(default_factory=lambda: [
        "statement", "problem", "informal_statement", "question",
    ])

    # Fallback header
    default_header: str = """import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat
"""

    # Generation (minif2f paper: temperature=0.6, top_p=0.95, max_tokens=8192)
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    stop_tokens: list = field(default_factory=lambda: [
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
        "</s>", "<|end|>", "[/INST]", "<|eot_id|>",
    ])

    # Output directory (local)
    local_results_dir: str = "results"


def _get_field(data: dict, field_names: list, default: str = "") -> str:
    for field_name in field_names:
        if field_name in data and data[field_name]:
            value = data[field_name]
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return str(value[0])
    return default


# -----------------------------------------------------------------------------
# Kimina verification service (Modal)
# -----------------------------------------------------------------------------

if modal is not None:

    @app.cls(
        image=kimina_image,
        cpu=8,
        memory=16384,
        timeout=600,
        scaledown_window=300,
    )
    @modal.concurrent(max_inputs=100)
    class KiminaLeanServer:
        """Runs the Kimina Lean Server. Hard timeout on verify + poisoned-worker reset."""

        @modal.enter()
        def start_server(self):
            import subprocess
            import httpx

            self._reset_lock = Lock()

            env = {
                **os.environ,
                "LEAN_SERVER_HOST": "0.0.0.0",
                "LEAN_SERVER_PORT": "8000",
                "LEAN_SERVER_MAX_REPLS": "7",
                "LEAN_SERVER_LOG_LEVEL": "INFO",
            }

            self.server_proc = subprocess.Popen(
                ["python", "-m", "server"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            print("Starting Kimina Lean Server...")

            max_wait = 60
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    with httpx.Client(timeout=5.0) as client:
                        resp = client.post(
                            "http://localhost:8000/verify",
                            json={
                                "codes": [{"custom_id": "health", "proof": "example : True := trivial"}],
                                "infotree_type": "original",
                            },
                        )
                        if resp.status_code == 200:
                            print(f"Kimina ready after {time.time() - start_time:.1f}s")
                            return
                except Exception:
                    pass
                time.sleep(2)

            print(f"Warning: Kimina may not be ready after {max_wait}s")

        def _reset_server(self) -> bool:
            """Restart Kimina subprocess. Concurrency-safe. Returns True if success."""
            import subprocess
            import httpx

            with self._reset_lock:
                try:
                    if self.server_proc and self.server_proc.poll() is None:
                        self.server_proc.terminate()
                        try:
                            self.server_proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            self.server_proc.kill()
                            self.server_proc.wait()

                    env = {
                        **os.environ,
                        "LEAN_SERVER_HOST": "0.0.0.0",
                        "LEAN_SERVER_PORT": "8000",
                        "LEAN_SERVER_MAX_REPLS": "7",
                        "LEAN_SERVER_LOG_LEVEL": "INFO",
                    }
                    self.server_proc = subprocess.Popen(
                        ["python", "-m", "server"],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                    for _ in range(30):
                        time.sleep(2)
                        try:
                            with httpx.Client(timeout=5.0) as client:
                                resp = client.post(
                                    "http://localhost:8000/verify",
                                    json={
                                        "codes": [{"custom_id": "health", "proof": "example : True := trivial"}],
                                        "infotree_type": "original",
                                    },
                                )
                                if resp.status_code == 200:
                                    print("Kimina reset complete")
                                    return True
                        except Exception:
                            pass
                    print("Kimina reset: health check failed")
                    return False
                except Exception as e:
                    print(f"Kimina reset error: {e}")
                    return False

        @modal.method()
        def verify(self, lean_code: str, custom_id: str = "1") -> dict:
            import httpx

            try:
                with httpx.Client(timeout=float(VERIFY_TIMEOUT_S)) as client:
                    resp = client.post(
                        "http://localhost:8000/verify",
                        json={
                            "codes": [{"custom_id": custom_id, "proof": lean_code}],
                            "infotree_type": "original",
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TimeoutException as e:
                print(f"TIMEOUT: resetting Kimina server (custom_id={custom_id})")
                self._reset_server()
                return {
                    "error": f"Verification timed out after {VERIFY_TIMEOUT_S}s",
                    "is_server_error": True,
                    "is_timeout": True,
                }
            except httpx.ConnectError as e:
                print(f"CONN_REFUSED: resetting Kimina server (custom_id={custom_id})")
                self._reset_server()
                return {
                    "error": f"Connection refused: {e}",
                    "is_server_error": True,
                    "is_timeout": False,
                }
            except Exception as e:
                err_str = str(e)
                if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                    print(f"TIMEOUT: resetting Kimina server (custom_id={custom_id})")
                    self._reset_server()
                    return {
                        "error": f"Verification timed out: {err_str}",
                        "is_server_error": True,
                        "is_timeout": True,
                    }
                return {"error": err_str, "is_server_error": True, "is_timeout": False}


    @app.cls(image=inference_image, timeout=300, scaledown_window=300)
    class LeanVerifier:
        """Thin wrapper that normalizes Kimina results."""

        @modal.method()
        def verify(self, lean_code: str, custom_id: str = "1") -> dict:
            server = KiminaLeanServer()
            result = server.verify.remote(lean_code, custom_id=custom_id)

            if "error" in result:
                return {
                    "success": False,
                    "complete": False,
                    "has_sorry": "sorry" in lean_code.lower(),
                    "feedback": f"Kimina server error: {result['error']}",
                    "errors": [result["error"]],
                    "messages": [],
                    "sorries": [],
                    "source": "kimina",
                    "is_server_error": True,
                    "is_timeout": result.get("is_timeout", False),
                }

            if "results" in result and result["results"]:
                r = result["results"][0]
                messages = r.get("messages", []) or []
                sorries = r.get("sorries", []) or []
                status = r.get("status", "")

                errors = []
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("severity") == "error":
                        errors.append(msg.get("data", str(msg)))

                has_error = bool(errors) or status == "error"
                has_sorry = bool(sorries) or ("sorry" in lean_code.lower())

                return {
                    "success": not has_error,
                    "complete": (not has_error) and (not has_sorry),
                    "has_sorry": has_sorry,
                    "feedback": "\n".join(errors) if errors else "",
                    "errors": errors,
                    "messages": [str(m) for m in messages],
                    "sorries": [str(s) for s in sorries],
                    "source": "kimina",
                    "is_server_error": False,
                }

            return {
                "success": False,
                "complete": False,
                "has_sorry": "sorry" in lean_code.lower(),
                "feedback": "Unexpected response format from Kimina",
                "errors": ["Unexpected response format"],
                "messages": [],
                "sorries": [],
                "source": "kimina",
                "is_server_error": True,
                "is_timeout": False,
            }


# -----------------------------------------------------------------------------
# Proof generation (Modal)
# -----------------------------------------------------------------------------

def _strip_special_tokens(text: str) -> str:
    if not text:
        return text
    for tok in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(tok):
            text = text[: -len(tok)].rstrip()
    start_marker = "<|im_start|>assistant"
    if text.startswith(start_marker):
        text = text[len(start_marker):].lstrip("\n").lstrip()
    elif text.startswith("<|im_start|>"):
        text = text[len("<|im_start|>"):].lstrip("\n").lstrip()
    return text.strip()


def _extract_tactics_from_code_block(block: str) -> str:
    """
    If the model outputs a full theorem/lemma/example with `:= by`,
    strip the declaration and return only the tactic body.
    If it outputs raw tactics, return them as-is.
    """
    lines = block.split("\n")
    result_lines: list[str] = []
    in_tactic_body = False

    for line in lines:
        stripped = line.rstrip("\n")
        s = stripped.strip()

        if not s:
            if in_tactic_body:
                result_lines.append("")
            continue

        if s.startswith(("import ", "open ", "set_option ")):
            result_lines.append(s)
            continue

        if not in_tactic_body:
            if s.startswith(("theorem ", "lemma ", "example ")):
                if ":= by" in s:
                    after_by = s.split(":= by", 1)[1].strip()
                    if after_by:
                        result_lines.append(after_by)
                    in_tactic_body = True
                continue

            if s.endswith(":= by") or s == ":= by" or s == "by":
                in_tactic_body = True
                continue

            # Heuristic: no header, treat as tactics
            in_tactic_body = True

        result_lines.append(s)

    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    return "\n".join(result_lines).strip()


def extract_proof_tactics(output: str) -> str:
    """
    Extract tactics from model output.
    Priority:
      1) best ```lean``` / ```lean4``` block anywhere (fewest 'sorry', then longest)
      2) if no code block: try content after last ':= by'
      3) else 'sorry'
    """
    output = (output or "").strip()
    tactics: Optional[str] = None

    code_pattern = r"```(?:lean4?|lean|tactics)?\n?(.*?)```"
    matches = re.findall(code_pattern, output, re.DOTALL)
    if matches:
        best, best_score = None, None
        for m in matches:
            extracted = _extract_tactics_from_code_block(m)
            if not extracted or extracted.lower() in ("sorry", "by"):
                continue
            sorry_count = extracted.lower().count("sorry")
            length = len(extracted)
            score = (-sorry_count, length)
            if best_score is None or score > best_score:
                best, best_score = extracted, score
        if best:
            tactics = best

    if not tactics and ":= by" in output:
        idx = output.rfind(":= by")
        after = output[idx + len(":= by"):].strip()
        if "```" in after:
            after = after.split("```", 1)[0]
        extracted = _extract_tactics_from_code_block(after)
        if extracted and extracted.lower() not in ("sorry", "by"):
            tactics = extracted

    if tactics:
        t = tactics.strip()
        if t in ("sorry", "by", "by sorry") or len(t) < 3:
            return "sorry"
        return tactics.strip()

    return "sorry"


def create_full_lean_code(cfg: EvalConfig, theorem_code: str, proof_tactics: str, header: str = "") -> str:
    """
    Build the exact Lean file we will verify:
      - If dataset provides `header`, use it.
      - Else if theorem_code already starts with `import`, use theorem_code as-is.
      - Else prepend cfg.default_header.
    Then replace the last `sorry` in that full file with proof_tactics.
    """

    # 1) Construct the full file (same policy as _build_prompt)
    if header.strip():
        full_file = f"{header.strip()}\n\n{theorem_code}"
    else:
        full_file = (
            theorem_code
            if theorem_code.lstrip().startswith("import ")
            else f"{cfg.default_header.strip()}\n\n{theorem_code}"
        )

    # 2) Clean tactics (strip any header-ish lines)
    clean_lines: List[str] = []
    for line in (proof_tactics or "").split("\n"):
        s = line.strip()
        if s.startswith(("import ", "open ", "set_option ", "namespace ", "section ")):
            continue
        clean_lines.append(line.rstrip("\n"))

    tactics_clean = "\n".join(clean_lines).strip() or "sorry"
    indented = "\n  ".join(tactics_clean.split("\n"))

    # 3) Replace last sorry in the *full file*
    tc = full_file
    if ":= by sorry" in tc:
        tc = tc.replace(":= by sorry", f":= by\n  {indented}")
    elif ":= by\n  sorry" in tc:
        tc = tc.replace(":= by\n  sorry", f":= by\n  {indented}")
    else:
        last = tc.rfind("sorry")
        if last != -1:
            tc = tc[:last] + indented + tc[last + len("sorry"):]
        else:
            tc = tc + "\n\n" + f"by\n  {indented}\n"

    return f"{tc}\n"


if modal is not None:

    @app.cls(
        image=inference_image,
        gpu="A100-80GB",
        startup_timeout=900,
        timeout=1200,
        scaledown_window=300,
        volumes={"/cache": hf_cache_volume},
        secrets=[modal.Secret.from_name("huggingface")],
    )
    class ProofGenerator:
        """
        Proof generation worker (vLLM).
        Each call generates exactly one attempt for exactly one problem index.
        """

        @modal.enter()
        def setup(self):
            os.environ["HF_HOME"] = "/cache"
            if os.environ.get("HF_TOKEN"):
                os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

            from vllm import LLM

            # Fast startup for short-ish runs
            os.environ["VLLM_USE_V1"] = "0"
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            model_name = os.environ.get("EVAL_MODEL_NAME", EvalConfig().model_name)

            self.llm = LLM(
                model=model_name,
                dtype="bfloat16",
                trust_remote_code=True,
                download_dir="/cache",
                gpu_memory_utilization=0.85,
                max_model_len=8192,
                max_num_seqs=1,
                enforce_eager=True,
            )

            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"[ProofGenerator] Loaded model: {model_name}")
            print(f"[ProofGenerator] Tokenizer name_or_path: {getattr(self.tokenizer, 'name_or_path', None)}")

        def _build_prompt(self, cfg: EvalConfig, header: str, theorem_code: str, problem_text: str = "") -> str:
            if header.strip():
                full_stmt = f"{header.strip()}\n\n{theorem_code}"
            else:
                full_stmt = theorem_code if theorem_code.lstrip().startswith("import ") else f"{cfg.default_header.strip()}\n\n{theorem_code}"
            
            prompt = "Think about and solve the following problem step by step in Lean 4."
            prompt += f"\n# Problem:{problem_text.strip()}"
            prompt += f"\n# Formal statement:\n```lean4\n{full_stmt}\n```\n"
            messages = [
                {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        @modal.method()
        def generate_one(
            self,
            cfg_dict: dict,
            problem: dict,
            problem_idx: int,
            attempt: int,
        ) -> dict:
            from vllm import SamplingParams

            cfg = EvalConfig()
            cfg.__dict__.update(cfg_dict)
            theorem_code = _get_field(problem, cfg.theorem_fields)
            header = _get_field(problem, cfg.header_fields)
            problem_text = _get_field(problem, cfg.problem_fields)
            pid = _get_field(problem, cfg.id_fields, f"problem_{problem_idx}")

            prompt = self._build_prompt(cfg, header, theorem_code, problem_text=problem_text)

            base = f"{cfg.dataset_name}:{pid}:{attempt}:{cfg.seed}"
            digest = hashlib.sha256(base.encode("utf-8")).digest()
            seed_val = int.from_bytes(digest[:4], "big")  # 32-bit stable seed

            sp = SamplingParams(
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens,
                stop=cfg.stop_tokens,
                seed=seed_val,
            )

            out = self.llm.generate([prompt], sp)[0].outputs[0].text
            out = _strip_special_tokens(out)

            tactics = extract_proof_tactics(out)
            full_code = create_full_lean_code(cfg, theorem_code, tactics, header=header)

            return {
                "problem_idx": problem_idx,
                "problem_id": pid,
                "attempt": attempt,
                "full_code": full_code,
                "raw_output": out,
            }


# -----------------------------------------------------------------------------
# Function 1: generate proofs in parallel and write to local output
# -----------------------------------------------------------------------------

def generate_proofs_parallel(
    cfg: EvalConfig,
    pass_k: int,
    n_problems: int,
    output_path: Path,
    full_outputs_path: Optional[Path] = None,
) -> Path:
    """
    Generates k attempts for each of n_problems from the dataset.
    If n_problems >= 244 (or >= dataset size), uses all test problems in order.
    Otherwise randomly samples n_problems (distinct). Runs generation in parallel (Modal map).
    Writes JSONL records to output_path.
    Returns output_path.
    """
    if modal is None:
        raise RuntimeError("Modal is not installed. `pip install modal` then retry.")

    from datasets import load_dataset

    # Load dataset locally (in the driver)
    if cfg.dataset_subset:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_subset, split=cfg.dataset_split)
    else:
        ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    ds_len = len(ds)
    print(f"[config] dataset={cfg.dataset_name} split={cfg.dataset_split} len={ds_len} n_problems={n_problems} seed={cfg.seed} pass_k={pass_k} verify_timeout_s={VERIFY_TIMEOUT_S}")
    if ds_len != 244:
        print(f"[WARNING] Test split has {ds_len} problems (expected 244). Dataset: {cfg.dataset_name}, split: {cfg.dataset_split}")

    n_sample = min(n_problems, ds_len)
    if n_problems >= 244 or n_problems >= ds_len:
        selected = list(range(ds_len))
        print(f"[generate] Using all {ds_len} test problems in order")
    else:
        random.seed(cfg.seed)
        selected = sorted(random.sample(range(ds_len), n_sample))
    if not selected:
        raise ValueError(f"No problems selected. n_problems={n_problems}, dataset_size={ds_len}")

    assert len(selected) == len(set(selected)), "selected indices must be distinct"
    print(f"[generate] selected indices: n={len(selected)}, first5={selected[:5]}, last5={selected[-5:]}")

    # Create all (problem_idx, attempt) pairs
    jobs: List[Tuple[int, int]] = []
    for pidx in selected:
        for a in range(pass_k):
            jobs.append((pidx, a))

    # Materialize problem dicts for each job (so generator has what it needs)
    # (Yes, this duplicates some data; it's the simplest reliable way.)
    problems: List[dict] = [dict(ds[pidx]) for (pidx, _) in jobs]
    pidxs: List[int] = [pidx for (pidx, _) in jobs]
    attempts: List[int] = [a for (_, a) in jobs]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    gen = ProofGenerator()
    cfg_dict = cfg.__dict__.copy()

    if n_problems < 244 and n_problems < ds_len:
        print(f"[generate] Selected {len(selected)} problems (random sample, seed={cfg.seed})")
    print(f"[generate] Generating pass@{pass_k}: total attempts = {len(jobs)}")
    print(f"[generate] Writing proofs JSONL to: {output_path}")
    if full_outputs_path:
        print(f"[generate] Writing full outputs to: {full_outputs_path}")

    # Parallel generation: one remote call per attempt
    results_iter = gen.generate_one.map(
        [cfg_dict] * len(jobs),
        problems,
        pidxs,
        attempts,
        order_outputs=False,  # allow Modal to stream results as they finish
    )

    # Stream to disk as results arrive (proofs only + optional full outputs)
    if full_outputs_path:
        full_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    full_file = full_outputs_path.open("w", encoding="utf-8") if full_outputs_path else None
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for r in results_iter:
                proofs_rec = {
                    "problem_idx": r["problem_idx"],
                    "problem_id": r["problem_id"],
                    "attempt": r["attempt"],
                    "full_code": r["full_code"],
                }
                f.write(json.dumps(proofs_rec) + "\n")
                if full_file:
                    full_file.write(json.dumps(r) + "\n")
                count += 1
                if count % 10 == 0:
                    print(f"[generate] wrote {count}/{len(jobs)} attempts...")
    finally:
        if full_file:
            full_file.close()

    print(f"[generate] done. wrote {count} attempts.")
    return output_path


# -----------------------------------------------------------------------------
# Function 2: verify proofs in series and report pass@k accuracy
# -----------------------------------------------------------------------------

def verify_proofs_serial(
    output_path: Path,
) -> dict:
    """
    Reads JSONL outputs. For each problem, verifies attempts serially (Kimina).
    Stops early per problem on first complete success.
    Prints per-problem result and final pass@k accuracy.
    Returns a summary dict.
    """
    if modal is None:
        raise RuntimeError("Modal is not installed. `pip install modal` then retry.")

    if not output_path.exists():
        raise FileNotFoundError(str(output_path))

    # Load all attempts from JSONL
    by_problem: Dict[int, List[dict]] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pidx = int(rec["problem_idx"])
            by_problem.setdefault(pidx, []).append(rec)

    # Deterministic attempt order
    for pidx in by_problem:
        by_problem[pidx].sort(key=lambda r: int(r.get("attempt", 0)))

    verifier = LeanVerifier()

    n_total = len(by_problem)
    n_success = 0
    per_problem_results: List[dict] = []

    print(f"[verify] loaded {n_total} problems from: {output_path}")
    print("[verify] Verifying in series with Kimina (stop on first success per problem)\n")

    for i, (pidx, attempts) in enumerate(sorted(by_problem.items(), key=lambda x: x[0]), start=1):
        pid = attempts[0].get("problem_id", f"problem_{pidx}")
        success = False
        chosen_attempt = None
        chosen_verification = None

        for rec in attempts:
            full_code = rec["full_code"]
            attempt_id = rec.get("attempt", 0)
            v = None

            for retry in range(VERIFY_RETRIES):
                print(f"[verify] start pidx={pidx} attempt={attempt_id} retry={retry}")
                v = verifier.verify.remote(full_code, custom_id=f"{pidx}:{attempt_id}")
                print(
                    f"[verify] done  pidx={pidx} attempt={attempt_id} retry={retry} "
                    f"success={v.get('success', False)} complete={v.get('complete', False)} "
                    f"has_sorry={v.get('has_sorry', False)} is_server_error={v.get('is_server_error', False)} "
                    f"is_timeout={v.get('is_timeout', False)}"
                )

                needs_retry = v.get("is_server_error", False) or v.get("is_timeout", False)
                if not needs_retry:
                    break
                if retry < VERIFY_RETRIES - 1:
                    print(f"[verify] server error/timeout, retrying in {VERIFY_RETRY_WAIT_S}s...")
                    time.sleep(VERIFY_RETRY_WAIT_S)

            chosen_attempt = attempt_id
            chosen_verification = v if v is not None else {"success": False, "complete": False, "feedback": "No verification result"}

            if v and v.get("success") and v.get("complete") and not v.get("has_sorry", False):
                success = True
                break

        if success:
            n_success += 1

        per_problem_results.append({
            "problem_idx": pidx,
            "problem_id": pid,
            "success": success,
            "chosen_attempt": chosen_attempt,
            "verification": chosen_verification,
        })

        status = "OK" if success else "FAIL"
        print(f"[{i:>3}/{n_total}] {status}  problem_idx={pidx}  id={pid}  chosen_attempt={chosen_attempt}")
        if not success and chosen_verification:
            fb = (chosen_verification.get("feedback") or "").strip()
            if fb:
                print(f"      feedback: {fb.splitlines()[0][:200]}")

    acc = (n_success / n_total) if n_total > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"pass@k accuracy: {n_success}/{n_total} = {acc:.3f}")
    print("=" * 60)

    summary = {
        "output_path": str(output_path),
        "n_problems": n_total,
        "n_success": n_success,
        "accuracy": acc,
        "results": per_problem_results,
        "ts": datetime.now().isoformat(),
    }
    return summary


# -----------------------------------------------------------------------------
# Modal CLI entrypoint
# -----------------------------------------------------------------------------

if modal is not None:

    @app.local_entrypoint()
    def main(
        n_problems: int = 244,
        pass_k: int = 1,
        dataset: str = "HaimingW/minif2f-lean4",
        split: str = "test",
        model: str = "Goedel-LM/Goedel-Prover-V2-8B",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 8192,
        out_dir: str = "results",
        out_name: str = "",
        verify_only: bool = False,
        generate_only: bool = False,
        seed: int = 42,
    ):
        """
        Default behavior: generate (parallel) -> verify (serial) and print summary.

        Examples:
            modal run baseline/lean_baseline_eval_modal.py --n-problems 100 --pass-k 4
            modal run baseline/lean_baseline_eval_modal.py --verify-only True --out-name results/run_.../proofs.jsonl
        """
        cfg = EvalConfig(
            model_name=model,
            dataset_name=dataset,
            dataset_split=split,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            local_results_dir=out_dir,
        )

        os.environ["EVAL_MODEL_NAME"] = cfg.model_name
        print(f"[config] model: {cfg.model_name}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = cfg.model_name.replace("/", "-")
        run_dir = Path(cfg.local_results_dir) / f"run_{model_safe}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # If user passes an explicit out_name, treat it as a path
        if out_name.strip():
            proofs_path = Path(out_name).expanduser()
        else:
            proofs_path = run_dir / "proofs.jsonl"

        summary_path = run_dir / "summary.json"
        full_outputs_path = proofs_path.parent / "full_outputs.jsonl"

        # If verify_only, we assume out_name points at an existing JSONL file
        if verify_only:
            summary = verify_proofs_serial(proofs_path)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[verify] wrote summary to: {summary_path}")
            return

        # Otherwise generate unless generate_only=False?
        if not verify_only:
            generate_proofs_parallel(cfg, pass_k=pass_k, n_problems=n_problems, output_path=proofs_path, full_outputs_path=full_outputs_path)
        if generate_only:
            print("[generate-only] Done.")
            return

        # Verify unless generate_only
        summary = verify_proofs_serial(proofs_path)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] wrote summary to: {summary_path}")

else:
    # Non-modal fallback
    def main():
        raise RuntimeError("This script requires Modal. Install with: pip install modal")


if __name__ == "__main__":
    main()
