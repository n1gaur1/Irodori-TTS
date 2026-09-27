"""Measure InferenceRuntime speed, memory and output hashes.

Two workloads:

* ``--workload sequence`` (default) reads a list of different lines once per
  round, like a script read line by line. Every request has a new length, so
  this is the number that matters for interactive use.
* ``--workload repeat`` synthesizes the same text several times after warmup
  and reports median / p95.

Compare implementations by running the same command against two checkouts
(for example with ``PYTHONPATH`` pointing at each) and diffing the ``hash``
fields: identical hashes mean identical waveforms.

Full results, including the device name and absolute paths, go to
``artifacts/local/`` (git-ignored). Pass ``--public`` to also print a summary
without environment details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)

SEQUENCE_LINES = [
    "おはようございます。",
    "今日はどこへ行くの？",
    "ちょっと待って、忘れ物をしたみたい。",
    "駅までは歩いて十分くらいだから、まだ間に合うと思うよ。",
    "ねえ、あのお店のケーキ、覚えてる？去年の誕生日に一緒に食べたやつ。",
    "うん、もちろん。あのときは雨が降っていて、傘を一本しか持っていなかったよね。",
    "そう、それで二人ともびしょ濡れになって、店員さんにタオルを貸してもらったんだった。",
    "今度の週末、もう一度行ってみない？新しい季節のメニューが出ているらしいんだ。",
    "いいね。じゃあ土曜日の午後二時に、いつもの改札の前で待ち合わせしよう。",
    "わかった。楽しみにしてる。",
]
REPEAT_TEXTS = {
    "short": "こんにちは、今日はいい天気ですね。",
    "medium": "昨日の会議で決まったことを、もう一度まとめておきます。来週の水曜日までに資料を準備してください。",
}


def _audio_hash(audios: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for audio in audios:
        digest.update(audio.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
    }


def _run(runtime: InferenceRuntime, request: SamplingRequest) -> dict:
    device = runtime.model_device
    _sync(device)
    t0 = time.perf_counter()
    result = runtime.synthesize(request)
    _sync(device)
    wall = time.perf_counter() - t0
    audio_sec = result.audio.shape[-1] / result.sample_rate
    return {
        "wall": wall,
        "audio_sec": audio_sec,
        "rtf": wall / audio_sec if audio_sec > 0 else None,
        "stages": dict(result.stage_timings),
        "hash": _audio_hash(result.audios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--cuda-graph", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--ref-wav", default=None, help="Reference audio; omit for no_ref.")
    parser.add_argument("--workload", default="sequence", choices=["sequence", "repeat"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output", default=None, help="JSON path (default: artifacts/local/bench-<time>.json)."
    )
    parser.add_argument("--public", action="store_true", help="Print a summary without env info.")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not Path(checkpoint).is_file():
        checkpoint = download_hf_checkpoint(checkpoint)
    t0 = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint,
            model_device=args.device,
            codec_device=args.device,
            model_precision=args.precision,
            cuda_graph=args.cuda_graph,
        )
    )
    device = runtime.model_device
    _sync(device)
    load_sec = time.perf_counter() - t0

    def request(text: str) -> SamplingRequest:
        if args.ref_wav:
            return SamplingRequest(text=text, ref_wav=args.ref_wav, seed=args.seed)
        return SamplingRequest(text=text, no_ref=True, seed=args.seed)

    # Process-level warmup with a text outside both workloads.
    _run(runtime, request("準備運動です。"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    runs: list[dict] = []
    summary: dict[str, object] = {}
    if args.workload == "sequence":
        for round_index in range(args.rounds):
            total = 0.0
            for line_index, line in enumerate(SEQUENCE_LINES):
                run = _run(runtime, request(line))
                run.update(round=round_index, line=line_index)
                runs.append(run)
                total += run["wall"]
            summary[f"round{round_index}_total_sec"] = total
            print(f"round {round_index}: total {total:.2f}s", flush=True)
    else:
        for name, text in REPEAT_TEXTS.items():
            walls = []
            for index in range(args.warmup + args.repeats):
                run = _run(runtime, request(text))
                run.update(case=name, warm=index >= args.warmup)
                runs.append(run)
                if index >= args.warmup:
                    walls.append(run["wall"])
            ordered = sorted(walls)
            summary[name] = {
                "median_sec": statistics.median(walls),
                "p95_sec": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
                "min_sec": ordered[0],
                "stable_hash": len({r["hash"] for r in runs if r.get("case") == name}) == 1,
            }
            print(f"{name}: {summary[name]}", flush=True)
    summary["memory"] = _memory(device)

    graph = getattr(runtime, "_graph_forward", None)
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "irodori_tts": os.path.dirname(sys.modules["irodori_tts"].__file__),
            "checkpoint": checkpoint,
        },
        "settings": vars(args),
        "load_sec": load_sec,
        "summary": summary,
        "cuda_graph_stats": None if graph is None else vars(graph.stats),
        "reference_cache": {
            "hits": runtime.reference_cache.hits,
            "misses": runtime.reference_cache.misses,
        },
        "runs": runs,
    }
    output = Path(
        args.output
        or f"artifacts/local/bench-{args.workload}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    if args.public:
        print(json.dumps({"settings": {"workload": args.workload}, "summary": summary}, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
