#!/usr/bin/env python3
import argparse
import json
import random
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from annotate_qwen import SHIFT_REVIEW, project, reconcile
from birdsong_detect_distill.qwen import (LABELS, REVIEW, SELF_REVIEW, SYSTEM, call, canonical_events, context, image, map_final,
    ownership_x, preview_image, read_tiles, split_tiles)


VARIANTS = ("direct", "reasoning", "self_review", "shifted_review", "full")


def held_out_ids(path):
    with zipfile.ZipFile(path) as archive:
        return {match.group().upper() for name in archive.namelist() if (match := re.search(r"XC\d+", name, re.IGNORECASE))}


def materialize(master, out_dir, metadata, selected):
    rows = [json.loads(line) for line in master.open()]
    for variant in VARIANTS:
        path = out_dir / f"{variant}.jsonl"
        output = [{**metadata, "variant": variant}]
        output.extend(row["variants"][variant] for row in rows
            if row.get("status") == "ok" and row["recording"] in selected)
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output))


def main():
    parser = argparse.ArgumentParser(description="Run cumulative Qwen annotation ablations on recording-matched XCL tiles.")
    parser.add_argument("--spec-dir", type=Path, default=Path("data/xcl"))
    parser.add_argument("--tiles-from", type=Path,
        default=Path("data/annotations/xcl/qwen38_agentic_annotations.jsonl"))
    parser.add_argument("--xcaj-audio", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/annotations/xcl/ablation_10"))
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--recordings", type=int, default=10)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--recording", dest="selected_recordings", action="append")
    parser.add_argument("--reasoning-budget", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.spec_dir, args.out_dir = args.spec_dir.resolve(), args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    master = args.out_dir / "passes.jsonl"
    params = json.loads((args.spec_dir / "audio_params.json").read_text())
    bins_per_second = params["sr"] / params["hop_size"]
    second, ms_per_bin = round(bins_per_second), 1000 / bins_per_second

    source = read_tiles(args.tiles_from.resolve())
    excluded = held_out_ids(args.xcaj_audio)
    available = {tile[0] for tile in source if tile[0].upper() not in excluded}
    recordings = sorted(available)
    random.Random(args.seed).shuffle(recordings)
    selected = args.selected_recordings or recordings[:args.recordings]
    if len(selected) != args.recordings:
        raise ValueError(f"requested {args.recordings} recordings, found {len(selected)}")
    if invalid := set(selected) - available:
        raise ValueError(f"unavailable or held-out recordings: {sorted(invalid)}")
    keep = set(selected)
    all_tiles = split_tiles([tile for tile in source if tile[0] in keep], 5 * second)
    selection = args.out_dir / "selection.json"
    if args.max_tiles:
        if selection.exists():
            chosen = {tuple(tile) for tile in json.loads(selection.read_text())}
        else:
            chosen = set(random.Random(args.seed).sample(all_tiles, min(args.max_tiles, len(all_tiles))))
            selection.write_text(json.dumps(sorted(chosen), separators=(",", ":")))
        all_tiles = [tile for tile in all_tiles if tile in chosen]
        if len(all_tiles) != len(chosen):
            raise ValueError("saved tile selection no longer matches the source annotations")
    done = {(row["recording"], row["owner_start"], row["owner_end"])
        for row in (json.loads(line) for line in master.open())
        if row.get("status") == "ok" and row["recording"] in keep} if master.exists() else set()
    tiles = [tile for tile in all_tiles if (tile[0], tile[4], tile[5]) not in done]
    lock = threading.Lock()

    metadata = {"type": "metadata", "schema_version": 1, "workflow": "qwen_cumulative_ablation",
        "model": "Qwen3.8-27B-Q8_0", "recordings": selected, "recording_count": len(selected),
        "tile_seconds": 5, "window_count": len(all_tiles), "seed": args.seed, "reasoning_budget": args.reasoning_budget,
        "max_tokens": args.max_tokens, "excluded_xcaj_recordings": len(excluded), "labels": LABELS, "system_prompt": SYSTEM}

    def finish(events, start, end, owner_start, owner_end):
        canonical = canonical_events(events, start, end, start, end, owner_start, owner_end)
        return canonical, map_final(canonical, start, end, owner_start, owner_end, params["mels"], ms_per_bin)

    def row(name, tile, view_start, view_end, events, quality, summary, extra=None):
        recording, shard, source_start, source_end, owner_start, owner_end = tile
        value = {"type": "annotation", "workflow": "qwen_cumulative_ablation", "variant": name, "status": "ok",
            "recording": recording, "source": {"shard": shard, "start": source_start, "end": source_end},
            "tile": {"start_timebin": view_start, "end_timebin": view_end,
                "ownership_start_timebin": owner_start, "ownership_end_timebin": owner_end,
                "onset_ms": round(owner_start * ms_per_bin, 3), "offset_ms": round(owner_end * ms_per_bin, 3),
                "ownership_onset_ms": round(owner_start * ms_per_bin, 3),
                "ownership_offset_ms": round(owner_end * ms_per_bin, 3)},
            "events": events, "window_quality": quality, "summary": summary}
        return {**value, **(extra or {})}

    def annotate(index, tile):
        recording, _, _, _, owner_start, owner_end = tile
        view_start = owner_start - (5 * second - (owner_end - owner_start)) // 2
        spec, start, end = context(args.spec_dir, tile, view_start, view_start + 5 * second)
        clean, owner = image(spec), ownership_x(owner_start, owner_end, start, end)
        instruction = f"Ownership is x={owner[0]}..{owner[1]} on the 0-1000 axis. Return only events whose midpoint lies inside it."
        seed = args.seed + index * 31

        direct = call(args, SYSTEM, instruction, [clean], seed, reasoning_budget=0)
        initial = call(args, SYSTEM, instruction, [clean], seed)
        reviewed = call(args, SELF_REVIEW, instruction + " Picture 2 shows your current boxes in red. Current event JSON: "
            + json.dumps(initial["events"], separators=(",", ":")),
            [clean, preview_image(spec, [event["bbox_2d"] for event in initial["events"]])], seed + 1)
        primary, primary_final = finish(reviewed["events"], start, end, owner_start, owner_end)

        shift = -1 if index % 2 == 0 else 1
        shifted_spec, shifted_start, shifted_end = context(args.spec_dir, tile, start + shift * second, end + shift * second)
        shifted_primary = project(primary, start, end, shifted_start, shifted_end)
        shifted_owner = ownership_x(owner_start, owner_end, shifted_start, shifted_end)
        independent = call(args, SHIFT_REVIEW,
            f"Ownership is x={max(0, shifted_owner[0])}..{min(1000, shifted_owner[1])}. Picture 2 shows the reviewed primary "
            "boxes in red. Their exact coordinates in this shifted view are: " + json.dumps(shifted_primary, separators=(",", ":")),
            [image(shifted_spec), preview_image(shifted_spec, [event["bbox_2d"] for event in shifted_primary])], seed + 2)
        reviewer = canonical_events(independent["events"], shifted_start, shifted_end, start, end, owner_start, owner_end)
        agreed = reconcile(primary, reviewer)
        shifted = primary if agreed is None else agreed
        shifted_final = map_final(shifted, start, end, owner_start, owner_end, params["mels"], ms_per_bin)

        adjudicated = agreed is None
        if adjudicated:
            proposals = json.dumps({"primary": primary, "shifted_reviewer": reviewer}, separators=(",", ":"))
            final = call(args, REVIEW, instruction + " Pictures 2 and 3 show the primary and shifted reviewer proposals. "
                "Use these exact proposal coordinates; do not expect coordinate text inside the images: " + proposals,
                [clean, preview_image(spec, [event["bbox_2d"] for event in primary]),
                    preview_image(spec, [event["bbox_2d"] for event in reviewer])], seed + 3)
            full, full_final = finish(final["events"], start, end, owner_start, owner_end)
            fallback = not full and primary and reviewer
            if fallback:
                full, full_final = primary, primary_final
            full_quality, full_summary = final["window_quality"], final["public_summary"]
        else:
            full_final, fallback = shifted_final, False
            full_quality, full_summary = independent["window_quality"], "Reviewers agreed; matched boxes were averaged."

        _, direct_final = finish(direct["events"], start, end, owner_start, owner_end)
        _, initial_final = finish(initial["events"], start, end, owner_start, owner_end)
        variants = {
            "direct": row("direct", tile, start, end, direct_final, direct["window_quality"], direct["public_summary"]),
            "reasoning": row("reasoning", tile, start, end, initial_final, initial["window_quality"], initial["public_summary"]),
            "self_review": row("self_review", tile, start, end, primary_final, reviewed["window_quality"], reviewed["public_summary"]),
            "shifted_review": row("shifted_review", tile, start, end, shifted_final, independent["window_quality"],
                "Reviewers disagreed; retained self-reviewed primary without adjudication." if adjudicated else full_summary,
                {"reviewers_agreed": not adjudicated}),
            "full": row("full", tile, start, end, full_final, full_quality, full_summary,
                {"adjudicated": adjudicated, "high_recall_fallback": bool(fallback)}),
        }
        return {"status": "ok", "recording": recording, "owner_start": owner_start, "owner_end": owner_end,
            "variants": variants, "passes": {"direct": direct, "primary": initial, "self_review": reviewed,
                "independent_shifted_review": independent}}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(annotate, index, tile): tile for index, tile in enumerate(tiles)}
        for index, future in enumerate(as_completed(futures), 1):
            tile = futures[future]
            try:
                output = future.result()
            except Exception as error:
                output = {"status": "error", "recording": tile[0], "owner_start": tile[4], "owner_end": tile[5],
                    "error": str(error)}
            with lock, master.open("a") as file:
                file.write(json.dumps(output, separators=(",", ":")) + "\n")
            print(f"{index}/{len(tiles)} {tile[0]}: {output['status']}", flush=True)

    materialize(master, args.out_dir, metadata, keep)
    completed = {(row["recording"], row["owner_start"], row["owner_end"])
        for row in (json.loads(line) for line in master.open())
        if row.get("status") == "ok" and row["recording"] in keep}
    report = {"recordings": selected, "new_tiles": len(tiles), "completed_tiles": len(completed),
        "expected_tiles": len(all_tiles), "output": str(args.out_dir)}
    print(json.dumps(report, indent=2))
    if len(completed) != len(all_tiles):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
