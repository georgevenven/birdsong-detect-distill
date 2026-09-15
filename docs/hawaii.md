# Hawaii evaluation data

Source: Navine et al. (2022), [Zenodo 7078499](https://zenodo.org/records/7078499),
the Hawaii collection evaluated by [BirdBox](https://arxiv.org/html/2606.10407v1).
It provides time **and frequency** boxes, not only clip tags or temporal events.

## Original release: downloaded and verified

Zenodo initially returned gateway/access errors on 2026-09-07, then recovered. The original
release is now downloaded and extracted, with all five published MD5 checksums verified:

- `data/hawaii/zenodo/audio/`: 635 original mono 32-kHz FLAC recordings, 50.883 hours.
- `data/hawaii/zenodo/raw/annotations.csv`: 59,583 time–frequency boxes for 27 eBird species.
- `data/hawaii/zenodo/raw/`: original 5,770,895,017-byte ZIP, species/location tables and description PDF.
- `data/hawaii/zenodo/manifest.json`: file checksums, individual audio SHA-256 hashes,
  duration inventory and annotation warnings. Large data files are ignored by Git.

The [BirdSet-maintained copy](https://huggingface.co/datasets/DBD-research-group/BirdSet/tree/806ed2cda4ddcbe6efa194ccafff930aa0e557ce/UHH)
remains separately under `data/hawaii/birdset/` as a fallback. It is a lossy OGG/Vorbis
derivative; use the original FLAC for evaluation. Its annotation values match the original
CSV exactly after normalizing filename extensions. No XC training subset was downloaded.

The downloader verifies published checksums, resumes partial transfers, preserves existing
files and separates the two distributions. To reproduce the original download on another machine:

```bash
.venv/bin/python scripts/download_hawaii.py --source zenodo --connections 4
```

## Evaluation caveats

The original annotations contain **196 zero-duration boxes**; these are not mirror damage.
They remain unchanged in the source CSV. The 2026-09-08 external-evaluation policy gives them
zero reference area, before rasterization rounding, leaving 59,387 positive-area rectangles.
`src/birdsong_detect_distill/hawaii.py` enforces this policy and verifies the original inventory.
The original download manifest remains untouched: its `evaluation_run` and readiness flags
describe the state at download time. Current evaluation provenance lives in
`results/competitors_external_2026-09-08/`, not in that immutable source manifest.

Keep the original 59,583 labels as the source. BirdBox's 81,691 count comes after clipping
frequency to 500–12,000 Hz and splitting/duplicating annotations across overlapping windows;
it is not a different set of human annotations.
In the raw CSV, 4,890 distinct boxes cross a frequency boundary: 3,130 below 500 Hz and
1,894 above 12,000 Hz, with 134 crossing both. Thus 5,024 counts boundary crossings,
not distinct boxes. No annotation lies entirely outside this band.

Reference boxes may merge consecutive calls separated by less than 0.5 s; faint or
unidentifiable vocalizations were intentionally omitted. Those omissions matter for
class-agnostic detection. Check pretraining exposure before calling this collection unseen.
BirdSet metadata identifies the soundscape license as CC BY 4.0.
