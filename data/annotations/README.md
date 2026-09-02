# Annotation bundle

`xcl/qwen38_adaptive_review_5s_annotations.jsonl` is the current teacher: 2,346 reviewed five-second windows from 290 recordings with 11,989 foreground events.

The other `qwen38_*` files preserve earlier annotation experiments, progress snapshots, and public decision summaries. `qwen38_tool_views/` contains 19 retained inspection images.

`xcl/XCL_train_annotations.json.gz` is the lossless compressed 205 MB source annotation index. The unpacked local copy is ignored by Git.

Paths inside the copied Qwen records are portable shard basenames or repository-relative metadata paths. Run `sha256sum -c data/annotations/SHA256SUMS` to verify the bundle.
