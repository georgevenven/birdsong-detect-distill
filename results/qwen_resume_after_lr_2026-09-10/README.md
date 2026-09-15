# Unattended queue

User-authorized on September 10: finish YOLO11l, finish the SongMAE 3e-4 learning-rate
comparison, then resume the original XC Qwen queue with 16 workers / 16,384-token slots.

Independent systemd user services, in order:

1. `birdsong-yolo11l-external-20260910.service`: external evaluation and expanded tables.
2. `birdsong-learning-rate-20260910.service`: wait, train, evaluate Powdermill, publish comparison.
3. `birdsong-qwen-after-learning-rate-20260910.service`: verify successful completion and free GPUs,
   archive the pause marker, start the Qwen server, verify health/slots, then start XC continuation.

User lingering is enabled. These processes do not depend on Codex, a terminal or an SSH connection.
Keep Lambda-Twins powered on; this queue does not promise restart recovery after a machine reboot.
Failed experiments, changed pause state, changed inputs or insufficient disk stop the handoff.
The annotation runner skips accepted windows, reuses stage checkpoints, and allows three retry passes.
At scheduling: 4,819 accepted windows, 31 remaining. The frozen queue is not expanded.

`queued.json` records the authorization snapshot; `resumed.json` records a successful handoff.
The original pause marker is retained as `pause-marker-before-resume.txt` when resumption begins.
Cancel the pending Qwen handoff with:

```bash
systemctl --user stop birdsong-qwen-after-learning-rate-20260910.service
```

Once Qwen has resumed, stopping this handoff service alone will not stop the separate server/client.
Request a graceful pause of the annotation service before stopping its server.
