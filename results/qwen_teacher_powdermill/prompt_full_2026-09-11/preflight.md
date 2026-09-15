# Prelaunch verification

- Regenerated both image variants for every one of the 250 pilot windows:
  all 500 PNG SHA-256 hashes exactly match the original files.
- Constructed all five request payloads for every pilot window using the full-run
  code: all 1,250 request SHA-256 hashes exactly match the original requests.
- Reused event lists and raw annotations exactly match their pilot sources.
  Review parent hashes point to the corresponding imported full-run records.
- Full inventory: 4,620 unique full windows and unique seeds; 250 reused windows,
  4,370 new windows. All 77 WAV files were independently checked as 300 seconds.
- Source, font, code, reference, pilot annotation, and image hashes frozen in
  `manifest.json`. Python syntax and both user service definitions validated.
- No inference calls were made during this verification. The original pilot
  annotations and source files were not modified.

Local host: Lambda-Twins, two idle RTX 4090s before launch. The separate original
XC service remains inactive and its pause marker is untouched. SSH discovery
identified `lamda-vector.uoregon.edu` at `163.41.128.63`, but the current account
could not authenticate; its GPU hardware has not been verified or used.
