# Initial sanity check

The first batch was gracefully paused after an unusually large number of empty
responses. In-flight requests drained: 89 formal outputs were saved, of which
22 contained detections (7/45 plain; 15/44 axes). The output schema therefore
does permit nonempty event lists. All 89 outputs remain in the formal study.

Five diagnostic calls used the first selected window, Recording_1_Segment_26,
235–240 seconds. The unchanged prompt/schema and an added format example both
returned empty lists. An unconstrained JSON-mode variant returned two boxes;
the legacy prompt/schema returned three on both cropped and padded images.
These are wiring diagnostics, not a matched performance comparison, and do
not establish a parser defect. The legacy padded call uses whole-image
coordinates and is not scored. Diagnostic variants are never pooled with the
formal conditions and are excluded from their timing and metrics.

No frozen input, prompt, schema, sample, or selection rule was changed following
this inspection. The formal run resumes from its 89 saved outputs. In
particular, this evaluation-window inspection is not used to optimize a prompt.
