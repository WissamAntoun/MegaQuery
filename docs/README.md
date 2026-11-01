# Documentation

For an existing MegaQuery-680M download, start with [the data guide](data-guide.md).
It describes both language configurations, all columns, the pinned document order,
and how to recover full source text without rerunning inference.

To generate a new dataset, read these guides in order:

1. [Environment and hardware](environment.md): CPU/GPU environments and required APIs.
2. [Pipeline](pipeline.md): source preparation and commands for every stage.
3. [Generation](generation.md): personas, configuration sampling, prompts, outputs.
4. [Mining](mining.md): compaction, embedding, HNSW, candidate retrieval, scoring.
5. [Export](export.md): remapping, cleanup, language-local combined layout.
6. [Reproduction limits](reproduction.md): known recipe versus unavailable provenance.

[Attribution](attribution.md) distinguishes code, data, models, and dependency licenses.
The stage guides adapt the existing generation, embedding/mining, and release
READMEs to the English/French release and its published schema.
