# Documentation

Start with the repository README and use the model profiles in `configs/models/`.

| Guide | Purpose |
|---|---|
| [Reproduction](REPRODUCING.md) | Model preparation, workloads, GPU counts, timing and validation boundaries |
| [Backend](BACKEND.md) | Pinned EasyFHE wheel, source, compatibility and shared CUDA extensions |
| [Design](DESIGN.md) | Runtime components, layouts, KV continuity, Softmax and parallel execution |
| [Packing examples](PACKING.md) | Worked slot-layout examples; model-specific geometry comes from the profiles |

The model asset workflow is in [tools/models/README.md](../tools/models/README.md).
Measurement entry points are described in [bench/README.md](../bench/README.md).
Result summaries are available under [`bench/records/`](../bench/records/README.md).
