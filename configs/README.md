# Frozen benchmark profiles

`models/1b.json`, `3b.json`, and `8b.json` describe the supplied Couplet
model profiles. Paths under `paths` are relative to the chosen data root;
polynomial paths are relative to the repository. `{repo}` in an environment
value is expanded by the unified launcher.

Each profile records:

- official HF repository, revision and weight-blob SHA-256/size;
- pinned mirror revision, native/physical geometry and converted paths;
- frozen polynomial table paths and fixture-generation table paths;
- Couplet environment, common arguments and world-size-specific arguments;
- the timing protocol: first Decode or same-token hot replay.

The 1B fixture trajectory uses the registered reference pack,
while runtime applies the measured RMSNorm/SiLU tables and the layer-0/1
Softmax overlay. Both sets of table paths are explicit. The 3B RMSNorm table
is selected by the model profile; its generic 1/2/4/8-card profile
retains padded query heads. The separate P3 adapter is a single-GPU option,
not the profile used by the generic multi-GPU table.

8B's profile enables FastEncode and weight eviction. Timing comparisons must
use the matching profile and measurement protocol.

The profiles select the configurations documented in the benchmark records.
CPU contracts check configuration and interfaces; GPU execution and numerical
acceptance require the corresponding GPU checks.
