# Llama benchmark results

`results.json` contains curated 1B/3B/8B measurements and per-cell status;
`results.csv` contains wide tables. Hardware: RTX PRO 6000 Blackwell Server
Edition 96 GB, 1/2/4/8 physical GPUs, 128 tokens, Prefix 0.

Values are online per-layer maximum-rank forward sums. Initialization, client
work, and separately measured transfers are excluded.

SC multi-GPU entries use the nonlinear-sharded implementation. Most
cells are one complete run. Couplet3B uses a two-run median; Couplet8B W1
uses 143.088 and 143.177 seconds, median 143.1325. Both repetitions recorded
1069 BS calls, including 64 KV-refresh BS.

Decode first calls and hot replay are separate rows. Replay fixes the query,
position and original ciphertext KV; it does not generate successive tokens.
SC8B's upload-inclusive hot-loop wall times were 42.55 / 36.98 / 32.44 / 28.42
seconds for 1 / 2 / 4 / 8 GPUs. Its forward-only rows are not complete latency.

Retained limitations: Couplet3B W8 Prefill met the speed-run error threshold 0.06,
rather than 0.02. Couplet8B W4 Decode did not pass the configured quality audit
and remains a raw performance observation. Couplet8B's speed-only status is
recorded per cell. These calibrated profiles do not establish uniform quality
on arbitrary inputs.

Per-cell status and configuration details are recorded in `results.json`.
CPU source checks do not establish GPU execution or performance acceptance.
