# EasyFHE wheel

Place a Python 3.12/CUDA-compatible EasyFHE wheel here or set
`EASYFHE_WHEEL=/absolute/path/to/easyfhe.whl` before `tools/setup_env.sh`.
The build must implement the reusable/bounded-baby `hoisted_mac_sum` API
described in `docs/BACKEND.md`.

For a compiled checkout, set `EASYFHE_SOURCE_DIR` instead; no wheel is then
installed by the setup script.
