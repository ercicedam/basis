# Weights

The weights file (`BASIS.weights.h5`, ~9.5 MB) isn't committed to git —
it's published as a [GitHub Release](https://github.com/ercicedam/basis/releases/tag/v0.1.0)
asset instead. Download it into this folder:

```bash
curl -L -o weights/BASIS.weights.h5 \
    https://github.com/ercicedam/basis/releases/download/v0.1.0/BASIS.weights.h5
```

> **Note:** this repository is currently private, so the plain `curl`
> command above only works for accounts with access, and only after
> authenticating (e.g. `curl` with a GitHub token, or a browser session
> that's logged in). Until the repo is made public, the reliable way to
> fetch this file is with an authenticated GitHub CLI:
> ```bash
> gh release download v0.1.0 --repo ercicedam/basis --pattern "BASIS.weights.h5" -O weights/BASIS.weights.h5
> ```
> Once the repository is public, the plain `curl` command works for anyone.

This file is loaded with `Model.load_weights(...)`, so it only contains
weight values, not the architecture — the architecture is defined in
`src/basis/model.py` and must match how the weights were trained.

Run the model with the configuration recorded for this weights file
(patch size, number of classes, number of input features, ASPP kernel
size). The released weights (from run
`3-classes_with_pan-input_danish_data_on_ME-1200_PS-512_BS-56`) use:

| Setting        | Value | CLI flag        |
|----------------|-------|------------------|
| Patch size     | 512   | `--patch-size`   |
| Classes        | 3     | `--n-classes`    |
| Input features | 1 (panchromatic) | `--n-features` |
| Kernel size    | 1     | `--kernel-size`  |

These are also the CLI defaults, so with this weights file you only need
to pass `--input`, `--weights`, and `--output-dir` (see the main README).
