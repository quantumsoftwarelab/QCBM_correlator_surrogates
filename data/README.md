# Reference inputs

`dmrg/` contains six original amplitude exports, copied from the local DMRG project. Files retain their original names and values, with lossless gzip compression. Skip the CSV header and square the absolute amplitudes to obtain probabilities. Rows are computational-basis bitstrings in binary order, with the last site varying fastest.

| Original filename | Use |
| --- | --- |
| `sq_6_1_amplitudes_ising_2d.csv` | Six-site 1D TFIM; Figures 4, 8–12 |
| `sq_6_1_amplitudes_heisenberg_2d.csv` | Six-site alternating Heisenberg; Figures 4, 9 |
| `sq_6_1_amplitudes_2d.csv` | Six-site 1D Haldane; Figures 4, 9 |
| `sq_2_3_amplitudes_2d.csv` | Six-site rectangular Haldane; Figures 4, 8, 9, 12 |
| `sq_12_1_amplitudes_ising_2d.csv` | Twelve-site TFIM; batch experiment |
| `sq_20_1_amplitudes_ising_2d.csv` | Twenty-site TFIM; batch experiment |

The historical `_2d` suffix does not reliably denote a two-dimensional model. The six-site interpretation and coupling values come from the paper and notebook target mapping: TFIM `J=0.7, h=0.33`; Heisenberg `J_even=1.4, J_odd=0.6`; Haldane `J=1, h_1=0.7, h_2=0.33`. Do not infer unrecorded DMRG solver settings or a generation seed from these filenames. The newer DMRG project defaults were not substituted for the original exports.

`n_6_bio.npy` is the already-binarized six-gene probability distribution used by the notebooks. It is the input needed for the published plots, not the original scRNA-seq matrix or a new preprocessing pipeline.

The remaining directories hold selected original experiment outputs. `G_info`/`G_info_rfc` retain deployed probabilities; `TN_data` holds training curves; `TN_k_data` holds correlator records used in the TN discrepancy analysis; `diff_losses` holds per-run histories; `results` holds averaged batch KL records. No trained environments, caches, unrelated data, or full parameter/checkpoint trees are included.

Historical pickle dictionaries were converted to safe NPZ files. `structure` is a JSON tree mapping original dictionary keys to named numeric arrays. `born.common.load_results` reads them with `allow_pickle=False`. Lists of homogeneous numerical values become arrays without changing ordering or values. `manifest.json` records both the current checksum and the original uncompressed pickle/CSV checksum.

All numerical inputs were available locally. The manuscript schematic assets came from the TeX source of arXiv:2511.01845v1. Attribution and original source names are retained; this repository does not assign a new license to third-party data or artwork.
