# Spike Encodings for Remaining Useful Life Prediction with Spiking Neural Networks

Code for the paper **"Encoding Matters: A Systematic Study of Spike Encodings and a Two-Timescale Spiking Neural Network for Energy-Efficient Remaining Useful Life Prediction of Turbofan Engines"** (under review).

The code compares seven spike encodings (rate, time-to-first-spike, population, send-on-delta, anchored send-on-delta, level-crossing, direct input) for predicting the remaining useful life (RUL) of turbofan engines on the NASA C-MAPSS FD001 and FD004 benchmarks, using a two-timescale recurrent spiking neural network, and compares them with LSTM and 1D-CNN baselines. Every number, table and figure in the paper is produced by the scripts in this repository.

## Main results (test RMSE in cycles, mean ± s.d. over 3 seeds, capped RUL)

| Model | FD001 | FD004 | Estimated energy per inference (FD001) |
|---|---|---|---|
| LSTM | 14.10 ± 0.59 | 16.67 ± 1.39 | 2,756 nJ |
| 1D-CNN | 14.89 ± 0.87 | 17.30 ± 0.41 | 1,727 nJ |
| SNN – SoD-anchored | **16.20 ± 0.29** | 21.56 ± 0.52 | **64.6 nJ** |
| SNN – population | 17.19 ± 0.45 | **19.63 ± 0.28** | 277.1 nJ |

Energy values are estimates from operation counts (see Section 4.2 of the paper), not hardware measurements.

## Quick start

### Option A – your own computer (Windows, macOS or Linux; CPU only)
```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python step2_download_data.py           # NASA C-MAPSS data -> data/
python step3_quick_test.py              # installation check
python step4_fig1_encodings.py          # Fig. 1
python step5_fig2_architecture.py       # Fig. 2
python step6_table3_fig4_information.py # Table 3, Fig. 4
python step7_run_experiments.py         # 54 runs, ~1.5-3 h, resumable
python step8_analyze.py                 # summary; compare with expected_output/
python step9_tables_5_to_8.py           # Tables 5-8
python step10_figs_3_5_6.py             # Figs. 3, 5, 6
```
Outputs are written to `figures/` and `tables/`. `expected_output/` contains our own outputs for comparison; with the pinned package versions on a CPU the numbers match exactly.

### Option B – Google Colab (no installation)
Open `AROGYA_N1_Colab.ipynb` in Colab and follow the cells. Detailed instructions: `docs/Google_Colab_Guide.docx`.

## Repository structure
| File | Purpose |
|---|---|
| `n1.py` | Preprocessing, the seven encodings and decoders, the two-timescale SNN, LSTM and CNN baselines, metrics, training |
| `step2` … `step10` scripts | One script per step; each produces the data, figure or table named in its file name |
| `common_plot.py`, `dl.py` | Plotting helpers |
| `AROGYA_N1_Colab.ipynb` | Google Colab notebook running all steps |
| `expected_output/` | Our outputs (tables, summary, figures) for comparison |
| `docs/` | Beginner guides (local computer and Google Colab) |

## Paper ↔ code
| Paper | Code |
|---|---|
| Section 3.2 preprocessing | `load()` in `n1.py` |
| Section 3.3 encodings | `enc_*()` and `decode()` in `n1.py` |
| Section 3.4 network (Eqs. 2–4) | `init_snn()`, `snn_apply()` in `n1.py` |
| Section 3.6 training | `train()` in `n1.py` |
| Section 4.2 energy model, 4.3 robustness | `evaluate_snn()`, `perturb()` in `step7_run_experiments.py` |
| Appendix A settings | constants at the top of `n1.py` |

## Data
NASA C-MAPSS turbofan engine degradation simulation data (Saxena et al., 2008), from the NASA Prognostics Center of Excellence data repository. The data are not stored in this repository; `step2_download_data.py` downloads them.

## Citation
See `CITATION.cff`. The full reference will be added after publication.

## License
MIT – see `LICENSE`.
"# snn_spike_encoding_rul" 
