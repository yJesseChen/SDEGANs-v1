# Stochastic FLow Map Learning with GANs

This repository contains a cleaned code package for all numerical examples from

Chen, Y. and Xiu, D. (2024), "Learning stochastic dynamical system via flow map operator", Journal of Computational Physics.

Paper: https://iamyuanchen.xyz/pdf/2024ChenXiu.pdf

The code implements the GAN-based stochastic sub-map used in the stochastic flow map learning (sFML) framework. In these examples, a pretrained deterministic/residual flow-map component is loaded first, and a WGAN-GP model is used to learn the stochastic residual component.

## Repository Structure

Top-level Python files:

- `SolveMixGANSde.py`: main training script for the mixed GAN-SDE model. It reads a config, loads data and the pretrained deterministic model, trains `MixGANSde3`, and writes output under `results/<test_name>/`.
- `MixGANSde3.py`: the main GAN model used in the provided examples. The runs use `model_name = WGAN-GP3`.
- `GANSde.py`: shared data handling, monitoring, prediction, and plotting utilities used by the GAN-SDE models.
- `ResnetPDEwM.py`: pretrained deterministic/residual model class used by `MixGANSde3` to load the deterministic sub-map.
- `ModelCheckpoint.py` and `cyc_callback.py`: helper callbacks used by `ResnetPDEwM.py`.
- `Evaulation.py`: evaluation and plotting utilities for mean/std comparisons, sample plots, loss plots, and Wasserstein-distance plots.
- `ShowPerformance.py`: postprocess script for generating standard evaluation plots from an existing result folder.
- `ShowValidationMixGAN.py`: postprocess script for generating paper-style ensemble validation plots using the saved ensemble checkpoints in `Monitor/Ens_model/`.

Folders:

- `configs/`: example JSON configs for the supported numerical examples.
- `data/`: MATLAB `.mat` datasets. In the current GitHub-ready copy, only the OU example data is included.
- `pretrainmodel/`: pretrained deterministic/residual models needed by the examples.
- `results/`: trained GAN results. Each result folder is intentionally reduced to the information needed for reproduction:
  - `Test_config.json`: exact config used for the saved run.
  - `Test_model/`: final trained GAN checkpoint.
  - `Monitor/Ens_model/`: ensemble checkpoints used by the paper-style validation code.

Generated plots, monitor figures, logs, and `predict.mat` files are not included in the cleaned result folders.

## Deterministic Map Pretraining

The GAN code does not learn the entire stochastic flow map from scratch. Each run first loads a deterministic ResNet flow map, then trains the GAN to learn the stochastic component around that deterministic prediction.

The deterministic map can be generated with the companion repository:

https://github.com/yJesseChen/ResNetPDE-v1

In that repository, train the deterministic model with `SolveResnetwM.py`. For example, for the OU case:

```bash
python SolveResnetwM.py --test_name=SDEEx3OU --config_path=./configs/SDEEx3OU.json
```

After training, copy the deterministic-map files from the ResNetPDE result folder into this repository under `pretrainmodel/<example_name>/`.

Each deterministic model folder in this repository should keep:

```text
pretrainmodel/<example_name>/Test_config.json
pretrainmodel/<example_name>/Best_model/
```

Keep the entire `Best_model/` folder, since TensorFlow checkpoints may contain several files that must stay together.

The GAN config reads the deterministic map through:

```json
"eqn_config": {
  "resmodel": "ResNetwM",
  "resmodel_path": "./pretrainmodel/<example_name>/Best_model/checkpoint",
  "resconfig_path": "./pretrainmodel/<example_name>/Test_config.json"
}
```

For an existing result folder, `results/<test_name>/Test_config.json` is the most reliable source for the exact deterministic-map paths used by that run.

The provided examples use the following deterministic-map folders:

| GAN result folder | Deterministic map folder |
|---|---|
| `Excute_Ex3OUM3_1s1` | `pretrainmodel/Ex3OU/` |
| `Ex1GeoBrownianM3_3c9` | `pretrainmodel/Ex1GeoBrownianm2s1/` |
| `Excute_Ex4ExpDiffM3_2s5` | `pretrainmodel/Ex4ExpDiff/` |
| `Excute_Ex5TrigM3_2s1` | `pretrainmodel/Ex5Trig/` |
| `Excute_Ex8s05DW_3s3` | `pretrainmodel/Ex8DWs05/` |
| `Excute_Ex9_2s5` | `pretrainmodel/Ex9Expdis/` |
| `Excute_Ex6ExpOUM3_1s7` | `pretrainmodel/Ex6ExpOU/` |
| `Excute_Ex7MdOU_4s1` | `pretrainmodel/Ex7MdOU/` |
| `Excute_Ex10_1s9` | `pretrainmodel/Ex10SOs01/` |

To make a GAN run portable, keep both the deterministic model folder under `pretrainmodel/` and the GAN run folder under `results/`. The deterministic folder supplies the ResNet sub-map, while the result folder supplies the GAN config, final GAN weights, and ensemble checkpoints.

## Environment

The original experiments were run with TensorFlow/Keras. A working environment should include:

- Python 3
- TensorFlow
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- munch
- absl-py
- tqdm

## Running the Main Training Script

Run from the repository root:

```bash
python SolveMixGANSde.py \
  --test_name=<new_result_name> \
  --config_path=./configs/Ex3OUMix.json \
  --model=WGAN-GP3
```

For a new run, choose a fresh `--test_name` so that existing result folders are not overwritten.

The script writes to:

```text
results/<test_name>/
```

Typical outputs from a full run include `Test_config.json`, `Test_history.json`, `Test_model/`, `predict.mat`, and monitor/evaluation figures. The cleaned result folders in this repository keep only the config and trained weights.

## Running Postprocessing Scripts

Standard evaluation plots:

```bash
python ShowPerformance.py \
  --test_name=Excute_Ex3OUM3_1s1
```

Paper-style ensemble validation plots:

```bash
python ShowValidationMixGAN.py \
  --test_name=Excute_Ex3OUM3_1s1
```

`ShowValidationMixGAN.py` uses `results/<test_name>/Monitor/Ens_model/` and writes the paper-style figures into:

```text
results/<test_name>/a/
```

The generated `a/` folder is not tracked in the cleaned result package by default.

## Prediction

The provided runs use ensemble prediction to reduce uncertainty from a single saved neural-network checkpoint.

During training, the code saves 10 consecutive models from the later stage of training under:

```text
results/<test_name>/Monitor/Ens_model/
```

During prediction, each generated trajectory is propagated by randomly selecting one of these 10 saved models. The selection is uniform: each model is chosen with probability `1/10`. A fresh random draw is used during prediction, so the final prediction ensemble mixes outputs from the 10 consecutive saved models rather than relying on a single checkpoint.

This is the reason the cleaned result folders keep `Monitor/Ens_model/` in addition to `Test_model/`: `Test_model/` stores the final GAN checkpoint, while `Monitor/Ens_model/` stores the ensemble checkpoints used by the paper-style prediction and validation routines.

## Data Format

Training and test data are stored as MATLAB `.mat` files. The expected key is:

```text
data
```

The expected array shape is:

```text
[dim, number_of_time_steps, number_of_trajectories]
```

For example, for a one-dimensional SDE, `data` has shape:

```text
[1, number_of_time_steps, number_of_trajectories]
```

The config file specifies the data paths:

```json
"dat_config": {
  "TrainData_dir": "./data/Ex3OU/Ex3OU_t1m12s03_train.mat",
  "TestData_dir": "./data/Ex3OU/Ex3OU_t1m12s03_test.mat",
  "n_ea_traj": 1,
  "N_pred": 5000
}
```

`N_pred` should match the number of test trajectories used when writing prediction arrays.

## Config

Each run is controlled by a JSON config file. The saved config in `results/<test_name>/Test_config.json` is the most reliable record for reproducing an existing run.

The main sections are:

### `eqn_config`

Equation and deterministic sub-map settings.

- `eqn_name`: name of the SDE example. The code uses this to choose reference densities, reference moments, and plotting ranges.
- `dim`: dimension of the state variable.
- `Delta`: time step size.
- Example-specific parameters: for example `mu`, `sigma`, `theta`, `k`, or other equation parameters used by the corresponding reference solution or drift model.
- `resmodel`: deterministic sub-map type. In the provided examples this is usually `ResNetwM`.
- `resmodel_path`: TensorFlow checkpoint path for the pretrained deterministic/residual model.
- `resconfig_path`: config file for the pretrained deterministic/residual model.

### `net_config`

GAN architecture and training settings.

- `N_rec`: number of time steps used in each training sequence. In the GAN code this is the length of the trajectory segment seen by the discriminator.
- `n_Z`: dimension of the Gaussian latent variable used by the generator.
- `G_type`: generator architecture type. The provided runs use MLP-type generators.
- `D_type`: discriminator architecture type. The provided runs use MLP-type discriminators.
- `G_hidden`, `D_hidden`: number of hidden layers in the generator and discriminator.
- `G_nodes`, `D_nodes`: number of nodes per hidden layer.
- `G_opt`, `D_opt`: optimizer settings for generator and discriminator, including learning rate and optimizer hyperparameters.
- `n_critic`: number of discriminator/critic updates per generator update.
- `batch_size`: training batch size.
- `N_epochs`: number of training epochs.
- `Test_mode`: prediction mode during training. The provided runs use `Multiple_last` to collect predictions from multiple late-stage checkpoints.
- `model_name`: model selector. The provided examples use `WGAN-GP3`.

### `dat_config`

Data paths and prediction settings.

- `TrainData_dir`: path to the training `.mat` file.
- `TestData_dir`: path to the test `.mat` file.
- `n_ea_traj`: number of sampled training segments per long trajectory.
- `N_pred`: number of prediction trajectories expected in the output. This should match the test-data trajectory count for the standard prediction routines.

### `show_config`

Controls standard postprocessing plots in `ShowPerformance.py`.

- `plot_samplecompare`: generate sample trajectory comparison plots.
- `plot_meancompare`: generate mean/std comparison plots.
- `plot_losthist`: generate loss-history and Wasserstein-distance plots.

### `monitor_config`

Controls diagnostics during training and paper-style validation.

- `pdf_monitor`: conditional density plotting during training.
- `repdf_display`: repeated conditional density display.
- `traindata_hist`: training-data histogram diagnostics.
- `traintransin_hist`: transition-input histogram diagnostics.
- `fake_check`: generated-sample diagnostics.
- `cond_mv`: conditional mean/variance diagnostics.
- `Evameanv`: recursive prediction and mean/std diagnostics.
- `loss`: loss plotting frequency.
- `Ens_monitor`: ensemble-checkpoint diagnostics. In the provided result folders, the saved ensemble checkpoints are kept under `Monitor/Ens_model/`.

## Provided Examples

The result folders correspond to the numerical examples in Section 5 of the paper.

| Result folder | Paper example | Equation name in config | Provided in this repository |
|---|---|---|---|
| `Excute_Ex3OUM3_1s1` | Section 5.1.1, Ornstein-Uhlenbeck process | `OU Process` | Config, final weights, ensemble weights, and OU data |
| `Ex1GeoBrownianM3_3c9` | Section 5.1.2, geometric Brownian motion | `Geometric Brownian Motion` | Config, final weights, and ensemble weights |
| `Excute_Ex4ExpDiffM3_2s5` | Section 5.2.1, SDE with nonlinear diffusion | `Exp_diffusion` | Config, final weights, and ensemble weights |
| `Excute_Ex5TrigM3_2s1` | Section 5.2.2, trigonometric drift/diffusion example | `Trig_drift` | Config, final weights, and ensemble weights |
| `Excute_Ex8s05DW_3s3` | Section 5.2.3, double-well potential | `Double_well` | Config, final weights, and ensemble weights |
| `Excute_Ex9_2s5` | Section 5.3.1, exponential-noise example | `Exp_dis` | Config, final weights, and ensemble weights |
| `Excute_Ex6ExpOUM3_1s7` | Section 5.3.2, exponential OU / non-Gaussian example | `Exp_OU` | Config, final weights, and ensemble weights |
| `Excute_Ex7MdOU_4s1` | Section 5.4.1, two-dimensional OU process | `MdOU` | Config, final weights, and ensemble weights |
| `Excute_Ex10_1s9` | Section 5.4.2, stochastic oscillator | `SO` | Config, final weights, and ensemble weights |

Only the OU data folder is included in this cleaned copy. For the other examples, the configs retain the expected data paths; provide the corresponding `.mat` files under `data/` before rerunning training or validation.
