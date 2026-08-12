# Stochastic FLow Map Learning with GANs

This repository contains a cleaned code package for all numerical examples from the stochastic flow map learning with GANs paper. The code implements the GAN-based stochastic sub-map used in the stochastic flow map learning (sFML) framework. In these examples, a pretrained deterministic/residual flow-map component is loaded first, and a WGAN-GP model is used to learn the stochastic residual component.

## References

- Yuan Chen and Dongbin Xiu, `Learning stochastic dynamical system via flow map operator`, Journal of Computational Physics, 508, 112984, 2024.
  https://iamyuanchen.xyz/pdf/2024ChenXiu.pdf

## Requirements

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

## Repository Structure

The main folders are:

| Folder | Contents |
| --- | --- |
| `configs/` | Example JSON configs for the supported numerical examples. |
| `data/` | MATLAB `.mat` datasets. In the current GitHub-ready copy, only the OU example data is included. |
| `pretrainmodel/` | Pretrained deterministic/residual models needed by the GAN examples. |
| `results/` | Trained GAN result folders. Each provided result folder is intentionally reduced to the config, final GAN weights, and ensemble weights needed for reproduction. |

The main files are:

| File | Purpose |
| --- | --- |
| `SolveMixGANSde.py` | Main training script for the mixed GAN-SDE model. It reads a config, loads data and the pretrained deterministic model, trains `MixGANSde3`, and writes output under `results/<test_name>/`. |
| `MixGANSde3.py` | Main GAN model used in the provided examples. The runs use `model_name = WGAN-GP3`. |
| `GANSde.py` | Shared data handling, monitoring, prediction, and plotting utilities used by the GAN-SDE models. |
| `ResnetPDEwM.py` | Pretrained deterministic/residual model class used by `MixGANSde3` to load the deterministic sub-map. |
| `ModelCheckpoint.py` | Custom checkpoint callback used by the deterministic/residual model code. |
| `cyc_callback.py` | Cyclic learning-rate callback. |
| `Evaulation.py` | Evaluation and plotting utilities for mean/std comparisons, sample plots, loss plots, and Wasserstein-distance plots. |
| `ShowPerformance.py` | Postprocessing script for generating standard evaluation plots from an existing result folder. |
| `ShowValidationMixGAN.py` | Postprocessing script for generating paper-style ensemble validation plots using the saved ensemble checkpoints in `Monitor/Ens_model/`. |

Generated plots, monitor figures, logs, and `predict.mat` files are not included in the cleaned result folders.

## Included Example

The result folders correspond to all numerical examples in Section 5 of the paper.

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

**Generated `.mat` data files are not included in this repository by default, except for the small OU data folder kept as a runnable example. For the other examples, generate the data with `SDEDATA-v1` or place the corresponding `.mat` files under the paths specified in the config before rerunning training or validation.**

## Preparation

### Deterministic Map Pretraining

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

### Config

Each run is controlled by a JSON config file. The saved config in `results/<test_name>/Test_config.json` is the most reliable record for reproducing an existing run.

The main config sections are:

`eqn_config`: equation and deterministic sub-map settings.

- `eqn_name`: name of the SDE example. The code uses this to choose reference densities, reference moments, and plotting ranges.
- `dim`: dimension of the state variable.
- `Delta`: time step size.
- Example-specific parameters: for example `mu`, `sigma`, `theta`, `k`, or other equation parameters used by the corresponding reference solution or drift model.
- `resmodel`: deterministic sub-map type. In the provided examples this is usually `ResNetwM`.
- `resmodel_path`: TensorFlow checkpoint path for the pretrained deterministic/residual model.
- `resconfig_path`: config file for the pretrained deterministic/residual model.

`net_config`: GAN architecture and training settings.

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

`dat_config`: data paths and prediction settings.

- `TrainData_dir`: path to the training `.mat` file.
- `TestData_dir`: path to the test `.mat` file.
- `n_ea_traj`: number of sampled training segments per long trajectory.
- `N_pred`: number of prediction trajectories expected in the output. This should match the test-data trajectory count for the standard prediction routines.

`show_config`: controls standard postprocessing plots in `ShowPerformance.py`.

- `plot_samplecompare`: generate sample trajectory comparison plots.
- `plot_meancompare`: generate mean/std comparison plots.
- `plot_losthist`: generate loss-history and Wasserstein-distance plots.

`monitor_config`: controls diagnostics during training and paper-style validation.

- `pdf_monitor`: conditional density plotting during training.
- `repdf_display`: repeated conditional density display.
- `traindata_hist`: training-data histogram diagnostics.
- `traintransin_hist`: transition-input histogram diagnostics.
- `fake_check`: generated-sample diagnostics.
- `cond_mv`: conditional mean/variance diagnostics.
- `Evameanv`: recursive prediction and mean/std diagnostics.
- `loss`: loss plotting frequency.
- `Ens_monitor`: ensemble-checkpoint diagnostics. In the provided result folders, the saved ensemble checkpoints are kept under `Monitor/Ens_model/`.

### Data Format

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

## Model Execution

### Training

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

See the Model Outputs section for the files and figures generated by a full run. The cleaned result folders in this repository keep only the config and trained weights needed for reproduction.

### Post Test and Validation

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

## Model Outputs

During training, `SolveMixGANSde.py` writes outputs under:

```text
results/<test_name>/
```

The main outputs from a full run are:

```text
results/<test_name>/Test_config.json
results/<test_name>/Test_history.json
results/<test_name>/Test_model/
results/<test_name>/predict.mat
results/<test_name>/Monitor/
```

`Test_config.json` records the exact config used for the run. `Test_history.json` records the training history. `Test_model/` stores the final GAN checkpoint. `predict.mat` stores generated prediction trajectories. `Monitor/` stores training diagnostics and checkpoint snapshots.

The provided runs use ensemble prediction to reduce uncertainty from a single saved neural-network checkpoint. During training, the code saves 10 consecutive models from the later stage of training under:

```text
results/<test_name>/Monitor/Ens_model/
```

During prediction, each generated trajectory is propagated by randomly selecting one of these 10 saved models. The selection is uniform: each model is chosen with probability `1/10`. A fresh random draw is used during prediction, so the final prediction ensemble mixes outputs from the 10 consecutive saved models rather than relying on a single checkpoint.

This is the reason the cleaned result folders keep `Monitor/Ens_model/` in addition to `Test_model/`: `Test_model/` stores the final GAN checkpoint, while `Monitor/Ens_model/` stores the ensemble checkpoints used by the paper-style prediction and validation routines.

The plotting scripts generate figures from these outputs. `ShowPerformance.py` can create sample-comparison plots, mean/std comparison plots, loss-history plots, and Wasserstein-distance plots. `ShowValidationMixGAN.py` uses `Monitor/Ens_model/` and writes paper-style ensemble validation figures into:

```text
results/<test_name>/a/
```

The generated figures, monitor images, logs, `predict.mat`, and `a/` folders are not tracked in the cleaned result package by default. The cleaned result folders keep the config, final GAN weights, and ensemble weights needed for reproduction.

## Minimal Reproduction Workflow

For a saved GAN example in `results/<test_name>/`, the usual reproduction workflow is:

1. Generate or copy the required `.mat` data files into the paths specified by `results/<test_name>/Test_config.json`.
2. Make sure the deterministic flow-map folder referenced by `resmodel_path` and `resconfig_path` exists under `pretrainmodel/`.
3. Generate standard evaluation plots:

```bash
python ShowPerformance.py --test_name=<test_name>
```

4. Generate paper-style ensemble validation plots:

```bash
python ShowValidationMixGAN.py --test_name=<test_name>
```

Generated prediction files and figures are written under the corresponding `results/<test_name>/` subfolders and are not tracked by git.

