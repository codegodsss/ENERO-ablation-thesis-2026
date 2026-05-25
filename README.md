# This repository is a personal research project based on the original ENERO paper and codebase. The main branch preserves the original source code as the baseline reference, while the variant/v1 branch contains my personal modifications, improvements, and experimental changes for research and evaluation purposes. In other words, main is used to keep the original implementation intact, and variant/v1 is where I develop and test my own ideas based on the original work. This repository is intended for academic study, reproduction, comparison, and further exploration rather than serving as an official implementation from the original authors.
# ENERO: Efficient real-time WAN routing optimization with Deep Reinforcement Learning

#### Link to paper: [here](https://www.sciencedirect.com/science/article/pii/S1389128622002717)

#### Paul Almasan, Shihan Xiao, Xiangle Cheng, Xiang Shi, Pere Barlet-Ros, Albert Cabellos-Aparicio

Contact: <felician.paul.almasan@upc.edu>

[![Twitter Follow](https://img.shields.io/twitter/follow/PaulAlmasan?style=social)](https://twitter.com/PaulAlmasan)
[![GitHub watchers](https://img.shields.io/github/watchers/BNN-UPC/ENERO?style=social&label=Watch)](https://github.com/BNN-UPC/ENERO)
[![GitHub forks](https://img.shields.io/github/forks/BNN-UPC/ENERO?style=social&label=Fork)](https://github.com/BNN-UPC/ENERO)
[![GitHub stars](https://img.shields.io/github/stars/BNN-UPC/ENERO?style=social&label=Star)](https://github.com/BNN-UPC/ENERO)

## Instructions to set up the Environment (It is recommended to use Linux)
This paper implements the PPO algorithm to train a DRL agent that learns to route src-dst traffic demands using middelpoint routing. 

1. First, create the virtual environment and activate the environment.
```ruby
virtualenv -p python3 myenv
source myenv/bin/activate
```

2. Then, we install all the required packages.
```ruby
pip install -r requirements.txt
```

3. Register custom gym environment.
```ruby
pip install -e gym-graph/
```

## Instructions to prepare the datasets

The source code already provides the data, the results and the trained model used in the paper. Therefore, we can start by using the datasets provided to obtain the figures used in the paper.

1. Download the dataset from [here](https://drive.google.com/file/d/1gem-VQ5MY3L54B77XUYt-rTbemyKmaqs/view?usp=sharing) or [here](https://bnn.upc.edu/download/enero-dataset/) and unzip it. The location should be immediatly outside of Enero's code directory. 

![image](https://user-images.githubusercontent.com/87467979/215685300-de8c071d-c8f7-4ffa-be6a-c642f04a7d76.png)

2. Then, enter in the unziped "Enero_datasets" directory and unzip everything.

## Extended Analysis: Geant2001-Focused Evaluation

### Dataset Setup for Analysis Scripts

The analysis scripts require pre-computed evaluation results. After downloading and setting up the `Enero_datasets` folder as described above, the following directory structure should exist:

```
~/ENERO-Thesis/Enero_datasets/
├── dataset_sing_top/
│   └── data/results_single_top/
│       ├── Geant2001/
│       │   ├── Geant2001.graph          # Network topology
│       │   └── TM/                      # Traffic matrices (TM_0 to TM_49)
│       ├── evalRes_Geant2001/
│       │   ├── Enero_3top_15_B_NEW/Geant2001/    # DRL evaluation results
│       │   └── SP_3top_15_B_NEW/Geant2001/       # Shortest path results
└── rwds-results-1-link_capacity-unif-05-1-zoo/
    └── SP_3top_15_B_NEW/Geant2001/    # Alternative SP results (fallback path)
```

**Key Files Needed:**
- `Geant2001.graph` - Network topology file with 27 nodes, 38 edges
- `Geant2001.0.demands` through `Geant2001.49.demands` - 50 traffic matrix files
- Pickle files (`*.pckl`) in `evalRes_Geant2001/` directories - Pre-computed routing and utilization data

### Running the Analysis Suite

The analysis is organized in two folders with specific execution order:

#### **Step 1: Prerequisite Analysis** (`geant2001_analysis_v3/`)

These scripts extract basic inference metrics and prepare data for hypothesis testing:

```bash
cd ENERO-ablation-thesis-2026
python3 geant2001_analysis_v3/geant2001_analysis_v3.py       # Generates inference_results.csv
python3 geant2001_analysis_v3/geant2001_plots_v3.py          # Creates preliminary visualizations
python3 geant2001_analysis_v3/geant2001_deep_analysis_v3.py  # Generates deep_analysis.csv & convergence.csv
```

**Outputs (saved to `analysis_output/`):**
- `geant2001_inference_results.csv` - Core metrics (delays, costs, convergence)
- `geant2001_deep_analysis.csv` - Feature-level analysis data
- `geant2001_convergence.csv` - Training convergence statistics

#### **Step 2: Main Analysis** (`geant2001_3focused_v3/`)

These three scripts perform comparative analysis on the prerequisite outputs:

```bash
# Analysis A: Link-level utilization (per-link performance metrics)
python3 geant2001_3focused_v3/analysis_h1_link_level.py
# Outputs: fig_h1a.pdf/png (heatmap), fig_h1b.pdf/png (scatter), fig_h1c.pdf/png (top-10),
#          fig_h1d.pdf/png (regression), fig_h1e.pdf/png (bottleneck identification)
#          geant2001_link_utilization.csv

# Analysis B: OSPF vs ENERO comparison (statistical significance testing)
python3 geant2001_3focused_v3/analysis_h2_ospf_vs_enero.py
# Outputs: fig_h2a.pdf/png (violin plot), fig_h2b.pdf/png (confidence intervals),
#          fig_h2c.pdf/png (CDF), fig_h2d.pdf/png (quality vs time trade-off),
#          fig_h2e.pdf/png (scatter correlation), ospf_vs_enero_stats.csv

# Analysis C: Failure mode classification (DRL rescue scenarios)
python3 geant2001_3focused_v3/analysis_h4_failure_mode.py
# Outputs: fig_h4a.pdf/png (quadrant classification), fig_h4b.pdf/png (feature importance),
#          fig_h4c.pdf/png (convergence analysis), fig_h4d.pdf/png (contribution heatmap),
#          fig_h4e.pdf/png (case study), failure_mode_classification.csv
```

**Total Output:** 30 figure files (15 figures × PDF+PNG) + 3 CSV data files

### What Each Analysis Script Does

| Script | Focus | Key Metrics | Output |
|--------|-------|-------------|--------|
| **H1** | Per-link utilization | Betweenness-centrality correlation, relief patterns | Heatmaps, scatter plots, link rankings |
| **H2** | Routing method comparison | Mean delay, cost differences, significance tests | Violin plots, confidence intervals, CDF |
| **H4** | When DRL fails | Quadrant classification, feature importance | Failure mode matrix, feature contribution |

### Prerequisites for Running Analysis

Before running the scripts, ensure:

1. **Dataset is ready** - Run the dataset preparation steps above
2. **Directory structure exists** - Scripts auto-detect common paths; if paths differ, edit `geant2001_common.py` to add your paths
3. **Python environment active** - `source myenv/bin/activate`
4. **Required packages installed** - `pip install -r requirements.txt`

### Troubleshooting Common Issues

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: Thiếu đường dẫn bắt buộc` | Dataset paths not found. Verify `Enero_datasets` location matches `geant2001_common.py` |
| `KeyError: 'tm_id'` | CSV headers don't match. Ensure prerequisite scripts (geant2001_analysis_v3) ran successfully |
| Empty plots / all zeros | Check that pickle files exist and contain valid data. May need to re-run evaluation |
| Memory issues with large datasets | Scripts handle up to 50 TMs; if timeout, reduce TM range in script source |

## Instructions to obtain the Figures from the paper

1. First, we execute the following command:

```ruby
python figures_5_and_6.py -d SP_3top_15_B_NEW 
```

2. Then, we execute the following (one per topology):
```ruby
python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_EliBackbone/EVALUATE/ -t EliBackbone

python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_HurricaneElectric/EVALUATE/ -t HurricaneElectric

python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_Janetbackbone/EVALUATE/ -t Janetbackbone
```

3. Next, we generate the link failure Figures (one per topology):
```ruby
python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../Enero_datasets/dataset_sing_top/LinkFailure/rwds-LinkFailure_HurricaneElectric

python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../Enero_datasets/dataset_sing_top/LinkFailure/rwds-LinkFailure_Janetbackbone

python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../Enero_datasets/dataset_sing_top/LinkFailure/rwds-LinkFailure_EliBackbone
```

4. Finally, we generate the generalization figure
```ruby
python figure_9.py -d SP_3top_15_B_NEW -p ../Enero_datasets/rwds-results-1-link_capacity-unif-05-1-zoo
```

## Instructions to EVALUATE

To evaluate the model we should execute the following scripts. Each script should be executed independently for each topology where we want to evaluate the trained model. Notice that we should point to each topology were we want to evaluate with the flag -f2. In the paper, we evaluated on "NEW_EliBackbone", "NEW_Janetbackbone" and "NEW_HurricaneElectric".

1. First of all, we need to split the original data from the desired topology between training and evaluation. To do this, we should choose from "../Enero_datasets/results-1-link_capacity-unif-05-1/results_zoo/" one topology that we want to evaluate on. Lets say we choose the Garr199905 topology. Then, we need to execute:
```ruby
python convert_dataset.py -f1 results_single_top -name Garr199905
```

2. Next, we proceed to evaluate the model. For example, let's say we want to evaluate the provided trained model on the Garr199905 topology. To do this we execute the followinG script were we indicate with the flag '-d' to select the trained model, with the flag '-f1' we indicate the directory (it has to be the same like in the previous command!) and with '-f2' we specify the topology.
```ruby
python eval_on_single_topology.py -max_edge 100 -min_edge 5 -max_nodes 30 -min_nodes 1 -n 2 -f1 results_single_top -f2 NEW_Garr199905/EVALUATE -d ./Logs/expSP_3top_15_B_NEWLogs.txt
```

3. Once we evaluated over the desired topologies, we can plot the boxplot (Figures 5 and 6 from the paper). Before doing this, we should edit the script and make the "folder" list contain only the desired folder with the results of the previous experiments. Specifically, we edited folders like:
```ruby
folders = ["../Enero_datasets/dataset_sing_top/data/results_single_top/evalRes_NEW_Garr199905/EVALUATE/"]
```
In addition, we also need to modify the script to plot the boxplots properly. Then, we can execute the following command. If we evaluated our model on different topologies, we should modify the script and make the "folders" list include the proper directories.

```ruby
python figures_5_and_6.py -d SP_3top_15_B_NEW 
```

4. We can obtain the Figure 7 from the paper executing the following script for each topology (i.e., "Garr199905").

```ruby
python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/data/results_single_top/evalRes_NEW_Garr199905/EVALUATE/ -t Garr199905
```


5. The next experiment would be the link failure scenario. To do this, we first need to generate the data with link failures. Specifically, we maintain the TMs but we remove links from the network.
```ruby
python3 generate_link_failure_topologies.py -d results-1-link_capacity-unif-05-1 -topology Garr199905 -num_topologies 1 -link_failures 1
```

6. Now we already have the new topologies with link failures. Next is to execute DEFO on the new topologies. To do this, we need to edit the script *run_Defo_all_Topologies.py* and make it point to the new generated dataset. Then, execute the following command and run DEFO. Notice that with the '--optimizer' flag we can indicate to run other optimizers implemented in [REPETITA](https://github.com/svissicchio/Repetita).

```ruby
python3 run_Defo_all_Topologies.py -max_edges 80 -min_edges 20 -max_nodes 25 -min_nodes 5 -optim_time 10 -n 15 --optimizer 100
```

7. The next step is to evaluate the DRL agent on the new topologies. The following script will create the directory "rwds-LinkFailure_Garr199905" which is then used to create the figures.

```ruby
python eval_on_link_failure_topologies.py -max_edge 100 -min_edge 2 -max_nodes 30 -min_nodes 1 -n 2 -d ./Logs/expSP_3top_15_B_NEWLogs.txt -f LinkFailure_Garr199905

python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/data/results_single_top/evalRes_LinkFailure_Garr199905/EVALUATE/ -t Garr199905
```

## Instructions to TRAIN

1. To trail the DRL agent we must execute the following command. Notice that inside the *train_Enero_3top_script.py* there are different hyperparameters that you can configure to set the training for different topologies, to define the size of the GNN model, etc. Then, we execute the following script which executes the actual training script periodically. 

```ruby
python train_Enero_3top.py
```

2. Now that the training process is executing, we can see the DRL agent performance evolution by parsing the log files from another terminal session. Notice that the following command should point to the proper Logs.
```ruby
python parse_PPO.py -d ./Logs/expEnero_3top_15_B_NEWLogs.txt
```

Please cite the corresponding article if you use the code from this repository:

```
@article{ALMASAN2022109166,
title = {ENERO: Efficient real-time WAN routing optimization with Deep Reinforcement Learning},
journal = {Computer Networks},
volume = {214},
pages = {109166},
year = {2022},
issn = {1389-1286},
doi = {https://doi.org/10.1016/j.comnet.2022.109166},
url = {https://www.sciencedirect.com/science/article/pii/S1389128622002717},
author = {Paul Almasan and Shihan Xiao and Xiangle Cheng and Xiang Shi and Pere Barlet-Ros and Albert Cabellos-Aparicio},
keywords = {Routing, Optimization, Deep Reinforcement Learning, Graph Neural Networks},
abstract = {Wide Area Networks (WAN) are a key infrastructure in today’s society. During the last years, WANs have seen a considerable increase in network’s traffic and network applications, imposing new requirements on existing network technologies (e.g., low latency and high throughput). Consequently, Internet Service Providers (ISP) are under pressure to ensure the customer’s Quality of Service and fulfill Service Level Agreements. Network operators leverage Traffic Engineering (TE) techniques to efficiently manage the network’s resources. However, WAN’s traffic can drastically change during time and the connectivity can be affected due to external factors (e.g., link failures). Therefore, TE solutions must be able to adapt to dynamic scenarios in real-time. In this paper we propose Enero, an efficient real-time TE solution based on a two-stage optimization process. In the first one, Enero leverages Deep Reinforcement Learning (DRL) to optimize the routing configuration by generating a long-term TE strategy. To enable efficient operation over dynamic network scenarios (e.g., when link failures occur), we integrated a Graph Neural Network into the DRL agent. In the second stage, Enero uses a Local Search algorithm to improve DRL’s solution without adding computational overhead to the optimization process. The experimental results indicate that Enero is able to operate in real-world dynamic network topologies in 4.5 s on average for topologies up to 100 links.}
}
```
