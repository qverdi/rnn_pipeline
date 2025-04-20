<h1 align="center">RNN training pipeline</h1>

<p align="center">
  <a href="https://greenai-report.streamlit.app/">
    <img src="https://img.shields.io/badge/Deployed%20on-Streamlit-red?logo=streamlit">
  </a>
</p>




This repository contains a command-line program for experimenting with search spaces to optimize the training time of RNN models. This program uses a set of parameters (search space and experiment parameters) to run experiments and calculate metrics such as MAE, MSE, RMSE, SMAPE, AUNL, and training time. During training, the model's metrics are evaluated every nth epoch (specified as the step check) after a defined patience period, both of which are set in the experiment parameters.

## Table of Contents

- [Installation](#installation)
- [Appflow](#appflow)
- [Project Structure](#project-structure)
- [Features](#features)
- [Streamlit](#streamlit)


## Installation

### Docker

**Requirements**:

- Docker desktop app
- Directory for exposing files from docker container
  - On example, directory named docker-green-ai
    - docker-green-ai/input/data -> for data
    - docker-green-ai/input/params -> for parameters
    - docker-green-ai/output -> for output files

**Installation**

You can build the solution locally
```bash
docker build -t green-ai .
```
or download already built image from DockerHub.

```bash
docker pull janajankovic/green_ai:<version>
```

To run make sure you have absolute path to input and output directories and replace them accordingly in the command below.

```bash
docker run -v <absolute_path/docker-green-ai/input>:/app/input -v <absolute_path/docker-green-ai/output>:/app/output -it <green-ai or janajankovic/green_ai:v>  /bin/bash
```

**Running**

Before running the experiment, make sure to provide files to `input/data` directory and modify experiment parameters in `input/params`.

Run the program by executing following command in the docker container through command line.

```bash
poetry run experiment
```
Check your ouput folder for results.

**Running and commiting to git (local machine only)**

**❗ WARNING:**  This command can be used only if you are collaborator and if you are running it outside of docker container. If you want to run it through docker, you need to set up git credentials inside docker container before running this command.

If you want to keep version of code and results along with experiment parameters, you can run following command:

```bash
poetry run experiment --push --name(opt) <name>
```
This command will run experiments with given parameters, checkout to newly created git branch with name template `experiment\{ddmmyyyy-hhmm}` or `experiment\{ddmmyyyy-hhmm_name}` if name is given, commit and push changes (in this case parameters and experiment result). 


### Local machine

This project is using python version 3.10. Package management is carried by poetry. Hence, it's required to have [pyenv](https://github.com/pyenv/pyenv) installed, and within it python version 3.10.

**Requirements**:

- installed pyenv
- installed python version 3.10.x
- installed poetry

**How to start experiment**

I use VSCode terminal (to open it: `` CTRL + `  ``) I run the project with the following command.

```
poetry run experiment
```

**How to generate report**

Report is generated using streamlit framework and the latest experiment is deployed to the website stated in section [Streamlit](#streamlit). 

However, if you want to run report locally on experiment that you don't want to publish, you can do so by running following command:

```
poetry run report
```
This command will create web application accessible on `http://localhost:8501/`.


**Versioning errors**

In case that poetry throws errors regarding python version, the following approach works for me:

```
> poetry env info

Virtualenv
Python:         3.10.5 (here might be a different version)
Implementation: CPython
Path:           \AppData\Local\pypoetry\Cache\virtualenvs\green-ai-j1pS00bB-py3.10
Executable:     \AppData\Local\pypoetry\Cache\virtualenvs\green-ai-j1pS00bB-py3.10\Scripts\python.exe
Valid:          True

Base
Platform:   win32
OS:         nt
Python:     3.10.5
Path:       \.pyenv\pyenv-win\versions\3.10.5
Executable: \.pyenv\pyenv-win\versions\3.10.5\python.exe

```

Copy Executable path from Virtualenv.

```
> poetry env remove {executable path}
```

Set pyenv local python version.

```
> pyenv local 3.10.6
```

Copy path set by pyenv.

```
> pyenv which python

{path}\.pyenv\pyenv-win\versions\3.10.6\python.exe

```

If there is a different version of python in pyproject.toml change it to desired 3.10 version. Delete poetry lock. Set python version for poetry.

```
> poetry env use {path}\.pyenv\pyenv-win\versions\3.10.6\python.exe
```

Install poetry.

```
poetry install
```

## Appflow

The app requires the following inputs:

- Tabular data from .txt, .csv, .xls, or .xlsx files located in the /input/data/ directory.
- Experiment parameters from /input/config/experiment_params.json.
- Search space parameters from /input/config/model_params.json.


Currently, the app supports multiple files for the same experiment parameters and multiple files for the same the search space. This is handled by the `ExperimentHandler`, which assigns each data source the defined parameters. It is possible to have uni-variate and multi-variate time series. By default program will load all columns from csv, if there are more than one. If you want to do time series analysis for uni-variate type, make sure that csv file has only one column and that `target_column` is set to 0. In case, you have to do multi-variate analysis make sure to set `target_column` to the index of the desired column which will be used for prediction. By default RNN models will be trained on all columns, but predict value based only on column defined as `target_column`.

After this step, the `HPOManager`, based on the experiment settings, selects the HPO Optimizer (RS, BO). The HPO Optimizer defines an objective function that executes the model training. The model is compiled and built based on the hyperparameters chosen by the HPO Optimizer.

During model training, the `MetricCallback` is called for evaluating results. This happens after every epoch for the purpose of post hoc analysis. Metrics such as MAE, MSE, RMSE, and SMAPE on the test data are calculated, while AUNL is calculated on both the training loss and validation loss. These results are logged in an experiment-specific file in the output directory as `e_{experiment.id_filename}.csv`, alongside all experiment parameters, budget parameters, search space parameters, model ID, epoch number, training time, loss, and validation loss values.

After training completes through all epochs, the final metric (currently, loss) is sent back to the HPO Optimizer, which, based on these results, selects the next set of parameters until the given number of iterations is reached.

Another way of completing the optimization process is through use of budget, which is defined in experiment parameters as `include_budget` property in JSON. Budget represents total number of epochs available to experiment. If budget strategy is set to `fixed`, optimization will terminate when last epoch reaches budget. In case of `greedy` setting, it will terminate last generation (DE, GA) or model (Search), based on tolerance and remainig epochs. 

<p align="center">
    <img src="doc/img/AppFlow.png" alt="Example Image"/>
</p>

## Project Structure

### Folder structure

Description of folder structure for this project.

```
├───doc (Documentation, examples and files and images for README)
│
├───input(Required files for the program to run)
│   │
|   ├───data (Files for model training. Required at least one is expected)
│   │       data.txt
│   │       
│   └───params (User defined parameters)
│           experiment_params.json (Parameters on experiment level)
│           model_design.json (Model architecture definition file)
│           model_optimizer_params.json (Model optimizer parameters for fine tuning)
│           model_params.json (Parameters for search space)
│           search_params.json (Parameters for RS, TPE)
│
├───output (Results: Parameters and metrics of all experiments)
│   │   e_{id}_{file_name}.csv
│
└───src (Source code)
    │
    ├───cmd (Scripts to be called from cmd)
    │   │   main.py (Main entry to the application)
    │   │   report.py (Generate report based on results from main program)
    │
    ├───config (Constants and other needed configuration of the app)
    │   │   file_constants.py
    │   
    │
    ├───data (Classes for all data related processes)
    │   │   data_container.py (Contains data, that's ready for training)
    │   │   data_loader.py (Loads different data source files)
    │   │   data_processing.py (Splits data, scales and creates sequences)
    │   
    │
    ├───experiments (Experiment related classes and operations)
    │   │   experiment_budget.py (For storing budget params)
    │   │   experiment_handler.py (Creates experiments and conducts them)
    │   │   experiment_param.py (For storing experiment params)
    │   
    │
    ├───models (All model related operations)
    │   │   model.py (Builds and compiles model)
    │   │   model_budget_callback.py (Callback for budget control)
    │   │   model_evaluation.py (Evaluates model)
    │   │   model_metrics.py (Stores all metrics for result file)
    │   │   model_metric_callback.py (Callback for evaluation during training)
    │   │   model_optimizer_params.py (Model optimizer parameters)
    │   │   model_params.py (Model parameters)
    │   │   model_stopping_callback.py (Callback for model training termination)
    │   │   model_training.py (Trains model)
    │   
    │
    ├───notebooks (Development folder)
    │       main.ipynb
    │
    ├───optimization (All HPO related operations)
    │   │   hpo_manager.py (Selects optimizer based on experiment)
    │   │   hpo_stopping.py (Stopping logic due to budget exhaustion)
    │   │   hpo_termination_error.py (Custom error for terminating optimization)
    │   │
    │   ├───search (Search algorithm processes)
    │   │   │   search_optimizer_callback.py (Stopping callback for search)
    │   │   │   search_optimizer_params.py (Container class for parameters)
    │   │   │   search_optimizer_set.py (Parameter set creator for model-expected classes)
    │   │   │   search_optimizier.py (Entry for Search algorithm for HPO)
    │
    ├───report (Report generation)
    │   │   report.py (Wrapper class)
    │   │   report_data_filter.py (Process data for plots)
    │   │   report_initializer.py (Generates streamlit app)
    │   
    ├───utils (Other utility functions)
    │   │   git_commands.py (Commit, push and deploy experiment results)
    │   │   gpu_config.py (Configure GPU in case of multiple GPU)
    │   │   input_validator.py (Validates input from JSON files)
    │   │   json_utils.py (JSON related utilities)
    │   │   logger.py (Logging related utilities)
    │   │   model_calculations.py (Commonly used mathematical formulas: AUNL, SMAPE)
    │   │   os_utils.py (OS related operations)
    │   │   plots.py (Functions for plot creation)
    │   │   value_mapper.py (Maps valules between dimensions, like normalization)
```

### Input data structure

**Experiment** options settings are presented below from the file input/params/experiment_params.json

```json
{
  "gpu": 0, // Index of preferred GPU, if multiple
  "target_column": 0, // Index of the target column in the dataset
  "loss": "mse", // Loss function to be used in model training
  "optimizer": "search", // Type of optimization: RS (Random Search), BO (Bayesian Optimization), or genetic
  "early_stopping": "aunl", // Which time of early stopping is used
  "num_epochs": 100, // Number of epochs for training
  "patience": 10, // Number of epochs to wait before early stopping
  "step_check": 5, // Number of epochs to wait before evaluating again
  "train_size": 0.7, // Proportion of the dataset used for training
  "val_size": 0.1, // Proportion of the dataset used for validation
  "window_size": 30, // Size of the moving window for creating time-series sequences
  "optimizer_finetuning": false, // If true, optimizer hyperparameters will be fine-tuned
  "is_descrete": true, // If true, GA and DE parameters are discretely mapped; if false, they are continuously mapped
  "has_model": true, // If true, a fixed model architecture is used from input/params/model_design.json
  "include_budget": false, // If true, optimization will terminate when budget is exhausted
  "budget": {
    "budget": 5000, // Maximum budget for optimization (e.g., number of evaluations, time, or iterations)
    "strategy": "fixed", // Strategy for budget allocation (e.g., fixed, greedy)
    "reference_metric": "mse", // Metric used to determine budget utilization
    "tolerance": 100 // Tolerance level for budget-based stopping
  }
}

```

**Search** parameters for RS, BO search algorithms are presented below from the file input/params/search_params.json

```json
{
  "sampler": "bayesian", // Sampler for search algorithm. Only from the list [random, bayesian]
  "trials": 3
}
```

**Search space** options settings are presented below from the file input/params/model_params.json

```json
{
  "activation": ["relu", "tanh", "sigmoid"], //Must be from official tensorflow functions - tf.keras.activations
  "batch_size": [16, 32, 64], // Any int number
  "dropout_rate": [0.1, 0.2, 0.3], // Any float number, for each layer
  "learning_rate": [0.01, 0.001, 0.0001], // Any float number
  "network_type": ["rnn", "lstm", "gru"], // Only from the list [rnn, lstm, gru]
  "num_layers": [1, 2, 3], // Total number of reccurrent layers
  "num_neurons": [16, 32, 64], // Number of neurons per each layer
  "optimizer": ["adam", "rmsprop", "sgd"] // Only from the list [adam, rmsprop, sgd]
}
```

**Optimizer parameters for** finetuning are presented below from the file input/params/model_optimizer_params.json
```json
{
    "beta_1": [0.8, 0.85, 0.9, 0.95, 0.99],
    "beta_2": [0.99, 0.995, 0.999, 0.9995, 0.9999],
    "rho": [0.85, 0.9, 0.95, 0.99],
    "momentum": [0.0, 0.5, 0.9, 0.99],
    "centered" : [true, false],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "nesterov": [true, false],
    "amsgrad": [true, false],
    "clipnorm": [0.0, 1.0, 5.0, 10.0],
    "clipvalue": [0.0, 0.5, 1.0],
    "clipping": ["clipnorm", "clipvalue"]
}
```
Since there its allowed to have only one clipping out of `["clipnorm", "clipvalue"]`, the HPO algorithm will pick one from this categorical list. Others will be set to none. 

❗Note that the list of parameters is big, because HPO algorithm can pick any of the optimizers defined in **search space** file (Adam, SGD or RMSProp). They all have different hyperparameters, so they have to be listed here. However if you have smaller search space for optimizer, on example, you are picking between Adam and SGD, then hyperparameters for RMSProp are ignored. 

### Output
Output of this project is .csv file that contains following information.

```csv
timestamp,experiment_id,file_path,loss_function,num_epochs,hpo,patience,train_size,val_size,step_check,window_size,optimizer_finetuning,is_descrete,has_model,aunl_stopping,include_budget,budget,budget_strategy,reference_metric,tolerance,n_trails,search_sampler,population_size,generations,crossover_prob,mutation_prob,elite,indpb,tournsize,strategy,popsize,maxiter,recombination,mutation,init,polish,tol,atol,model_id,num_layers,layers,neurons,activation,dropout_rate,learning_rate,batch_size,optimizer,beta_1,beta_2,rho,momentum,centered,weight_decay,clipnorm,clipvalue,evaluation_epoch,epoch,mae,mse,smape,rmse,loss,val_loss,aunl,aunl_val,training_time
2025-02-10 16:08:35,a4fe3b3e,mm_103770.csv,mse,100,search,10,0.7,0.1,5,30,False,True,True,False,False,,,,,100,random,,,,,,,,,,,,,,,,,c36cd07d,3,"['lstm', 'lstm', 'lstm']","[8, 32, 8]","['relu', 'relu', 'relu']","[0.1, 0.2, 0.1]",0.0001,64,adam,,,,,,,,,False,0,5.0198,32.8786,0.7108,5.734,0.1547,0.1138,1,1,2.6407
2025-02-10 16:08:35,a4fe3b3e,mm_103770.csv,mse,100,search,10,0.7,0.1,5,30,False,True,True,False,False,,,,,100,random,,,,,,,,,,,,,,,,,c36cd07d,3,"['lstm', 'lstm', 'lstm']","[8, 32, 8]","['relu', 'relu', 'relu']","[0.1, 0.2, 0.1]",0.0001,64,adam,,,,,,,,,False,1,4.9182,31.8434,0.6896,5.643,0.1486,0.1102,0.5,0.5,2.9883
2025-02-10 16:08:36,a4fe3b3e,mm_103770.csv,mse,100,search,10,0.7,0.1,5,30,False,True,True,False,False,,,,,100,random,,,,,,,,,,,,,,,,,c36cd07d,3,"['lstm', 'lstm', 'lstm']","[8, 32, 8]","['relu', 'relu', 'relu']","[0.1, 0.2, 0.1]",0.0001,64,adam,,,,,,,,,False,2,4.8369,31.022,0.6729,5.5697,0.1445,0.1074,0.4496,0.4712,3.3504
2025-02-10 16:08:36,a4fe3b3e,mm_103770.csv,mse,100,search,10,0.7,0.1,5,30,False,True,True,False,False,,,,,100,random,,,,,,,,,,,,,,,,,c36cd07d,3,"['lstm', 'lstm', 'lstm']","[8, 32, 8]","['relu', 'relu', 'relu']","[0.1, 0.2, 0.1]",0.0001,64,adam,,,,,,,,,False,3,4.7458,30.1135,0.6546,5.4876,0.1411,0.1042,0.4349,0.4847,3.7128
2025-02-10 16:08:36,a4fe3b3e,mm_103770.csv,mse,100,search,10,0.7,0.1,5,30,False,True,True,False,False,,,,,100,random,,,,,,,,,,,,,,,,,c36cd07d,3,"['lstm', 'lstm', 'lstm']","[8, 32, 8]","['relu', 'relu', 'relu']","[0.1, 0.2, 0.1]",0.0001,64,adam,,,,,,,,,False,4,4.6387,29.066,0.6333,5.3913,0.1372,0.1006,0.4461,0.5041,4.0776
```

Output file contains following parameters:

- `timestamp`: Timestamp of the experiment.
- `experiment_id`: Unique identifier for the experiment.
- `file_path`: Path to the experiment file.
- `loss_function`: Loss function used in the model.
- `num_epochs`: Number of epochs for training.
- `hpo`: Hyperparameter optimization method.
- `patience`: Patience for early stopping.
- `train_size`: Size of the training dataset.
- `val_size`: Size of the validation dataset.
- `step_check`: Step interval for monitoring.
- `window_size`: Size of the sliding window for data preprocessing.
- `budget`: Budget allocated for the experiment.
- `budget_strategy`: Strategy used to allocate the budget.
- `reference_metric`: Reference metric for updating global best AUNL.
- `tolerance`: Tolerance for early stopping.
- `n_trails`: Number of trials in hyperparameter optimization.
- `search_sampler`: Sampling strategy for hyperparameter search.
- `init`: Initialization strategy for optimization.
- `model_id`: Unique identifier for the trained model.
- `num_layers`: Number of layers in the model architecture.
- `layers`: Types of layers used in the model.
- `neurons`: Number of neurons per layer.
- `activation`: Activation functions used in the model.
- `dropout_rate`: Dropout rate for regularization.
- `learning_rate`: Learning rate for optimization.
- `batch_size`: Size of batches used in training.
- `optimizer`: Optimizer used for training the model.
- `beta_1`: Exponential decay rate for the first moment estimates (Adam).
- `beta_2`: Exponential decay rate for the second moment estimates (Adam).
- `rho`: Decay parameter for RMSprop.
- `momentum`: Momentum parameter for optimizers.
- `centered`: Whether the optimizer uses centered gradients (RMSprop).
- `weight_decay`: Weight decay for regularization.
- `clipnorm`: Gradient clipping by norm.
- `clipvalue`: Gradient clipping by value.
- `evaluation_epoch`: Epoch interval for evaluation.
- `epoch`: Current epoch during training.
- `mae`: Mean Absolute Error metric.
- `mse`: Mean Squared Error metric.
- `smape`: Symmetric Mean Absolute Percentage Error metric.
- `rmse`: Root Mean Squared Error metric.
- `loss`: Training loss at a given epoch.
- `val_loss`: Validation loss at a given epoch.
- `aunl`: A custom evaluation metric (e.g., area under normalized loss).
- `aunl_val`: Validation score for the custom metric.
- `training_time`: Time taken to train the model.


## Features

- ✅ Dynamic customization of parameters (through JSON file)
- ✅ Support for conducting same experiment parameters on multiple data files
- ✅ Support for file extenstions CSV, TXT, XLS, XLXS
- ✅ Search (RS, BO) algorithms for HPO
- ✅ AUNL Calculation
- ✅ Patience, step evaluation
- ✅ Results logging
- ✅ Modularity for implementation of different HPO Optimizers
- ✅ Report generation
- ✅ Fine tuning already known model architecture (optimizer, learning rate, batch_size)
- ✅ HPO for unfrozen layers (freezing feature) of known model architecture
- ✅ HPO for specific layer parameters of known model architecture
- ✅ Input Validator
- ✅ GPU Adaptation (Runs automatically on GPU, if multiple, user sets which) * only on WSL, requires setting up CUDA and cuDNN
- ✅ Parallel computing of multiple experiments (only not in case of DE)
- ✅ Dockerized solution
- ✅ AUNL Stopping
- ✅ Budget based termination
- ✅ Model optimizer parameter finetuning
- ✅ Multi-variate time series 
- ✅ Automatic experiment versioning by using 

## Streamlit

The results of the latest experiment are available on this streamlit site:

https://greenai-report.streamlit.app/

❗In case of inactivity app can hibernate. Please wait few minutes before it becomes available again.

💡 Currently showing results from: `experiment/11022025`
