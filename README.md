# Lab 1 - Artificial Neural Networks and Deep Architectures, Fall 2020

## Part 2
### Setting up

*All instructions assume you're using the command prompt/terminal/console. Execute each line in the instruction boxes separately.*

1. Clone the repository from the command line, then move into the directory: 
   ```bash
   git clone https://github.com/molguin92/ANNDA_HT20_LAB1.git
   cd ANNDA_HT20_LAB1
   ```

2. Set up a virtual Python 3.8 environment using either VirtualEnv or Conda. In my case, I prefer VirtualEnv, so I would do:
   ```bash
   virtualenv --python=python3.8 ./venv
   ```
   
   And then to activate the virtualenv
   
   (on Linux/Mac)
   ```bash
   . ./venv/bin/activate
   ```
   
   (on Windows)
   ```cmd
   ./venv/Scripts/activate
   ```
   
3. Install the required libraries:
   ```bash
   pip install -Ur requirements.txt
   ```
   
4. Compile Jupyter (some plugins require this step, sorry). This step requires you to have `nodejs` on your system, [which you can get from here](https://nodejs.org/en/download/).
   ```bash
   jupyter nbextension install --py jupytext
   jupyter nbextension enable --py jupytext
   jupyter lab build
   ```
   
### Running the code

For Part II, there's three main chunks of code:

1. The script `mackey_glass.py`, which defines the Mackey-Glass time-series function.

2. The script `part2.py`. This module automates the evaluation of the different configurations of MLPs. It generates combinations of different numbers of nodes in hidden layers and regularization parameters, evaluates them all in parallel and outputs the results to CSV files in `./data`. Each MLP configuration is tested a 100 times on different train/test splits, so this takes a while.

   **Be careful when running this one, as it takes a long time to run and overwrites previous data!!**

3. The `lab1_part2.ipynb` Jupyter Notebook. This notebook takes the results from `part2.py` and performs the analysis and plotting.
