# CS 170 Project Spring 2020

Take a look at the project spec before you get started!

Requirements:

Python 3.6+

You'll only need to install networkx to work with the starter code. For installation instructions, follow: https://networkx.github.io/documentation/stable/install.html

If using pip to download, run `python3 -m pip install networkx`

Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs

When writing inputs/outputs:
- Make sure you use the functions `write_input_file` and `write_output_file` provided
- Run the functions `read_input_file` and `read_output_file` to validate your files before submitting!
  - These are the functions run by the autograder to validate submissions

## To run the program:

There are three main ways of running the code. If you have a single file that you want to run it on, use

```
python3 help_horizon.py <input>.in <output>.out
```

If you have more than one input file, you can run it on all of them with

```
python3 help_horizon.py <inputs> <outputs>
```

for existing directories `<inputs>` and `<outputs>`. Finally, if you want to run it on just a portion of the inputs in the directory, use

```
python3 help_horizon.py <inputs> <outputs> --low lo --high hi
```

where `lo` and `hi` are integral upper and lower limits. For example, if `lo == 10` and `hi == 20`, this will run the 10th through 19th, inclusive, files in `<inputs>`. Numbers outside the possible range will default to the entire directory.
