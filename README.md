## Quick Start

- Open the notebook `NeuroML_to_Jaxley_Tutorial.ipynb` in Jupyter (or VS Code/Jupyter).
- Follow the cells in order; each section mirrors the overview below.

Notebook: [NeuroML_to_Jaxley_Tutorial.ipynb](NeuroML_to_Jaxley_Tutorial.ipynb)

## Tutorial Overview

### Part 1: Model Specification
First, it is important to specify the neural model assumptions in Jaxley in the form of cell, channel, and synapse dynamics. Because the C. elegans worm has unique properties, we show how to implement a custom Calcium channel and graded synapse as specified by the c302 models, while using the existing Potassium channels in Jaxley.

### Part 2: Connectome Translation
The next step is to translate from the connectome format into the Jaxley network. Because the model is specified, all the cell and synapse parameters are converted into a JAX-enabled model.

### Part 3: Simulation and Optimization
Lastly, we simulate, visualize, and perform a simple gradient based optimization on network parameters and stimulus input. Gradients can be accessed with any neurally encoded loss function that is itself differentiable.
