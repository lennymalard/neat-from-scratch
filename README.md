# NEAT Algorithm from Scratch in Python

 
<p align="center">
  <img src="/assets/snake_playing.gif" alt="GIF" />
</p>

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/lennymalard/neat-from-scratch/blob/main/LICENSE)

## Overview

This repository contains a Python implementation of the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm, built entirely from scratch. NEAT is a powerful evolutionary algorithm designed to evolve artificial neural networks (ANNs) by optimizing both their weights and their structure (topology) simultaneously.

This implementation aims to provide a clear and understandable foundation for NEAT, suitable for learning, experimentation, and extension. It includes the core components of NEAT, such as speciation, innovation tracking, and complexification of network structures.

## A Brief History of NEAT

NEAT was introduced by Kenneth O. Stanley and Risto Miikkulainen in their 2002 paper, "Evolving Neural Networks through Augmenting Topologies." It addressed key challenges in neuroevolution:

1.  **The Competing Conventions Problem:** Different network structures could represent the same solution, making crossover difficult. NEAT uses historical markings (innovation numbers) to align genes effectively.
2.  **Structural Innovation Protection:** Adding new structure often initially decreases fitness. NEAT uses speciation to allow new innovations time to optimize within a protected niche before competing globally.
3.  **Minimizing Topology:** NEAT starts with minimal network structures and complexifies them over generations, searching for solutions efficiently without needing to guess the optimal topology beforehand.

## NEAT Basics

The core ideas behind NEAT are:

1.  **Genetic Encoding:** Each ANN (genome) is represented by a list of node genes and connection genes.
2.  **Innovation Numbers:** Every new structural addition (a new node or connection type) receives a unique, global innovation number. This allows NEAT to track the historical origin of genes and perform meaningful crossover.
3.  **Initial Minimal Structure:** Evolution starts with simple genomes having only input and output nodes, fully connected.
4.  **Mutation:** Genomes evolve through mutations:
    *   **Weight Mutations:** Adjusting connection weights (perturbation or reset).
    *   **Bias Mutations:** Adjusting node biases (perturbation or reset).
    *   **Add Connection Mutation:** Adding a new connection between existing nodes.
    *   **Add Node Mutation:** Splitting an existing connection by inserting a new node.
    *   *(Optional: Remove Connection/Node Mutations)*
5.  **Speciation:** Genomes are grouped into species based on genetic similarity (compatibility distance). This protects novel structures by forcing them to compete primarily within their own species initially.
6.  **Fitness Sharing:** Fitness is adjusted (shared) among members of the same species to prevent single large species from dominating the population unfairly.
7.  **Reproduction:** Offspring are generated primarily through crossover and mutation within species, with fitter species allocated more offspring slots. Elitism preserves the best individuals.

## Features of This Implementation

This implementation incorporates several standard and specific NEAT techniques:

*   **Clear Object-Oriented Structure:** Uses classes like `Node`, `Connection`, `Genome`, `Species`, and `Population` for modularity.
*   **Innovation Tracking:** Uses global counters (`conn_id`, `node_id`) and a dictionary (`conn_genes`) in the `Population` class to assign and track historical markers for connections.
*   **Feed-Forward Network Evaluation:** Network activation is performed using **topological sorting** (`Genome.topological_sort`) based on enabled connections, ensuring correct evaluation order even in complex topologies. This handles non-layered networks efficiently.
*   **Compatibility Distance:** Uses the standard NEAT formula (`Population.calculate_compatibility`) based on excess genes (E), disjoint genes (D), and average weight differences (W), weighted by coefficients `c1`, `c2`, and `c3`.
    *   The normalization factor `N` is the number of genes in the larger genome (`max(1, max(n1, n2))`), following a common practice. (Note: Some variants adjust `N` for very small genomes, but this implementation uses the direct maximum size).
*   **Speciation:** Genomes are assigned to species based on comparing their compatibility distance to the species representative against the `species_threshold` (`Population.speciate`).
*   **Optional Adaptive Species Threshold:** The `species_threshold` can be dynamically adjusted (`Population.adjust_species_threshold`) based on the current number of species relative to a target (`target_species_number`), using a step size (`adaptive_threshold`). This helps maintain a desired level of diversity.
*   **Configurable Activation Functions:** Supports multiple activation functions (`relu`, `sigmoid`, `tanh`) for hidden and output nodes, selectable via the `NEATConfig` class.
*   **Detailed Mutation Operators:** Includes standard mutations for adding nodes (`Genome.add_node_mutation`), adding connections (`Genome.add_connection_mutation` with cycle checking), removing connections (`Genome.remove_connection_mutation`), removing nodes (`Genome.remove_node_mutation`), and mutating weights/biases (`Genome.weight_mutation`, `Genome.bias_mutation`) with Gaussian perturbation and random reset options.
*   **Explicit Fitness Sharing:** Fitness is adjusted within species by dividing by the number of members (`Species.adjust_fitness`). Optional linear scaling (`Species.linear_scale_fitness`) and offsetting (`Species.offset_fitness`) are also included.
*   **Multiprocessing Support (in Example):** The Snake example demonstrates using Python's `multiprocessing` module for parallel genome evaluation, significantly speeding up training.

## The Reproduction Function (`Population.reproduce`)

The reproduction process is central to NEAT and arguably the most complex part. Here's how it works in this implementation:

1.  **Fitness Adjustment:**
    *   **(Optional) Linear Scaling:** Fitness values within each species can be scaled linearly to potentially increase selection pressure (`Species.linear_scale_fitness`).
    *   **(Optional) Offset Fitness:** Fitness values can be offset to ensure they are positive, useful for some selection mechanisms (`Species.offset_fitness`).
    *   **Fitness Sharing:** Each genome's fitness is divided by the number of individuals in its species (`Species.adjust_fitness`). This prevents large species from overwhelming smaller ones purely due to size.
    *   **Ranking:** Members within each species are ranked based on their adjusted fitness (`Species.rank`).

2.  **Offspring Allocation:**
    *   The average adjusted fitness of each species is calculated.
    *   The total number of offspring for the next generation (`population_size`) is allocated proportionally to each species based on its relative average adjusted fitness compared to the global total average fitness.
    *   Rounding errors are handled by distributing the remaining slots (or removing excess slots) one by one, often prioritizing species with higher/lower allocations depending on whether adding or removing.

3.  **Generating the New Population:**
    *   **Elitism:** For each species, a predefined number (`num_elites`) of the highest-ranking genomes are copied directly (without mutation) into the next generation, ensuring the best solutions found so far are preserved.
    *   **Parent Selection:** From the remaining top fraction (`selection_share`) of each species (the "selection pool"), parents are chosen for crossover.
    *   **Crossover:** Two parents are selected from the pool. The `Population.cross_over` method combines their genes:
        *   The offspring inherits all genes present in the *fitter* parent.
        *   For *matching* genes (same innovation ID), the gene is chosen randomly from either parent.
        *   If a matching gene was disabled in *either* parent, there's a high probability (75% here) it will remain disabled in the offspring.
        *   The offspring inherits the node structure primarily from the fitter parent.
        *   A fallback mechanism exists: if crossover somehow results in an invalid structure (e.g., during internal checks, although cycle checks are primarily mutation-time), a mutated copy of the fitter parent might be used instead.
    *   **Mutation:** *All* newly created offspring (from crossover) are subjected to mutation via the `Genome.mutate` method, applying weight, bias, and structural mutations based on configured probabilities.
    *   **Population Size Management:** If the allocation and generation process results in slightly fewer or more individuals than the target `population_size`, random genomes are added or the list is truncated to ensure the correct size.

4.  **Speciation:**
    *   The old species structure is cleared (members are removed, but representatives might be temporarily remembered).
    *   Each genome in the new population is re-evaluated and assigned to a species using the `Population.speciate` method. It either joins an existing compatible species or forms a new one.
    *   Empty species are removed. Representatives are either kept if they survived or reassigned randomly from the remaining members.

5.  **Threshold Adjustment:** If adaptive thresholding is enabled, the `species_threshold` is adjusted based on the new number of species.

## Configuration (`NEATConfig`)

The `NEATConfig` class provides a convenient way to manage all the hyperparameters for the NEAT algorithm. Instead of passing dozens of arguments, you can instantiate this class and modify its attributes. Key configurable parameters include:

*   `population_size`: Number of genomes in the population.
*   `genome_shape`: Tuple `(num_inputs, num_outputs)` for initial genomes.
*   `hid_node_activation`, `out_node_activation`: Activation functions ('relu', 'sigmoid', 'tanh').
*   `add_node_mutation_prob`, `add_conn_mutation_prob`, etc.: Probabilities for structural mutations.
*   `sigma`, `perturb_prob`, `reset_prob`: Parameters for weight/bias mutations.
*   `species_threshold`: The compatibility distance threshold.
*   `adaptive_threshold`, `target_species_number`, `min/max_species_threshold`: Parameters for dynamic threshold adjustment.
*   `c1`, `c2`, `c3`: Coefficients for the compatibility distance formula.
*   `num_elites`: Number of elites preserved per species.
*   `selection_share`: Fraction of top individuals per species eligible for reproduction.
*   `save_path`: Directory to save results (like the best genome).

You can print a `NEATConfig` instance to see all current settings.

## Examples

Two examples are provided to demonstrate the usage of this NEAT implementation:

1.  **`xor.py`**:
    *   A classic benchmark task where NEAT evolves a network to solve the XOR (exclusive OR) problem.
    *   Demonstrates basic setup and evaluation.
    *   Input: `[x1, x2, bias_input (1.0)]`
    *   Output: `[xor_result]`
    *   Run using: `python xor.py`

2.  **`snake.py`**:
    *   A more complex control task where NEAT evolves an agent to play the game of Snake.
    *   Features two types of input representations for the snake:
        *   Absolute state (danger left/right/forward, apple direction, current direction).
        *   Ray casting vision (detecting wall/body/apple in 8 directions). Selectable via `use_ray_cast` flag in `SnakeGame`.
    *   Includes a Pygame visualization (`neat_play`) to watch the best genome from each specified generation interval play the game.
    *   Uses `multiprocessing` for parallel fitness evaluation, significantly speeding up training.
    *   Run using: `python snake.py`

## Future Work

*   **Stagnation Handling:** Implement species stagnation detection, where species that haven't improved for a certain number of generations are penalized or removed to prevent bloating the population with non-progressing niches.

## Reference

This implementation is based on the principles described in the original NEAT paper:

Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting Topologies. *Evolutionary Computation*, 10(2), 99-127.
[Link to Paper (MIT Press)](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lennymalard/neat-from-scratch
    cd neat-from-scratch
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pygame matplotlib
    ```
    (Pygame is needed for `snake.py`, Matplotlib for plotting fitness in `snake.py` and NumPy for identifying the indices of maximum output values in `snake.py`).
3.  **Run the examples:**
    ```bash
    python xor.py
    python snake.py
    ```
4.  **Integrate into your own project:**
    *   Import `NEATConfig`, `Population` from `neat.py`.
    *   Define your own fitness evaluation function that takes a `Genome` object as input and returns a fitness score (or sets `genome.fitness`).
    *   Configure `NEATConfig` according to your problem.
    *   Create a `Population` instance.
    *   Run the evolutionary loop: evaluate fitness for all genomes, then call `population.reproduce()`.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/lennymalard/neat-from-scratch/blob/main/LICENSE) file for details.
