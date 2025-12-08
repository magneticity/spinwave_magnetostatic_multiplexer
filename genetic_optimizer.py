"""
Genetic Algorithm Optimizer for Magnonic Multiplexer
Optimizes FeRh dot positions to maximize frequency-selective routing
"""

import numpy as np
import json
import shutil
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Callable


class GeneticOptimizer:
    """
    Genetic algorithm to optimize dot positions for magnonic multiplexer.
    
    Parameters
    ----------
    population_size : int
        Number of individuals per generation
    n_dots : int
        Number of FeRh dots to optimize
    x_range : tuple
        (min, max) x-coordinate bounds in meters
    y_range : tuple
        (min, max) y-coordinate bounds in meters
    mutation_rate : float
        Probability of mutation (0-1)
    mutation_std : float
        Standard deviation for position mutations in meters
    crossover_rate : float
        Probability of crossover (0-1)
    elite_fraction : float
        Fraction of top individuals to preserve
    """
    
    def __init__(self, population_size=10, n_dots=8, 
                 x_range=(-2.4e-6, -0.1e-6), y_range=(-0.4e-6, 0.4e-6),
                 mutation_rate=0.2, mutation_std=100e-9, 
                 crossover_rate=0.7, elite_fraction=0.2):
        
        self.population_size = population_size
        self.n_dots = n_dots
        self.x_range = x_range
        self.y_range = y_range
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.crossover_rate = crossover_rate
        self.n_elite = max(1, int(population_size * elite_fraction))
        
        # Storage for current generation
        self.population = []  # List of individuals (each is list of (x,y) tuples)
        self.fitness_scores = []  # Fitness for each individual
        self.generation = 0
        
        # History tracking
        self.history = {
            'generations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'best_individual': []
        }
    
    def initialize_population(self, seed=None):
        """Generate random initial population."""
        if seed is not None:
            np.random.seed(seed)
        
        self.population = []
        for _ in range(self.population_size):
            individual = []
            for _ in range(self.n_dots):
                x = self.x_range[0] + np.random.random() * (self.x_range[1] - self.x_range[0])
                y = self.y_range[0] + np.random.random() * (self.y_range[1] - self.y_range[0])
                individual.append((x, y))
            self.population.append(individual)
        
        print(f"Initialized population of {self.population_size} individuals")
        print(f"Each individual has {self.n_dots} dots")
    
    def set_fitness_scores(self, scores):
        """
        Set fitness scores for current population.
        
        Parameters
        ----------
        scores : list
            Fitness score for each individual (higher is better)
        """
        if len(scores) != self.population_size:
            raise ValueError(f"Expected {self.population_size} scores, got {len(scores)}")
        
        self.fitness_scores = np.array(scores)
        
        # Update history
        self.history['generations'].append(self.generation)
        self.history['best_fitness'].append(float(np.max(self.fitness_scores)))
        self.history['mean_fitness'].append(float(np.mean(self.fitness_scores)))
        
        best_idx = np.argmax(self.fitness_scores)
        self.history['best_individual'].append(self.population[best_idx])
        
        print(f"\nGeneration {self.generation} complete:")
        print(f"  Best fitness:  {np.max(self.fitness_scores):.4f}")
        print(f"  Mean fitness:  {np.mean(self.fitness_scores):.4f}")
        print(f"  Worst fitness: {np.min(self.fitness_scores):.4f}")
    
    def select_parents(self):
        """Tournament selection - pick 2 random, choose better one."""
        idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
        if self.fitness_scores[idx1] > self.fitness_scores[idx2]:
            return self.population[idx1]
        else:
            return self.population[idx2]
    
    def crossover(self, parent1, parent2):
        """
        Single-point crossover between two parents.
        
        Parameters
        ----------
        parent1, parent2 : list
            Parent individuals (lists of (x,y) tuples)
        
        Returns
        -------
        child1, child2 : list
            Two offspring
        """
        if np.random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        crossover_point = np.random.randint(1, self.n_dots)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, individual):
        """
        Mutate individual by perturbing dot positions.
        
        Parameters
        ----------
        individual : list
            Individual to mutate (list of (x,y) tuples)
        
        Returns
        -------
        mutated : list
            Mutated individual
        """
        mutated = []
        for x, y in individual:
            if np.random.random() < self.mutation_rate:
                # Add Gaussian noise to position
                x_new = x + np.random.normal(0, self.mutation_std)
                y_new = y + np.random.normal(0, self.mutation_std)
                
                # Clip to valid range
                x_new = np.clip(x_new, self.x_range[0], self.x_range[1])
                y_new = np.clip(y_new, self.y_range[0], self.y_range[1])
                
                mutated.append((x_new, y_new))
            else:
                mutated.append((x, y))
        
        return mutated
    
    def evolve(self):
        """Create next generation using selection, crossover, and mutation."""
        if len(self.fitness_scores) == 0:
            raise ValueError("Must set fitness scores before evolving")
        
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        
        # New population starts with elite individuals (exact copies)
        new_population = sorted_population[:self.n_elite]
        
        # Generate offspring to fill rest of population
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutate
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Update population
        self.population = new_population[:self.population_size]
        self.generation += 1
        self.fitness_scores = []  # Clear scores for next generation
        
        print(f"\nEvolved to generation {self.generation}")
        print(f"  Elite preserved: {self.n_elite}")
        print(f"  New offspring:   {self.population_size - self.n_elite}")
    
    def get_current_population(self):
        """Return current population as list of dot position lists."""
        return [individual.copy() for individual in self.population]
    
    def get_best_individual(self):
        """Return best individual from current generation."""
        if len(self.fitness_scores) == 0:
            raise ValueError("No fitness scores available")
        
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx].copy(), self.fitness_scores[best_idx]
    
    def save_checkpoint(self, filepath):
        """Save optimizer state to JSON file."""
        checkpoint = {
            'generation': self.generation,
            'population': [[list(pos) for pos in ind] for ind in self.population],
            'fitness_scores': self.fitness_scores.tolist() if len(self.fitness_scores) > 0 else [],
            'history': {
                'generations': self.history['generations'],
                'best_fitness': self.history['best_fitness'],
                'mean_fitness': self.history['mean_fitness'],
                'best_individual': [[list(pos) for pos in ind] for ind in self.history['best_individual']]
            },
            'config': {
                'population_size': self.population_size,
                'n_dots': self.n_dots,
                'x_range': list(self.x_range),
                'y_range': list(self.y_range),
                'mutation_rate': self.mutation_rate,
                'mutation_std': self.mutation_std,
                'crossover_rate': self.crossover_rate,
                'n_elite': self.n_elite
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load optimizer state from JSON file."""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.generation = checkpoint['generation']
        self.population = [[tuple(pos) for pos in ind] for ind in checkpoint['population']]
        self.fitness_scores = np.array(checkpoint['fitness_scores'])
        
        self.history = {
            'generations': checkpoint['history']['generations'],
            'best_fitness': checkpoint['history']['best_fitness'],
            'mean_fitness': checkpoint['history']['mean_fitness'],
            'best_individual': [[tuple(pos) for pos in ind] for ind in checkpoint['history']['best_individual']]
        }
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resumed at generation {self.generation}")


def cleanup_simulation_data(sim_name):
    """
    Delete simulation output files to save space.
    
    Parameters
    ----------
    sim_name : str
        Name of simulation (e.g., 'spinwave_device_dots')
    """
    # Remove .out directory
    out_dir = Path(f"{sim_name}.out")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    # Remove .mat file
    mat_file = Path(f"{sim_name}.mat")
    if mat_file.exists():
        mat_file.unlink()
    
    # Remove .txt script file
    txt_file = Path(f"{sim_name}.txt")
    if txt_file.exists():
        txt_file.unlink()


def get_available_gpus():
    """
    Detect available CUDA GPUs.
    
    Returns
    -------
    n_gpus : int
        Number of available GPUs
    """
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            n_gpus = len([line for line in result.stdout.strip().split('\n') if line])
            return n_gpus
    except:
        pass
    
    # Fallback: check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        return len(cuda_visible.split(','))
    
    # Default to 1 GPU if detection fails
    return 1


def run_simulation_on_gpu(args):
    """
    Worker function to run a single simulation on a specific GPU.
    This function is called by the parallel executor.
    
    Parameters
    ----------
    args : tuple
        (dot_positions, sim_name, gpu_id, params)
    
    Returns
    -------
    result : dict
        Dictionary containing fitness score and any other metrics
    """
    dot_positions, sim_name, gpu_id, params = args
    
    # Set CUDA_VISIBLE_DEVICES to use only this GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import evaluation function from worker module
    from simulation_worker import evaluate_individual
    
    # Run the evaluation
    result = evaluate_individual(dot_positions, sim_name, params)
    
    # Clean up simulation data
    cleanup_simulation_data(sim_name)
    
    return result


class ParallelEvaluator:
    """
    Handles parallel evaluation of population across multiple GPUs.
    
    Parameters
    ----------
    n_workers : int, optional
        Number of parallel workers. If None, uses number of available GPUs
    """
    
    def __init__(self, n_workers: int = None):
        if n_workers is None:
            self.n_workers = get_available_gpus()
        else:
            self.n_workers = n_workers
        
        print(f"Parallel evaluator initialized with {self.n_workers} workers")
    
    def evaluate_population(self, population: List[List[Tuple]], 
                          generation: int, params: dict) -> List[float]:
        """
        Evaluate entire population in parallel across GPUs.
        
        Parameters
        ----------
        population : list
            List of individuals (each is list of (x,y) tuples)
        generation : int
            Current generation number
        
        Returns
        -------
        fitness_scores : list
            Fitness score for each individual
        """
        # Prepare arguments for each simulation
        tasks = []
        for idx, dot_positions in enumerate(population):
            sim_name = f"opt_gen{generation}_ind{idx}"
            gpu_id = idx % self.n_workers  # Round-robin GPU assignment
            tasks.append((dot_positions, sim_name, gpu_id, params))
        
        fitness_scores = [None] * len(population)
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(run_simulation_on_gpu, task): idx 
                for idx, task in enumerate(tasks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    fitness_scores[idx] = result['fitness']
                    print(f"  Individual {idx+1}/{len(population)} complete: "
                          f"Fitness = {result['fitness']:.4f} "
                          f"(GPU {idx % self.n_workers})")
                except Exception as e:
                    print(f"  Individual {idx+1} failed: {e}")
                    fitness_scores[idx] = 0.0  # Assign zero fitness on failure
        
        return fitness_scores
