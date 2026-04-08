"""Genetic algorithm to fit a cubic polynomial to noisy observations."""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "wine_alcohol_quality.csv"
PLOT_PATH = Path(__file__).resolve().parents[1] / "results" / "ga_fit_plot.png"
METRIC_PATH = Path(__file__).resolve().parents[1] / "results" / "ga_metrics.json"


@dataclass
class GAConfig:
    population_size: int = 80
    generations: int = 300
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    mutation_scale: float = 0.5
    coefficient_bounds: Tuple[float, float] = (-5.0, 5.0)


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return np.array(xs), np.array(ys)


def random_individual(bounds: Tuple[float, float]) -> np.ndarray:
    low, high = bounds
    return np.array([random.uniform(low, high) for _ in range(4)])


def evaluate_mse(individual: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    a, b, c, d = individual
    preds = ((a * xs + b) * xs + c) * xs + d  # Horner form for stability
    return float(np.mean((preds - ys) ** 2))


def tournament_selection(population: List[np.ndarray], fitness: List[float], k: int = 3) -> np.ndarray:
    contenders = random.sample(list(zip(population, fitness)), k)
    winner = min(contenders, key=lambda item: item[1])[0]
    return winner.copy()


def blend_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2


def mutate(individual: np.ndarray, cfg: GAConfig) -> np.ndarray:
    for i in range(len(individual)):
        if random.random() < cfg.mutation_rate:
            individual[i] += random.gauss(0, cfg.mutation_scale)
            individual[i] = max(cfg.coefficient_bounds[0], min(cfg.coefficient_bounds[1], individual[i]))
    return individual


def run_ga(xs: np.ndarray, ys: np.ndarray, cfg: GAConfig) -> Tuple[np.ndarray, float, List[float]]:
    population = [random_individual(cfg.coefficient_bounds) for _ in range(cfg.population_size)]
    fitness = [evaluate_mse(ind, xs, ys) for ind in population]

    best_ind, best_fit = min(zip(population, fitness), key=lambda item: item[1])
    best_ind = best_ind.copy()
    history: List[float] = [best_fit]

    for gen in range(cfg.generations):
        new_population: List[np.ndarray] = []
        while len(new_population) < cfg.population_size:
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            if random.random() < cfg.crossover_rate:
                child1, child2 = blend_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            new_population.extend([mutate(child1, cfg), mutate(child2, cfg)])
        population = new_population[:cfg.population_size]
        fitness = [evaluate_mse(ind, xs, ys) for ind in population]
        current_best_ind, current_best_fit = min(zip(population, fitness), key=lambda item: item[1])
        if current_best_fit < best_fit:
            best_ind, best_fit = current_best_ind.copy(), current_best_fit
        history.append(best_fit)
        if gen % 50 == 0:
            print(f"Generation {gen}: MSE={best_fit:.4f}")
    return best_ind, best_fit, history


def plot_results(xs: np.ndarray, ys: np.ndarray, coeffs: np.ndarray) -> None:
    xs_dense = np.linspace(xs.min(), xs.max(), 200)
    preds = np.polyval(coeffs, xs_dense)
    plt.figure(figsize=(8, 4))
    plt.scatter(xs, ys, label="Data", color="#1f77b4")
    plt.plot(xs_dense, preds, label="GA Fit", color="#d62728")
    plt.title("Genetic Algorithm Cubic Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()


def plot_training_history(history: List[float]) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.plot(history, color="#17becf")
    plt.title("GA Best Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best MSE")
    plt.tight_layout()
    out_path = PLOT_PATH.with_name("ga_training_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_residuals(xs: np.ndarray, ys: np.ndarray, coeffs: np.ndarray) -> None:
    preds = np.polyval(coeffs, xs)
    residuals = ys - preds
    plt.figure(figsize=(6, 3.5))
    plt.scatter(xs, residuals, s=12, alpha=0.6, color="#ff7f0e")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.title("Residuals vs Alcohol Content")
    plt.xlabel("Alcohol (%)")
    plt.ylabel("Residual (quality)")
    plt.tight_layout()
    out_path = PLOT_PATH.with_name("ga_residuals.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_r_squared(preds: np.ndarray, ys: np.ndarray) -> float:
    ss_res = float(np.sum((ys - preds) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return 1 - ss_res / ss_tot


def main() -> None:
    cfg = GAConfig()
    xs, ys = load_dataset(DATA_PATH)

    best_coeffs, best_mse, history = run_ga(xs, ys, cfg)
    baseline_coeffs = np.polyfit(xs, ys, 3)
    baseline_mse = evaluate_mse(baseline_coeffs, xs, ys)
    preds = np.polyval(best_coeffs, xs)
    r_squared = compute_r_squared(preds, ys)

    plot_results(xs, ys, best_coeffs)
    plot_training_history(history)
    plot_residuals(xs, ys, best_coeffs)

    METRIC_PATH.write_text(json.dumps(
        {
            "ga_coefficients": best_coeffs.tolist(),
            "ga_mse": best_mse,
            "ga_r2": r_squared,
            "ga_history": history,
            "polyfit_coefficients": baseline_coeffs.tolist(),
            "polyfit_mse": baseline_mse,
        },
        indent=2,
    ))
    print("Best coefficients (GA):", best_coeffs)
    print("Best MSE (GA):", best_mse)
    print("Baseline coefficients (polyfit):", baseline_coeffs)
    print("Baseline MSE (polyfit):", baseline_mse)


if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    main()
