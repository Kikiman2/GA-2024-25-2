import tkinter as tk
import random
import math
import numpy as np
from joblib import Parallel, delayed

# Constants
WINDOW_WIDTH = 600  
WINDOW_HEIGHT = 400
POINT_RADIUS = 3
NUMBER_OF_POINTS = 50
POPULATION_SIZE = 60
GENERATIONS = 1001
MUTATION_RATE = 0.15
ELITE_RATE = 0.1

# Lists to store x and y coordinates
x_coords = []
y_coords = []

def draw_random_points(canvas):
    canvas.delete("all")  # Clear previous points
    x_coords.clear()
    y_coords.clear()

    for _ in range(NUMBER_OF_POINTS):
        x = random.randint(10, WINDOW_WIDTH-10)
        y = random.randint(10, WINDOW_HEIGHT-10)
        x_coords.append(x)
        y_coords.append(y)
        canvas.create_oval(
            x - POINT_RADIUS, y - POINT_RADIUS,
            x + POINT_RADIUS, y + POINT_RADIUS,
            fill="blue", outline=""
        )

def calculate_distance_matrix():
    """Calculate the distance matrix for all points."""
    num_points = len(x_coords)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            distance_matrix[i][j] = math.hypot(x_coords[j] - x_coords[i], y_coords[j] - y_coords[i])
    return distance_matrix

def initialize_population(population_size, num_points):
    """Initialize the population with random permutations of points."""
    return [np.random.permutation(num_points) for _ in range(population_size)]

def do_lines_intersect(p1, p2, p3, p4):
    """Check if two line segments intersect.""" 
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def calculate_fitness(population, distance_matrix):
    """Calculate fitness as the inverse of the total distance with a penalty for intersections."""
    fitness_scores = []
    for individual in population:
        indices = np.append(individual, individual[0])  # Add the starting point to complete the loop
        distance = np.sum(distance_matrix[indices[:-1], indices[1:]])
        intersections = 0

        # Check for intersections
        for i in range(len(individual) - 1):
            p1 = (x_coords[individual[i]], y_coords[individual[i]])
            p2 = (x_coords[individual[i + 1]], y_coords[individual[i + 1]])
            for j in range(i + 2, len(individual) - 1):
                if i == 0 and j == len(individual) - 2:
                    continue
                p3 = (x_coords[individual[j]], y_coords[individual[j]])
                p4 = (x_coords[individual[j + 1]], y_coords[individual[j + 1]])
                if do_lines_intersect(p1, p2, p3, p4):
                    intersections += 1

        # Penalize fitness for intersections
        fitness = 1 / (distance + intersections * 5000)  # Increased penalty for intersections
        fitness_scores.append(fitness)

    return np.array(fitness_scores)

def calculate_fitness_parallel(population, distance_matrix):
    """Calculate fitness in parallel."""
    def fitness_for_individual(individual):
        indices = np.append(individual, individual[0])
        distance = np.sum(distance_matrix[indices[:-1], indices[1:]])
        intersections = 0
        for i in range(len(individual) - 1):
            p1 = (x_coords[individual[i]], y_coords[individual[i]])
            p2 = (x_coords[individual[i + 1]], y_coords[individual[i + 1]])
            for j in range(i + 2, len(individual) - 1):
                if i == 0 and j == len(individual) - 2:
                    continue
                p3 = (x_coords[individual[j]], y_coords[individual[j]])
                p4 = (x_coords[individual[j + 1]], y_coords[individual[j + 1]])
                if do_lines_intersect(p1, p2, p3, p4):
                    intersections += 1
        return 1 / (distance + intersections * 1000)

    fitness_scores = Parallel(n_jobs=-1)(delayed(fitness_for_individual)(individual) for individual in population)
    return np.array(fitness_scores)

def select_elite(population, fitness_scores, elite_rate):
    """Select the top elite individuals."""
    elite_count = int(elite_rate * len(population))
    sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort in descending order
    return [population[i] for i in sorted_indices[:elite_count].tolist()], elite_count

def perform_crossover(population, elite_count, population_size, num_points):
    """Perform crossover to generate the next generation."""
    next_generation = list(population[:elite_count])
    while len(next_generation) < population_size:
        parent1, parent2 = random.sample(population, 2)
        start, end = sorted(random.sample(range(num_points), 2))
        child = np.full(num_points, -1)
        child[start:end] = parent1[start:end]
        pointer = end
        for gene in parent2:
            if gene not in child:
                if pointer == num_points:
                    pointer = 0
                child[pointer] = gene
                pointer += 1
        next_generation.append(child)
    return next_generation

def apply_mutation(population, mutation_rate):
    """Apply mutation to the population."""
    for individual in population:
        if random.random() < mutation_rate:
            i, j = np.random.choice(len(individual), 2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
    return population

def genetic_algorithm_tsp(distance_matrix, population_size, generations, mutation_rate, elite_rate):
    """Run the genetic algorithm to minimize the total distance."""
    num_points = len(distance_matrix)
    population = initialize_population(population_size, num_points)
    best_distance = float('inf')
    best_path = None

    for generation in range(generations):
        fitness_scores = calculate_fitness_parallel(population, distance_matrix)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices.tolist()]
        fitness_scores = fitness_scores[sorted_indices]

        # Track the best solution
        best_individual = population[0]
        distance = 1 / fitness_scores[0]
        if distance < best_distance:
            best_distance = distance
            best_path = best_individual

        if generation % 50 == 0:
            print(f"Generation {generation}: Best Distance = {best_distance}, Fitness = {fitness_scores[0]}")

        # Select elite individuals
        elite_population, elite_count = select_elite(population, fitness_scores, elite_rate)

        # Perform crossover and mutation
        population = perform_crossover(elite_population, elite_count, population_size, num_points)
        population = apply_mutation(population, mutation_rate)

    print(f"Optimal Distance: {best_distance}")
    print(f"Optimal Path: {best_path}")
    return best_path, best_distance

def connect_points_with_ga(canvas):
    """Use the genetic algorithm to connect points with minimal distance."""
    distance_matrix = calculate_distance_matrix()
    best_path, best_distance = genetic_algorithm_tsp(distance_matrix, POPULATION_SIZE, GENERATIONS, MUTATION_RATE, ELITE_RATE)

    # Draw the optimal path
    for i in range(len(best_path)):
        p1 = best_path[i]
        p2 = best_path[(i + 1) % len(best_path)]
        canvas.create_line(
            x_coords[p1], y_coords[p1],
            x_coords[p2], y_coords[p2],
            fill="green", width=2
        )
    print(f"Total Distance: {best_distance}")

# Create the main window
root = tk.Tk()
root.title("Genetic Algorithm - Minimal Distance")

# Create a Canvas widget
canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
canvas.pack()

# Draw points and connect them using the genetic algorithm
draw_random_points(canvas)
connect_points_with_ga(canvas)

# Start the GUI event loop
root.mainloop()
