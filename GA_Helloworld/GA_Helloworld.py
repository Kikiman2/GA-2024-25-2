import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def initialize_population(population_size, target_length):
    return np.floor(np.random.rand(population_size, target_length) * 96 + 32).astype(int)

def calculate_fitness(population, target_ascii_values):
    return np.sum(np.abs(population - target_ascii_values), axis=1)

def select_elite(population, fitness_scores, elite_rate):
    elite_count = int(elite_rate * len(population))
    return np.copy(population[:elite_count]), elite_count

def perform_crossover(population, elite_count, population_size, target_length):
    next_generation = np.copy(population[:elite_count])
    for i in range(elite_count, population_size):
        parent1, parent2 = np.random.randint(0, population_size, 2)
        crossover_point = np.random.randint(1, target_length)
        next_generation = np.vstack([next_generation, np.concatenate([population[parent1, :crossover_point], population[parent2, crossover_point:]])])
    return next_generation

def apply_mutation(next_generation, population_size, target_length, mutation_rate):
    for _ in range(int(population_size * mutation_rate)):
        next_generation[np.random.randint(0, population_size), np.random.randint(0, target_length)] = np.random.randint(32, 128)
    return next_generation

def genetic_algorithm(population_size=30, elite_rate=0.1, mutation_rate=0.25):
    target_string = 'Genetikus algoritmus'
    max_generations = 25003
    output_interval = 10

    generation = 0
    target_ascii_values = np.array([ord(character) for character in target_string])
    #print(f'Target ASCII values:\n{pd.DataFrame(target_ascii_values)}')
    
    population = initialize_population(population_size, len(target_string))
    #print(f'Initial population:\n{pd.DataFrame(population)}')
    
    fitness_scores = calculate_fitness(population, target_ascii_values)
    #print(f'Initial fitness scores:\n{pd.DataFrame(fitness_scores)}')

    fitness_history = []

    while generation < max_generations and fitness_scores[0] != 0:
        fitness_scores, sorted_indices = np.sort(fitness_scores), np.argsort(fitness_scores)
        population = population[sorted_indices]
        best_individual = ''.join([chr(value) for value in population[0]])

        
        if generation % output_interval == 0:
            separator_column = np.full((population_size, 1), '|')
            population_with_fitness = np.hstack((population, separator_column, fitness_scores.reshape(-1, 1)))
            print(f'Generation {generation} best individual: {best_individual}   fitness: {fitness_scores[0]}')
            #print(f'Population with fitness scores:\n{pd.DataFrame(population_with_fitness)}')
        

        next_generation, elite_count = select_elite(population, fitness_scores, elite_rate)
        next_generation = perform_crossover(population, elite_count, population_size, len(target_string))
        next_generation = apply_mutation(next_generation, population_size, len(target_string), mutation_rate)

        population = next_generation
        fitness_scores = calculate_fitness(population, target_ascii_values)
        generation += 1

    fitness_scores, sorted_indices = np.sort(fitness_scores), np.argsort(fitness_scores)
    best_result = ''.join([chr(value) for value in population[sorted_indices[0]]])

    print(f"Best result: {best_result}")

    return generation

genetic_algorithm()

"""""
time.sleep(2)

population_sizes = range(10, 100, 5)
generation_history = []

for population_size in population_sizes:
    generation = genetic_algorithm(population_size=population_size)
    generation_history.append(generation)
    print(f"Population size: {population_size}, Generation: {generation}")

plt.plot(population_sizes, generation_history)
plt.xlabel('Population Size')
plt.xlim([min(population_sizes), max(population_sizes)])  # Limit the X-axis to the range of population_sizes
plt.ylabel('Generation')
plt.title('Generation Count over Population Size')
plt.savefig('generation_count_over_population_size.png')  # Save the plot as an image file
plt.show()

time.sleep(2)

elite_rates = np.arange(10, 72, 4) / 100
generation_history = []

for elite_rate in elite_rates:
    generation = genetic_algorithm(elite_rate=elite_rate)
    generation_history.append(generation)
    print(f"Elit rate: {elite_rate}, Generation: {generation}")

plt.plot(elite_rates, generation_history)
plt.xlabel('Elit rate')
plt.xlim([elite_rates[0], elite_rates[-1]])  # Limit the X-axis to the range of population_sizes
plt.ylabel('Generation')
plt.title('Generation Count over Elit rate')
plt.savefig('generation_count_over_elite_rate.png')  # Save the plot as an image file
plt.show()

time.sleep(2)

mutation_rates = np.arange(10, 38, 2) / 100
generation_history = []

for mutation_rate in mutation_rates:
    generation = genetic_algorithm(mutation_rate=mutation_rate)
    generation_history.append(generation)
    print(f"Mutation Rate: {mutation_rate}, Generation: {generation}")

plt.plot(mutation_rates, generation_history)
plt.xlabel('Mutation Rate')
plt.xlim([mutation_rates[0], mutation_rates[-1]])  # Limit the X-axis to the range of population_sizes
plt.ylabel('Generation')
plt.title('Generation Count over Mutation Rate')
plt.savefig('generation_count_over_mutation_rates.png')  # Save the plot as an image file
plt.show()"
"""""
