import numpy as np
# 1
# Parametry algorytmu genetycznego
POPULATION_SIZE = 80
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.3
MAX_GENERATIONS = 70

# Dane dotyczące klientów i macierz kosztów podróży
customers = {
    'K1': (1, 2),
    'K2': (5, 6),
    'K3': (8, 3),
    'K4': (3, 10),
    'K5': (12, 5),
    'K6': (7, 8),
    'K7': (2, 15),
    'K8': (9, 12),
    'K9': (6, 4),
    'K10': (11, 9)
}
# 2
# Inicjalizacja populacji
def initialize_population(size):
    return [np.random.permutation(list(customers.keys())) for _ in range(size)]


# 3
# Funkcja celu (koszt podróży)
def objective_function(chromosome):
    total_cost = 0
    for i in range(len(chromosome) - 1):
        current_customer = chromosome[i]
        next_customer = chromosome[i + 1]
        total_cost += calculate_travel_cost(customers[current_customer], customers[next_customer])
    return total_cost


# Funkcja obliczająca koszt podróży między dwoma klientami
def calculate_travel_cost(location1, location2):
    return np.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)


# 4
# Operatory genetyczne
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate(
        (parent1[:crossover_point], [gene for gene in parent2 if gene not in parent1[:crossover_point]]))
    child2 = np.concatenate(
        (parent2[:crossover_point], [gene for gene in parent1 if gene not in parent2[:crossover_point]]))
    return child1, child2


def mutate(chromosome):
    mutation_point1, mutation_point2 = np.random.choice(len(chromosome), 2, replace=False)
    mutated_chromosome = np.copy(chromosome)
    mutated_chromosome[mutation_point1], mutated_chromosome[mutation_point2] = mutated_chromosome[mutation_point2], \
    mutated_chromosome[mutation_point1]
    return mutated_chromosome


# 5/6
# Algorytm genetyczny
def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE)

    for generation in range(MAX_GENERATIONS):
        # Ocena funkcji celu
        fitness_values = [objective_function(chromosome) for chromosome in population]

        # Wybór rodziców
        selected_indices = np.argsort(fitness_values)[:int(CROSSOVER_RATE * POPULATION_SIZE)]
        parents = [population[i] for i in selected_indices]

        # Krzyżowanie
        offspring = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([child1, child2])

        # Mutacja
        mutated_offspring = [mutate(child) if np.random.rand() < MUTATION_RATE else child for child in offspring]

        # Zastąpienie starej populacji nowym pokoleniem
        population = mutated_offspring

    # Znalezienie najlepszego rozwiązania
    best_solution = min(population, key=objective_function)
    best_cost = objective_function(best_solution)

    return best_solution, best_cost


# Uruchomienie algorytmu genetycznego
best_route, best_cost = genetic_algorithm()

# 7
# Wyświetlenie wyników
print("Najlepsza trasa:", best_route)
print("Koszt podróży:", best_cost)
