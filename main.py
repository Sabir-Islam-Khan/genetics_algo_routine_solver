import random
import matplotlib.pyplot as plt
import numpy as np

# Data
rooms = [{"name": "Room1"}, {"name": "Room2"}, {"name": "Room3"}, {"name": "Room4"}, {"name": "Room5"}]
num_days = 7
slots_per_day = 7
sections = [
    {"name": "64_A", "subjects": ["ENG101", "CSE112", "CSE114"]},
    {"name": "64_B", "subjects": ["CSE112", "ENG101", "CSE113"]},
    {"name": "64_C", "subjects": ["CSE113", "ENG101", "CSE114"]},
    {"name": "64_D", "subjects": ["CSE114", "CSE112", "ENG101"]},
    {"name": "64_E", "subjects": ["CSE114", "CSE113", "ENG101"]},
]
teachers = [
    {"name": "SAH", "subjects": ["ENG101", "CSE112"]},
    {"name": "SIK", "subjects": ["CSE112", "ENG101"]},
    {"name": "SIT", "subjects": ["CSE113", "CSE114"]},
    {"name": "WAS", "subjects": ["CSE114"]},
    {"name": "MHS", "subjects": ["ENG101"]},
]

# Constants
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.1

# Helper function to get the teacher for a subject
def get_teacher_for_subject(subject):
    for teacher in teachers:
        if subject in teacher["subjects"]:
            return teacher["name"]
    return None

def create_individual():
    individual = []
    for section in sections:
        for subject in section["subjects"]:
            for _ in range(2):  # Each subject has two classes per week
                room = random.choice(rooms)
                day = random.randint(0, num_days - 1)
                slot = random.randint(0, slots_per_day - 1)
                teacher = get_teacher_for_subject(subject)
                individual.append((section["name"], subject, room["name"], day, slot, teacher))
    return individual

def calculate_fitness(individual):
    conflicts = 0
    schedule = {}
    teacher_schedule = {}
    
    for section, subject, room, day, slot, teacher in individual:
        if (room, day, slot) in schedule:
            conflicts += 1
        else:
            schedule[(room, day, slot)] = (section, subject)
        
        if (teacher, day, slot) in teacher_schedule:
            conflicts += 1
        else:
            teacher_schedule[(teacher, day, slot)] = (section, subject)
    
    section_schedule = {}
    for section, subject, room, day, slot, teacher in individual:
        if (section, day, slot) in section_schedule:
            conflicts += 1
        else:
            section_schedule[(section, day, slot)] = (subject, room)
    
    return -conflicts

def mutate(individual):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(individual) - 1)
        section, subject, room, day, slot, teacher = individual[idx]
        new_room = random.choice(rooms)["name"]
        new_day = random.randint(0, num_days - 1)
        new_slot = random.randint(0, slots_per_day - 1)
        individual[idx] = (section, subject, new_room, new_day, new_slot, teacher)
    return individual

def crossover(parent1, parent2):
    idx = random.randint(0, len(parent1) - 1)
    child1 = parent1[:idx] + parent2[idx:]
    child2 = parent2[:idx] + parent1[idx:]
    return child1, child2

def select(population):
    fitness_values = [calculate_fitness(ind) for ind in population]
    min_fitness = abs(min(fitness_values))
    adjusted_fitness = [f + min_fitness + 1 for f in fitness_values]  # Ensure all fitness values are positive
    return random.choices(population, k=2, weights=adjusted_fitness)

def genetic_algorithm():
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))
        population = new_population
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {max([calculate_fitness(ind) for ind in population])}")

    best_individual = max(population, key=calculate_fitness)
    return best_individual

def visualize_schedule(individual):
    fig, axs = plt.subplots(nrows=len(rooms), ncols=num_days, figsize=(20, 10))
    for section, subject, room, day, slot, teacher in individual:
        room_idx = [r["name"] for r in rooms].index(room)
        axs[room_idx, day].text(0.5, 0.5, f"{section}\n{subject}\n{teacher}", ha="center", va="center", fontsize=10)
        axs[room_idx, day].set_xticks([])
        axs[room_idx, day].set_yticks([])
        axs[room_idx, day].set_title(f"Day {day + 1}", fontsize=12)
    
    for ax, room in zip(axs[:, 0], rooms):
        ax.set_ylabel(room["name"], fontsize=12, rotation=0, labelpad=20)
    
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

# Run the genetic algorithm and visualize the best schedule
best_schedule = genetic_algorithm()
visualize_schedule(best_schedule)
