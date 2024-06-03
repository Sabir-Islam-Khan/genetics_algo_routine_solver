import random
import matplotlib.pyplot as plt
import numpy as np

# Data
rooms = [{"name": "Room1"}, {"name": "Room2"}]
num_days = 7
slots_per_day = 7
time_slots = [
   "8:30-9:45AM", "9:45-11:00AM", "11:00-12:15PM", "12:15-1:30PM",
   "1:30-2:45PM", "2:45-4:00PM", "4:00-5:15PM"
]
sections = [
   {"name": "64_A", "subjects": ["ENG101", "CSE112", "CSE113", "CSE114", "CSE115", "PHY101", "MAT101"]},
   {"name": "64_B", "subjects": ["ENG101", "CSE112", "CSE113", "CSE114", "CSE115", "PHY101", "MAT101"]},
   {"name": "64_C", "subjects": ["ENG101", "CSE112", "CSE113", "CSE114", "CSE115", "PHY101", "MAT101"]},
   {"name": "64_D", "subjects": ["ENG101", "CSE112", "CSE113", "CSE114", "CSE115", "PHY101", "MAT101"]},
]
teachers = [
   {"name": "SAH", "subjects": ["ENG101", "CSE112"]},
   {"name": "SIK", "subjects": ["CSE113", "CSE114"]},
   {"name": "SIT", "subjects": ["CSE115", "PHY101", "MAT101"]},
]

# Constants
POPULATION_SIZE = 100
GENERATIONS = 300
MUTATION_RATE = 0.1
MAX_CLASSES_PER_DAY = 5
MAX_DAYS_PER_WEEK = 5
MAX_ATTEMPTS = 1000

# Helper function to get the teacher for a subject
def get_teacher_for_subject(subject):
   for teacher in teachers:
       if subject in teacher["subjects"]:
           return teacher["name"]
   return None

def create_individual():
    individual = []
    used_slots = set()
    teacher_day_count = {teacher["name"]: {day: 0 for day in range(num_days)} for teacher in teachers}
    teacher_week_count = {teacher["name"]: 0 for teacher in teachers}
    section_day_subject = {section["name"]: {day: set() for day in range(num_days)} for section in sections}

    for section in sections:
        for subject in section["subjects"]:
            for _ in range(2):  # Each subject has two classes per week
                attempts = 0
                while attempts < MAX_ATTEMPTS:
                    room = random.choice(rooms)["name"]
                    day = random.randint(0, num_days - 1)
                    slot = random.randint(0, slots_per_day - 1)
                    teacher = get_teacher_for_subject(subject)
                    if ((room, day, slot) not in used_slots and
                        (teacher, day, slot) not in used_slots and
                        (section["name"], day, slot) not in used_slots and
                        teacher_day_count[teacher][day] < MAX_CLASSES_PER_DAY and
                        (teacher_week_count[teacher] < MAX_DAYS_PER_WEEK or
                         teacher_day_count[teacher][day] > 0) and
                        subject not in section_day_subject[section["name"]][day]):
                        break
                    attempts += 1
                if attempts == MAX_ATTEMPTS:
                    break  # Could not find a valid slot, skip this class
                used_slots.add((room, day, slot))
                used_slots.add((teacher, day, slot))
                used_slots.add((section["name"], day, slot))
                section_day_subject[section["name"]][day].add(subject)
                teacher_day_count[teacher][day] += 1
                if teacher_day_count[teacher][day] == 1:
                    teacher_week_count[teacher] += 1
                individual.append((section["name"], subject, room, day, slot, teacher))
    return individual

def calculate_fitness(individual):
   conflicts = 0
   schedule = {}
   teacher_schedule = {}
   section_schedule = {}
   teacher_day_count = {teacher["name"]: {day: 0 for day in range(num_days)} for teacher in teachers}
   teacher_week_count = {teacher["name"]: 0 for teacher in teachers}

   for section, subject, room, day, slot, teacher in individual:
       if (room, day, slot) in schedule:
           conflicts += 1
       else:
           schedule[(room, day, slot)] = (section, subject, teacher)

       if (teacher, day, slot) in teacher_schedule:
           conflicts += 1
       else:
           teacher_schedule[(teacher, day, slot)] = (section, subject)

       if (section, day, slot) in section_schedule:
           conflicts += 1
       else:
           section_schedule[(section, day, slot)] = (subject, room, teacher)

       teacher_day_count[teacher][day] += 1
       if teacher_day_count[teacher][day] > MAX_CLASSES_PER_DAY:
           conflicts += (teacher_day_count[teacher][day] - MAX_CLASSES_PER_DAY)

   for teacher in teachers:
       teacher_week_count[teacher["name"]] = sum(1 for count in teacher_day_count[teacher["name"]].values() if count > 0)
       if teacher_week_count[teacher["name"]] > MAX_DAYS_PER_WEEK:
           conflicts += (teacher_week_count[teacher["name"]] - MAX_DAYS_PER_WEEK)

   # Penalize solutions with back-to-back classes or scattered classes
   for section in sections:
       section_classes = [cls for cls in individual if cls[0] == section["name"]]
       for i in range(len(section_classes) - 1):
           cls1 = section_classes[i]
           cls2 = section_classes[i + 1]
           if cls1[3] == cls2[3] and abs(cls1[4] - cls2[4]) == 1:
               conflicts += 1  # Back-to-back classes penalty
           if cls1[3] != cls2[3]:
               conflicts += 1  # Scattered classes penalty

   for teacher in teachers:
       teacher_classes = [cls for cls in individual if cls[5] == teacher["name"]]
       for i in range(len(teacher_classes) - 1):
           cls1 = teacher_classes[i]
           cls2 = teacher_classes[i + 1]
           if cls1[3] == cls2[3] and abs(cls1[4] - cls2[4]) > 1:
               conflicts += 1  # Large gap between teacher's classes penalty

   return -conflicts

def mutate(individual):
   if random.random() < MUTATION_RATE:
       idx = random.randint(0, len(individual) - 1)
       section, subject, room, day, slot, teacher = individual[idx]
       new_room, new_day, new_slot = room, day, slot
       attempts = 0
       while attempts < MAX_ATTEMPTS:
           new_room = random.choice(rooms)["name"]
           new_day = random.randint(0, num_days - 1)
           new_slot = random.randint(0, slots_per_day - 1)
           if (new_room, new_day, new_slot) not in {(r, d, s) for _, _, r, d, s, _ in individual} and \
              (teacher, new_day, new_slot) not in {(t, d, s) for _, _, _, d, s, t in individual} and \
              (section, new_day, new_slot) not in {(sec, d, s) for sec, _, _, d, s, _ in individual}:
               break
           attempts += 1
       if attempts < MAX_ATTEMPTS:
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
    return random.choices(population, weights=adjusted_fitness, k=2)


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
        best_fitness = max([calculate_fitness(ind) for ind in population])
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")
        if best_fitness == 0:
            break

    best_individual = max(population, key=calculate_fitness)
    return best_individual

def visualize_schedule(individual):
    fig, axs = plt.subplots(nrows=len(rooms), ncols=num_days, figsize=(20, 15))
    
    for room_idx, room in enumerate(rooms):
        for day in range(num_days):
            for slot in range(slots_per_day):
                events = [(sec, subj, tea) for sec, subj, r, d, s, tea in individual if r == room["name"] and d == day and s == slot]
                if events:
                    event_text = "\n".join([f"{sec}\n{subj}\n{tea}" for sec, subj, tea in events])
                    axs[room_idx, day].text(0.5, (slot + 0.5) / slots_per_day, event_text, ha="center", va="center", fontsize=8, backgroundcolor='white')
                else:
                    axs[room_idx, day].text(0.5, (slot + 0.5) / slots_per_day, "", ha="center", va="center", fontsize=8, backgroundcolor='white')
                
                # Draw horizontal lines for slots
                axs[room_idx, day].axhline(y=slot / slots_per_day, color='black', linestyle='-', linewidth=0.5)
            
            axs[room_idx, day].set_xticks([])
            axs[room_idx, day].set_yticks([])
            axs[room_idx, day].set_title(f"Day {day + 1}", fontsize=12)
    
    for ax, room in zip(axs[:, 0], rooms):
        ax.set_ylabel(room["name"], fontsize=12, rotation=0, labelpad=20)
    
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add time slots to the right side of the plots
    for ax in axs[:, -1]:
        for slot in range(slots_per_day):
            ax.text(1.05, (slot + 0.5) / slots_per_day, time_slots[slot], transform=ax.transAxes, fontsize=10, va='center')

    plt.tight_layout()
    plt.show()

# Run the genetic algorithm and visualize the best schedule
best_solution = genetic_algorithm()
visualize_schedule(best_solution)

