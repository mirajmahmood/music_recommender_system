import numpy as np
from deap import base
from deap import creator
from deap import tools, algorithms
import glob 
import scipy.spatial as sp
import random
from time import time

global normalized_population

def initialize_attributes(folder_dir):
  global normalized_population

  population_files = glob.glob(folder_dir)
  population = np.zeros(len(population_files), dtype=object)
  for i in range(len(population_files)):
    population[i] = np.load(population_files[i])

  #equalizing dim size of songs
  min_len = 10000
  for each in population:
    shape = each.shape
    if shape[1] < min_len:
      min_len = shape [1]

  normalized_population = np.copy(population)
  for i in range(len(normalized_population)):
    normalized_population[i].resize((normalized_population[i].shape[0], min_len), refcheck=False)
    normalized_population[i] = normalized_population[i][::2,:]

  ###

  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
  toolbox = base.Toolbox()

  # Randomly samples individuals in popluation
  toolbox.register("pickInd", random.randint, 0, len(normalized_population)-1)

  # Structure initializers
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.pickInd, 100)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", evalFitness)
  toolbox.register("mate", tools.cxTwoPoint)
  toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
  toolbox.register("select", tools.selTournament, tournsize=3)

  return toolbox

def evalFitness(individual):
  seed = 2
  score = 0
  for i in individual:
    score += calculate_song_score(seed, i)
        
  return score,

def calculate_song_score(seed, song):
  x = sp.distance.cdist(normalized_population[seed], normalized_population[song], 'euclidean')
  sim_x = 1/(x + 1)
  scale_x = np.sum(sim_x, axis=1)
  weights = [2, 1, 1, 1, 3, 1, 2, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  weighted_x = (scale_x * weights)/77
  score = np.sum(weighted_x)
  return score

def main(toolbox):
  random.seed(64)
  pop = toolbox.population(n=300)
  hof = tools.HallOfFame(1)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("std", np.std)
  stats.register("min", np.min)
  stats.register("max", np.max)
  
  pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
                                 stats=stats, halloffame=hof, verbose=True)
  
  return pop, log, hof

if __name__ == "__main__":
  folder_dir = "randomdata/*"
  toolbox = initialize_attributes(folder_dir)
  start_time = time()
  main(toolbox)
  print("Time taken: ", time()-start_time)