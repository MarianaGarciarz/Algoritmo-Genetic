#!/usr/bin/env python
# coding: utf-8

# In[10]:


import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

all_stations = [
    # ♡♡♡♡♡♡♡♡♡Linea 1 color rosa♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡
    ('Tacubaya', 'Balderas', {'weight': 6}),
    ('Balderas', 'Salto del Agua', {'weight': 1}),
    ('Salto del Agua', 'Pino Suárez', {'weight': 2}),
    ('Pino Suárez', 'Candelaria', {'weight': 2}),
    ('Candelaria', 'San Lázaro', {'weight': 1}),
    ('San Lázaro', 'Goméz Farías', {'weight': 4}),
    ('Goméz Farías', 'Pantitlán', {'weight': 2}),
    # ♡♡♡♡♡♡♡♡Linea 2 azul marino♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡
    ('Cuatro caminos', 'Tacuba', {'weight': 1}),
    ('Tacuba', 'Hidalgo', {'weight': 7}),
    ('Hidalgo', 'Bellas Artes', {'weight': 1}),
    ('Bellas Artes', 'Pino Suárez', {'weight': 3}),
    ('Pino Suárez', 'Chabacano', {'weight': 2}),
    ('Chabacano', 'Ermita', {'weight': 6}),
    ('Ermita', 'Tasqueña', {'weight': 1}),
    #♡♡♡♡♡♡♡♡ Linea 3 amarilla fuerte ♡♡♡♡♡♡♡♡♡♡♡♡♡
    ('Indios verdes', 'Deportivo 18 de Marzo', {'weight': 1}),
    ('Deportivo 18 de Marzo', 'La raza', {'weight': 2}),
    ('La raza', 'Guerrero', {'weight': 2}),
    ('Guerrero', 'Hidalgo', {'weight': 1}),
    ('Hidalgo', 'Balderas', {'weight': 2}),
    ('Balderas', 'Centro Médico', {'weight': 3}),
    ('Centro Médico', 'Zapata', {'weight': 4}),
    ('Zapata', 'Universidad', {'weight': 2}),
    # ♡♡♡♡♡♡♡♡♡♡Linea  4 azul turquesa ♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡
    ('Martín Carrera', 'Consulado', {'weight': 3}),
    ('Consulado', 'Morelos', {'weight': 2}),
    ('Morelos', 'Candelaria', {'weight': 1}),
    ('Candelaria', 'Jamaica', {'weight': 2}),
    # ♡♡♡♡♡♡♡♡♡♡♡♡Linea 5 amarilla ♡♡♡♡♡♡♡♡♡♡♡♡♡
    ('Politécnico', 'Instituto del Petróleo', {'weight': 1}),
    ('Instituto del Petróleo', 'La raza', {'weight': 2}),
    ('La raza', 'Consulado', {'weight': 3}),
    ('Consulado', 'Oceanía', {'weight': 3}),
    ('Oceanía', 'Pantitlán', {'weight': 3}),
    # ♡♡♡♡♡♡♡♡♡♡♡♡Linea 6 roja ♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡
    ('El Rosario', 'Instituto del Petróleo', {'weight': 6}),
    ('Instituto del Petróleo', 'Deportivo 18 de Marzo', {'weight': 2}),
    ('Deportivo 18 de Marzo', 'Martín Carrera', {'weight': 1}),
    # ♡♡♡♡♡♡♡♡♡♡♡♡♡♡ Linea 7 naranja ♡♡♡♡♡♡♡♡♡♡♡♡♡♡
    ('Barranca del Muerto', 'Mixcoac', {'weight': 1}),
    ('Mixcoac', 'Tacubaya', {'weight': 3}),
    ('Tacubaya', 'Tacuba', {'weight': 5}),
    ('Tacuba', 'El Rosario', {'weight': 4}),
    # ♡♡♡♡♡♡♡♡♡Linea 8 naranja ♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡
    ('Garibaldi', 'Bellas Artes', {'weight': 1}),
    ('Bellas Artes', 'Salto del Agua', {'weight': 2}),
    ('Salto del Agua', 'Chabacano', {'weight': 3}),
    ('Chabacano', 'Atlalilco', {'weight': 8}),
    ('Atlalilco', 'Tláhuac', {'weight': 1}),
    # ♡♡♡♡♡♡♡♡♡♡♡♡♡Linea 9 color cafe♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡ 
    ('Tacubaya', 'Centro Médico', {'weight': 3}),
    ('Centro Médico', 'Chabacano', {'weight': 2}),
    ('Chabacano', 'Jamaica', {'weight': 1}),
    ('Jamaica', 'Pantitlán', {'weight': 5}),
    # ♡♡♡♡♡♡♡♡♡♡♡♡♡Línea 12 color carne ♡♡♡♡♡♡♡♡♡♡♡♡♡♡
    ('Mixcoac', 'Zapata', {'weight': 3}),
    ('Zapata', 'Ermita', {'weight': 3}),
    ('Ermita', 'Atlalilco', {'weight': 2}),
    ('Atlalilco', 'Tláhuac', {'weight': 1}),
    # ♡♡♡♡♡♡♡♡♡♡♡♡♡Línea B color verde-azul (Line B)♡♡♡♡♡♡♡♡♡♡♡
    ('Guerrero', 'Garibaldi', {'weight': 1}),
    ('Garibaldi', 'Morelos', {'weight': 3}),
    ('Morelos', 'San Lázaro', {'weight': 1}),
    ('San Lázaro', 'Oceanía', {'weight': 3}),
    ('Oceanía', 'Ciudad Azteca', {'weight': 1})
]

G.add_edges_from(all_stations)

plt.figure(figsize=(7, 10))
pos = nx.spring_layout(G) 
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=500, font_size=8, node_color='pink')  
plt.title('Estaciones del metro')
plt.show()


# In[8]:


pip install deap


# In[9]:


import random
import numpy
from deap import creator, base, tools, algorithms

#♡♡♡♡♡♡♡♡♡♡♡♡Algoritmo genético♡♡♡♡♡♡♡♡♡♡♡♡
POPULATION_SIZE = 10000
P_CROSSOVER = 0.8  
P_MUTATION = 0.2   
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 5

#♡♡♡♡♡♡la semilla aleatoria ♡♡♡♡♡♡♡♡
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ♡♡♡♡♡♡Crear los objetos de DEAP♡♡♡♡♡♡♡♡♡
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# ♡♡♡♡♡♡♡♡♡ Mapeo de estaciones ♡♡♡♡♡♡♡♡♡
station_to_index = {station: i for i, station in enumerate(G.nodes)}
index_to_station = {i: station for station, i in station_to_index.items()}

def crossover_connected_routes(ind1, ind2):
    possible_crossover_points = [i for i in range(1, len(ind1)-1) if G.has_edge(index_to_station[ind1[i]], index_to_station[ind2[i+1]]) and G.has_edge(index_to_station[ind2[i]], index_to_station[ind1[i+1]])]
    if not possible_crossover_points:
        return ind1, ind2

    crossover_point = random.choice(possible_crossover_points)
    
    # ♡♡♡♡♡♡Nuevos hijos asegurando que las estaciones sean consecutivas en el grafo♡♡♡♡♡♡
    new_ind1 = ind1[:crossover_point] + ind2[crossover_point:]
    new_ind2 = ind2[:crossover_point] + ind1[crossover_point:]
    #♡♡♡♡♡♡♡♡♡Reparar hijos♡♡♡♡♡♡
    new_ind1 = repair_route(new_ind1)
    new_ind2 = repair_route(new_ind2)
    
    return new_ind1, new_ind2

# ♡♡♡Elimina estaciones no consecutivas♡♡♡
def repair_route(route):
    repaired_route = [route[0]]
    for station in route[1:]:
        if G.has_edge(index_to_station[repaired_route[-1]], index_to_station[station]):
            repaired_route.append(station)
        else:
            # ♡♡♡♡♡♡la estación conectada más cercana y ajústala♡♡♡♡
            connected_stations = [s for s in G.neighbors(index_to_station[repaired_route[-1]])]
            repaired_route.append(station_to_index[random.choice(connected_stations)])
    return repaired_route

# ♡♡♡♡♡♡ la función de cruce en la caja de herramientas♡♡♡♡♡♡♡♡♡♡♡♡
toolbox.register("mate", crossover_connected_routes)

# ♡♡♡♡♡♡ Función para generar permutaciones aleatorias de indices de las estaciones ♡♡♡♡
def random_route_indices():
    middle_stations = list(station_to_index.values())
    middle_stations.remove(station_to_index["El Rosario"])
    middle_stations.remove(station_to_index["San Lázaro"])
    random_route = random.sample(middle_stations, len(middle_stations))
    return [station_to_index["El Rosario"]] + random_route + [station_to_index["San Lázaro"]]

toolbox.register("individualCreator", tools.initIterate, creator.Individual, random_route_indices)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# ♡♡♡♡♡♡♡♡♡♡♡♡ el tiempo total para cada ruta ♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡
def route_time_indices(individual):
    total_time = 0
    for i in range(1, len(individual)):
        if G.has_edge(index_to_station[individual[i-1]], index_to_station[individual[i]]):
            total_time += G[index_to_station[individual[i-1]]][index_to_station[individual[i]]]['weight']
        else:
            total_time += 1000000  
    return total_time,

toolbox.register("evaluate", route_time_indices)

# ♡♡♡♡♡♡♡♡♡♡♡ Operador genético♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(G.nodes))

# ♡♡♡♡♡♡♡♡♡♡♡♡♡Resultados ♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡♡
def print_friendly_results(hof):
    print("\n---------------- Mejores Rutas Encontradas ----------------")
    for i, individual in enumerate(hof.items):
        route = [index_to_station[index] for index in individual]
        print(f"\nRuta {i+1} con el Tiempo Más Corto ({individual.fitness.values[0]} minutos):")
        print(" -> ".join(route))

# ♡♡♡♡♡♡♡♡♡♡♡♡ Flujo del algoritmo ♡♡♡♡♡♡♡♡♡♡♡♡
def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    print_friendly_results(hof)

    return population, logbook, hof

main()


# In[ ]:




