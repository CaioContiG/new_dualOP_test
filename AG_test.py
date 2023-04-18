import random as rd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from instancias_manuais import mundo_16x16, mundo_21x17
from itertools import chain # For unnest list
import heapq # For selecting n best

# TODO instead of using index, use x,y directly?

n_stops = 5
max_shelves_per_stop = 6
n_row = 16
n_col = 16
n_nodes = n_row*n_col
pop_size = 100
mut_prob_ugv = 0.1
mut_prob_uav = 0.2
n_generation = 200
n_tournament = 3
budget_ground = 200
budget_aerial = 60
region_near = 3
shelf_prob = 0.6

# Transforming graph from nx to dictionary
def graph_to_dict(grafo_org):
    grafo_dict = []
    for node in grafo_org.nodes():
        node_dict = {
            "index": grafo_org.nodes[node]['index'],
            "x": node[0],
            "y": node[1],
            "estante": grafo_org.nodes[node]['estante'],
            "corredor": grafo_org.nodes[node]['corredor'],
            "reward_est": grafo_org.nodes[node]['reward_est'],
            "near_nodes": nx.single_source_shortest_path_length(grafo,node,3).keys()
        }
        grafo_dict.append(node_dict)
    return grafo_dict

# Adding edges in the aerial graph
def add_edges_aerial(grafo, rows, cols):
    for col in range(cols):
        for row in range(rows):
            # Verifica todos pontos e tenta ligar                 
            for i in range(cols):
                    for j in range(rows):
                        if i != col or j != row:
                            a = np.array((row, col))
                            b = np.array((j,i))
                            distancia = np.linalg.norm(a-b) 
                            if distancia < 4.4:
                                grafo.add_edge((col, row), (i, j), budget=distancia)

# Removing edges from shelves (for UGV)
def rmv_shelves_edges(grafo):
    for node in (grafo):
        if grafo.nodes[node]['estante'] == 1:
            vizinhos = [n for n in grafo.neighbors((node))]
            for v in vizinhos:
                grafo.remove_edge(node, v)   

# Set attributes in a graph
def set_attrs(graph, m_ger):
    cnt = 0
    for i in range(n_col):
        for j in range(n_row):
            y = m_ger[i][j].y_network
            x = m_ger[i][j].x_network
            graph.nodes[(x,y)]['corredor'] = m_ger[i][j].corredor
            graph.nodes[(x,y)]['estante'] = m_ger[i][j].estante
            graph.nodes[(x,y)]['reward_est'] = m_ger[i][j].reward_est

    for node in graph:
        graph.nodes[node]['index'] = cnt
        cnt+=1

    for edge in graph.edges:
        graph.edges[edge]['budget'] = 1

# Calculate ground budget used
# TODO set graph_dict/ground/aerial, indexes as global?
# Using the list graph_dict to extract x,y, is there a way to extract directly from the nx graph using index?
def calc_bud_ground(graph_dict, graph_ground, individual):    
    stops_x_y = [(graph_dict[stop]["x"], graph_dict[stop]["y"]) for stop in individual['stops_ugv']] # x,y path (change AG from index to x,y?)
    budget_used = 0
    
    for i in range (1, len(stops_x_y)):
        shortest_path = nx.shortest_path(graph_ground, source=stops_x_y[i-1], target=stops_x_y[i], weight='budget')
        for j in range(1, len(shortest_path)):
            budget_used += graph_ground[shortest_path[j-1]][shortest_path[j]]["budget"]
    return budget_used

# Calculate aerial budget used
# TODO Use dictionary functions instead of making comprehension list?
def calc_bud_aerial(graph_dict, graph_aerial, individual):
    budget_used = 0
    shelves_list_x_y = []
    for shelves_stop in individual['shelves_uav']:
        if shelves_stop != -1: # if it is not empty individual
            shelves_list_x_y.append([(graph_dict[shelf_index]['x'], graph_dict[shelf_index]['y']) for shelf_index in shelves_stop])
     
    # Para cada parada, calcular seu caminho e somar à distância total.
    for i in range(0,len(individual['stops_ugv'])):
        stop_index = individual['stops_ugv'][i]
        lista_parada = [(graph_dict[stop_index]['x'], graph_dict[stop_index]['y'])]
        trajeto_aereo = lista_parada + shelves_list_x_y[i] + lista_parada # O trajeto aéreo sai da parada visita estantes e volta
        
        # Calcula trajeto final 
        # Pois pode n ser possível sair de uma estante direto para outra, por conta da correcao de estantes visitadas
        # que pode retirar uma estante intermediária
        trajeto_aereo_final = []
        trajeto_aereo_final.append(lista_parada[0])
        for i in range(1,len(trajeto_aereo)):
            trajeto_dois_nodes = nx.shortest_path(graph_aerial, source=trajeto_aereo[i-1], target=trajeto_aereo[i], weight='budget')
            for node in trajeto_dois_nodes[1:]:
                trajeto_aereo_final.append(node)
        
        # Distancia euclidiana
        for i in range(len(trajeto_aereo_final)-1):
            a = np.array(trajeto_aereo_final[i])
            b = np.array(trajeto_aereo_final[i+1])
            budget_used += np.linalg.norm(a-b)

    return budget_used

##################### NEW REGION - ag functions ###########################
# Ex.:
individual_ex1 = {
    "stops_ugv": [9, 7, 8, 13, 14],
    "shelves_uav": [[1, 3], [10, 4], [9, 1], [10, 21], [7,8]],
    "fitness": 8
}

individual_ex2 = {
    "stops_ugv": [1, 2, 3, 4, 5],
    "shelves_uav": [[0, 4], [11, 4876], [78, 438], [90, 21], [78, 36]],
    "fitness": 9
}

# Uni. randomly decides shelves to be visited in a stop point - several ways to do it
# This one picks uni randomly, another way is picking uni rand from the neighbors
def rd_shelves(near_nodes):
    shelves_selected = []
    if len(near_nodes) >= max_shelves_per_stop:
        shelves_selected = rd.sample(near_nodes,max_shelves_per_stop)
    else: 
        shelves_selected = rd.sample(near_nodes,len(near_nodes)) + [-1 for i in range(0, (max_shelves_per_stop - len(near_nodes)))]

    # Taking out some shelves uni. rand.
    for i in range(0,len(shelves_selected)):
        if (rd.random() < shelf_prob):
            shelves_selected[i] = -1
    
    return shelves_selected

# Decides uni rand which neighbor node it will go
def rd_shelves_neighbors(stop):
    initial_path = []
    previous_node_index = stop # Stop is integer, so no reference
    for i in range(0,max_shelves_per_stop):
        node = rd.choice(neighbors_list[previous_node_index])
        previous_node_index = graph_aerial.nodes[node]['index']        
        initial_path.append(previous_node_index)
    
    # Filtering to only desired shelves
    shelves_selected = [n for n in initial_path if graph_dict[n]['reward_est'] > 0]

    if shelves_selected == []:
        shelves_selected.append(-1)

    return shelves_selected

# Creates uniformly randomly a individual (solution)
def random_ind(indexes_shelves, indexes_aisle, near_nodes):
    stops = rd.sample(indexes_aisle,n_stops)
    shelves = []
    for i in range(n_stops):
        #shelves.append(rd_shelves(near_nodes[stops[i]])) old way
        shelves.append(rd_shelves_neighbors(stops[i]))

    individual = {
        "stops_ugv": stops,
        "shelves_uav": shelves,
        "fitness" : -1
    }
    
    return individual

# Fitness Calculation (number of non repeated shelves visited) NOT REWARD
# TODO correct if shelf is visited one more time ---> already orrected?
# TODO use reward instead of counting unique shelves
# TODO is there a faster way to do it?
def fitness(individual, graph_dict, graph_ground, graph_aerial):    
    unnested_shelves = list(chain.from_iterable(individual['shelves_uav'])) # Unnesting
    # Checking existence of empty shelves:
    empty = 0
    for shelf in unnested_shelves:
        if shelf == -1:
            empty = 1
            break

    if empty == 1: # In case a "empty" shelf appear
        fit = len(set(unnested_shelves)) -1# Conting unique shelves
    else:
        fit = len(set(unnested_shelves)) # Conting unique shelves

    bud_aer = calc_bud_aerial(graph_dict,graph_aerial,individual)
    bud_gr = calc_bud_ground(graph_dict,graph_ground,individual)

    # Setting fitness
    if (bud_aer > budget_aerial) and (bud_gr > budget_ground):
        fit = fit/20
    elif (bud_aer > budget_aerial) or (bud_gr > budget_ground):
        fit = fit/10

    individual['fitness'] = fit

    return fit

def fitness_all(population,graph_dict, graph_ground, graph_aerial):
    return [fitness(ind,graph_dict, graph_ground, graph_aerial) for ind in population]

#TODO not very optmized I guess
#TODO add upset probability
def tournament(pool):
    best = pool[0]
    for parent in pool:
        if parent["fitness"] > best["fitness"]:
            best = parent
    return best

# Single point crossover
# TODO put crossover probability
# TODO put for instead of creating 2 offspring manually?
def crossover_single_point(parents_org):
    # Manual copy so it doesn't use reference or deepcopy (is there a better way like what I'm doing in uniform crossover?)
    parent1 = manual_copy(parents_org[0])
    parent2 = manual_copy(parents_org[1])
    cut = rd.choice(range(0,n_stops-1))
    stops_off1 = parent1['stops_ugv'][:cut] + parent2['stops_ugv'][cut:]
    shelves_off1 = parent1['shelves_uav'][:cut] + parent2['shelves_uav'][cut:]

    stops_off2 = parent2['stops_ugv'][:cut] + parent1['stops_ugv'][cut:]
    shelves_off2 = parent2['shelves_uav'][:cut] + parent1['shelves_uav'][cut:]
    offspring1 = {
        "stops_ugv": stops_off1,
        "shelves_uav": shelves_off1,
        "fitness" : -1
    }

    offspring2 = {
        "stops_ugv": stops_off2,
        "shelves_uav": shelves_off2,
        "fitness" : -1
    }

    return offspring1, offspring2

# Uniform crossover
# TODO put crossover probability
# TODO put for instead of creating 2 offspring manually?
def crossover_uniform(parents):
    choices_list = rd.choices(range(0,2),k=n_stops)
    stops1 = []
    stops2 = []
    shelves1 = []
    shelves2 = []
    for i in range(0,len(choices_list)):
        stops1.append(parents[choices_list[i]]['stops_ugv'][i])
        shelves1.append([*parents[choices_list[i]]['shelves_uav'][i]]) 
        stops2.append(parents[(choices_list[i] + 1) % 2]['stops_ugv'][i])
        shelves2.append([*parents[(choices_list[i] + 1) % 2]['shelves_uav'][i]])

    offspring1 = {
        "stops_ugv": stops1,
        "shelves_uav": shelves1,
        "fitness" : -1
    }

    offspring2 = {
        "stops_ugv": stops2,
        "shelves_uav": shelves2,
        "fitness" : -1
    }

    return offspring1, offspring2


# Uniform mutation - changes stops and shelves uni rand
#TODO Need to pass indexes_aisle and indexes_shelves or can we declare as global?
def mutation(individual,indexes_aisle, indexes_shelves, near_nodes):

    # Mutation probability for stops_ugv
    if rd.random() < mut_prob_ugv:
        for i in range(0,len(individual['stops_ugv'])):
            if rd.random()<0.3: # prob mutate this stop
                new_stop = rd.choice(indexes_aisle)
                individual['stops_ugv'][i] = new_stop 
                individual['shelves_uav'][i] = rd_shelves_neighbors(new_stop) # I don't think there is any reference problem, but I didn't test it out

    # Mutation probability for shelves_uav
    if rd.random() < mut_prob_uav:
        for i in range(0,len(individual['stops_ugv'])):
            if rd.random()<0.3: # prob mutate this stop
                for j in range(0,len(individual['shelves_uav'][i])):
                    if rd.random()<0.3: # prob mutate this shelf
                        if rd.random()<0.5:
                            index_stop = individual['stops_ugv'][i]
                            if (near_nodes[index_stop] != []):
                                new_shelve = rd.choice(near_nodes[index_stop])
                            else:
                                new_shelve = -1
                            individual['shelves_uav'][i][j] = new_shelve # I guess it can't be by reference if indexes = [0,1,2,3,4,5...]
                        else:
                            new_shelve = -1
                            individual['shelves_uav'][i][j] = new_shelve 



# Manual copy of one individual (should be faster than deepcopy)
def manual_copy(individual):
    new_shelves_list = [[*shelves] for shelves in individual['shelves_uav']]
    new_ind = {
        "stops_ugv": [*individual['stops_ugv']],
        "shelves_uav": new_shelves_list,
        "fitness" : individual['fitness']
    }
    return new_ind

### setting up ###
# Creating graphs
matrix_attrs = mundo_16x16()
grafo = nx.grid_2d_graph(n_col,n_row)
graph_ground = nx.grid_2d_graph(n_col,n_row)
graph_aerial = nx.grid_2d_graph(n_col,n_row)

# Setting graph attributes from matrix_attrs (aisles, shelves and shelf reward)
set_attrs(graph_ground, matrix_attrs)
set_attrs(graph_aerial, matrix_attrs)
set_attrs(grafo, matrix_attrs)

# Setting edges on the graph
add_edges_aerial(graph_aerial,n_row,n_col)
rmv_shelves_edges(graph_ground)

# Transforming graph to dictionary
graph_dict = graph_to_dict(graph_ground)

# Taking useful information
shelves_list = list(filter(lambda x: x['estante'] == 1, graph_dict))
indexes_shelves = [x['index'] for x in shelves_list if x['reward_est'] > 0]
neighbors_list = []
for node in graph_aerial.nodes():
    neighbors_list.append([n for n in graph_aerial.neighbors(node)])

# Appending "virtual" nodes -1, to represent empty genes so we can have a fix shelves stop but with less real stops
# Those nodes will have a value (x,y) = empty = []
for i in range(0,50):
    indexes_shelves.append(-1)

# Creating the region for each node
near_nodes = []
for node in grafo.nodes():
    near_nodes_x_y = list(nx.single_source_shortest_path_length(grafo,node,region_near).keys())    
    near_nodes.append([grafo.nodes[n]['index'] for n in near_nodes_x_y if grafo.nodes[n]['reward_est'] == 1])

aisle_list = list(filter(lambda x: x['estante'] != 1, graph_dict))
indexes_aisle = [x['index'] for x in aisle_list]

### Starting AG ###

# Creating population
pop = [random_ind(indexes_shelves, indexes_aisle, near_nodes) for i in range(0,pop_size)]
list_fitness = fitness_all(pop,graph_dict, graph_ground, graph_aerial)
best_per_generation = [max(list_fitness)]
media = [np.mean(list_fitness)]

# Starting evolution
for gen in range(0,n_generation):
    print(gen)   
    filhos = []    

    #print(calc_bud_aerial(graph_dict,graph_aerial,individual_ex1))
    offspring = []
    for j in range((int)(len(pop)/2)):
        # Tournament, parents can repeat this way.
        parent1 = tournament(rd.sample(pop,n_tournament))
        parent2 = tournament(rd.sample(pop,n_tournament))

        # Uniform crossover (it is too messy? change to single point?)
        off1, off2 = crossover_uniform([parent1,parent2])

        # Mutation
        mutation(off1,indexes_aisle,indexes_shelves, near_nodes)
        mutation(off2,indexes_aisle,indexes_shelves, near_nodes)

        # Append to total offspring
        offspring.append(off1)
        offspring.append(off2)
    
    # Creating new population (just coping all offspring)
    # Maybe add elitism
    pop = [manual_copy(x) for x in offspring]
    
    # TODO guardar o melhor de todas populações
    list_fitness = fitness_all(pop,graph_dict, graph_ground, graph_aerial) # Evaluating whole population
    best_per_generation.append(max(list_fitness))
    media.append(np.mean(list_fitness)) 

#print(near_nodes)

plt.plot(media, label='Average per generation')
plt.plot(best_per_generation, label='Best individual per generation')
plt.legend()
plt.show()

# PRINTANDO NETWORKX ######################################################################################
#for i in range(len(graph_dict)):
#    print("Dict: ", i, ", Index: ", graph_dict[i]['index'], " X Y: ", graph_dict[i]['x'], ", ", graph_dict[i]['y'])

best_all = pop[list_fitness.index(max(list_fitness))]   

cam_tupla = []
#print("Stops best all ugv: ", best_all['stops_ugv'])
for stop in best_all['stops_ugv']:
    #print(stop)
    cam_tupla.append((graph_dict[stop]['x'], graph_dict[stop]['y']))
#print("cam_tupla: ", cam_tupla)
# Reconstruindo caminho
caminho_final = []
for i in range (1,len(cam_tupla)):
    menor_caminho = nx.shortest_path(graph_ground, source=cam_tupla[i-1], target=cam_tupla[i], weight='budget')
    for node in menor_caminho[0:-1]:
        caminho_final.append(node)
caminho_final.append(cam_tupla[-1])

cam_final_tupla = [(caminho_final[i-1],caminho_final[i]) for i in range(1,len(caminho_final))]
del cam_final_tupla[-1]

#plt.figure("Grafo anterior")
pos = {node:(node[0],node[1]) for node in grafo.nodes()}
nodes_corr = [node for node in grafo.nodes() if grafo.nodes[node]['estante'] == 0]
nodes_estante = [node for node in grafo.nodes() if grafo.nodes[node]['estante'] == 1 and grafo.nodes[node]['reward_est'] == 0]
nodes_estante_reward = [node for node in grafo.nodes() if grafo.nodes[node]['estante'] == 1 and grafo.nodes[node]['reward_est'] == 1]
edges_estante = [edge for edge in grafo.edges(nodes_estante)]

### PRINTANDO TERRESTRE ####
#nodes_grafo = []
#for node in grap:
#    nodes_grafo.append((node['x'], node['y']))
#e_vis = fitnessV4(best_all, grafo_aereo, budget_aereo, grafo_aereo, budget_aereo)
#titulo = "Budget terrestre: " + str(distancia_terrestre)  + "/" + str(bud_terrestre) + ", Budget aereo: " +str(distancia_aereo)+ "/"+str(budget_aereo) + ", Estantes visitadas: "+str(e_vis)+ "/"+str(n_estantes_desejadas_total) + ", Geracoes: " + str(n_ger) + ", Bud local: "+ str(budget_aereo_sobrevoo) + ", N pop: " + str(n_pop) + ", N paradas in: " + str(n_paradas)
titulo = "Titulo"
fig = plt.figure(titulo)
label_cam_final = {}
label_paradas = {}
cnt = 0
for node in caminho_final:
    cnt += 1
    label_cam_final[node] = cnt

cnt = 0
for node in cam_tupla:
    cnt += 1
    label_paradas[node] = cnt

edges_estante = [edge for edge in graph_ground.edges(nodes_estante)]


#nx.draw(grafo_terrestre, pos, font_size=8, node_size=20, node_color='green')
nx.draw(graph_ground, pos, labels = label_paradas, with_labels = True, font_size=12, font_color='black', node_size=10, node_color='black', edge_color = 'gray', width = 0.5)
#nx.draw_networkx_nodes(grafo_terrestre, pos, nodelist=nodes_corr, node_size=20, node_color='black')
nx.draw_networkx_nodes(graph_ground, pos, nodelist=cam_tupla, node_size=220, node_color='red') # Pontos de parada
nx.draw_networkx_nodes(graph_ground, pos, nodelist=(cam_tupla[0], cam_tupla[-1]), node_size=300, node_color='black') # Começo e fim
nx.draw_networkx_nodes(graph_ground, pos, nodelist=(cam_tupla[0], cam_tupla[-1]), node_size=250, node_color='red') # Começo e fim
nx.draw_networkx_nodes(grafo, pos, nodelist=nodes_estante_reward, node_size=120, node_color='red', node_shape = 's') # Estantes desejadas
nx.draw_networkx_nodes(graph_ground, pos, nodelist=nodes_estante, node_size=20, node_color='black', node_shape = 's') # Estantes não desejadas
nx.draw_networkx_edges(graph_ground,pos,edgelist=edges_estante,edge_color='red')
#nx.draw(grafo_terrestre, pos, labels = label_paradas, with_labels = True, font_size=15, font_color='red', node_size=20, node_color='green')
nx.draw_networkx_edges(graph_ground,pos, edgelist=cam_final_tupla,edge_color='blue', arrows=True, arrowsize = 10, width=1) # CAMINHO


#correct_vis_estantes(best_all)
cnt = 0
for i in range(len(best_all['stops_ugv'])):
#for parada in best_all['stops_ugv']:
    parada_index = best_all['stops_ugv'][i]
    lista_parada = [(graph_dict[parada_index]['x'], graph_dict[parada_index]['y'])]
    est_visitadas_parada = [(graph_dict[estante_index]['x'], graph_dict[estante_index]['y']) for estante_index in best_all['shelves_uav'][i] if estante_index >= 0]
    trajeto_aereo = lista_parada + est_visitadas_parada + lista_parada # O trajeto aéreo sai da parada visita estantes e volta

    caminho_parada = []
    caminho_parada.append(trajeto_aereo[0])
    for i in range (1,len(trajeto_aereo)):
        rec_cam_parada = nx.shortest_path(graph_aerial, source=trajeto_aereo[i-1], target=trajeto_aereo[i], weight='budget')
        for node in rec_cam_parada[1:]:
            caminho_parada.append(node)
    cam_parada_tupla = [(caminho_parada[i-1],caminho_parada[i]) for i in range(1,len(caminho_parada))]
    nx.draw_networkx_edges(graph_aerial,pos,edgelist=cam_parada_tupla,edge_color='red', arrows=True)
    cnt += 1

plt.show()
print(best_all['fitness'])