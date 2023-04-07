import random as rd
import networkx as nx
import numpy as np
from instancias_manuais import mundo_16x16, mundo_21x17

n_stops = 5
max_shelves_per_stop = 5
n_row = 16
n_col = 16

# Transforming graph from nx to dictionary
def graph_to_dict(grafo_org):
    grafo_dict = []
    for node in grafo_org.nodes():
        node_dict = {
            "index": grafo_org.nodes[node]['index'],
            "estante": grafo_org.nodes[node]['estante'],
            "corredor": grafo_org.nodes[node]['corredor'],
            "reward_est": grafo_org.nodes[node]['reward_est'],
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



##################### NEW REGION ###########################

# Uni. randomly decides shelves to be visited - several ways to do it
def rd_shelves(stop_node):
    return (1,2)

# Creates uniformly randomly a individual (solution)
def random_ind():
    n_nodes = (n_row* n_col) - 1
    stops = rd.sample(range(0,n_nodes),n_stops)
    shelves = []
    for i in range(n_stops):
        shelves.append(rd_shelves(1))

    individual = {
        "stops_UGV": stops,
        "shelves_UAV": shelves
    }
    
    return individual
        

# Ex.:
individual = {
    "stops_UGV": [2, 7, 8, 13],
    "shelves_UAV": [(1, 3), (10), (), (15, 21)]
}

# Creating graphs
matrix_attrs = mundo_16x16()
graph_ground = nx.grid_2d_graph(n_col,n_row)
graph_aerial = nx.grid_2d_graph(n_col,n_row)

# Setting graph attributes from matrix_attrs (aisles, shelves and shelf reward)
set_attrs(graph_ground, matrix_attrs)
set_attrs(graph_aerial, matrix_attrs)

# Setting edges on the graph
add_edges_aerial(graph_aerial,n_row,n_col)
rmv_shelves_edges(graph_ground)

# Transforming graph to dictionary
grafo_dict = graph_to_dict(graph_ground)
