#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:40:45 2022

@author: ajmatheson-lieber
"""

import pygame
from collections import defaultdict, deque
import numpy as np
import time
import random
from heapq import heapify, heappush, heappop

window_size = 500
background = (255, 255, 255)
shape_color = (0, 0, 255)
mid = window_size//2
square_size = 50
graph_size = 10

# Use run_prims() to generate a prims animation
# use rotating cube to see a rotating cube
# Use moiving cube to see a moving cube



class rendering ():
    """
    class used for rendering 3d graphics on the 2d plane
    """
    def __init__ (self, G = None, focal = 500, VertexTable = None, EdgeTable = None):
        self.focal = focal
        if G != None:
            self.G = G
        else:
            self.G = defaultdict(set)
            for i in EdgeTable:
                self.G[VertexTable[i[0]]].add(VertexTable[i[1]])
                self.G[VertexTable[i[1]]].add(VertexTable[i[0]])
        self.GP = self.project_graph(G)
        self.screen_on = False
        
    def update_GP (self):
        self.GP = self.project_graph(self.G)
        return 
    
    def change_vertex (self, old, new):
        for i in self.G[old]:
            #print(i, old,  new, self.G[i], self.G)
            self.G[i] = self.G[i] - {old}
            self.G[i].add(new)
        self.G[new] = self.G[old]
        del self.G[old]
        self.update_GP()
        return
    
    def shift_vertex (self, node, direction, value):
        if direction == "x" or direction == "X":
            self.change_vertex(node, (node[0] + value, node[1], node[1]))
        elif direction == "y" or direction == "Y":
            self.change_vertex(node, (node[0], node[1] + value, node[1]))
        elif direction == "y" or direction == "Y":
            self.change_vertex(node, (node[0], node[1], node[1] + value))
        elif direction not in {"x", "X", "y", "Y", "z", "Z"}:
            raise ValueError
        self.update_GP()
        return
    
    def shift (self, direction, value):
        temp = self.G.copy()
        for node in temp:
            if direction == "x" or direction == "X":
                self.change_vertex(node, (node[0] + value, node[1], node[2]))
            elif direction == "y" or direction == "Y":
                self.change_vertex(node, (node[0], node[1] + value, node[2]))
            elif direction == "y" or direction == "Y":
                self.change_vertex(node, (node[0], node[1], node[2] + value))
            elif direction not in {"x", "X", "y", "Y", "z", "Z"}:
                raise ValueError
        return
    
    def rotate_vertex_origin (self, node, direction, value):
        if direction == "x" or direction == "X":
            new = np.matrix(node) * np.matrix([[1, 0, 0], [0, np.cos(value), -np.sin(value)], [0, np.sin(value), np.cos(value)]])
            self.change_vertex(node, tuple(new.tolist()[0]))      
        elif direction == "y" or direction == "Y":
            new = np.matrix(node) * np.matrix([[np.cos(value), 0, np.sin(value)], [0, 1, 0], [-np.sin(value), 0, np.cos(value)]])
            self.change_vertex(node, tuple(new.tolist()[0]))
        elif direction == "y" or direction == "Y":
            new = np.matrix(node)
        elif direction not in {"x", "X", "y", "Y", "z", "Z"}:
            raise ValueError
        return
    
    def rotate_origin (self, direction, value):
        temp = rendering(self.G.copy())
        for node in temp.G:
            if direction == "x" or direction == "X":
                new = np.matrix(node) * np.matrix([[1, 0, 0], [0, np.cos(value), -np.sin(value)], [0, np.sin(value), np.cos(value)]])
                self.change_vertex(node, tuple(new.tolist()[0]))      
            elif direction == "y" or direction == "Y":
                new = np.matrix(node) * np.matrix([[np.cos(value), 0, np.sin(value)], [0, 1, 0], [-np.sin(value), 0, np.cos(value)]])
                self.change_vertex(node, tuple(new.tolist()[0]))
            elif direction == "z" or direction == "Z":
                new = np.matrix(node)
            elif direction not in {"x", "X", "y", "Y", "z", "Z"}:
                raise ValueError
        return
    
    def rotate_vertex_point (self, node, point, direction, value):
        node = (node[0] - value, node[1] - value, node[2] - value)
        if direction == "x" or direction == "X":
            new = np.matrix(node) * np.matrix([[1, 0, 0], [0, np.cos(value), -np.sin(value)], [0, np.sin(value), np.cos(value)]])
            node = (node[0] + value, node[1] + value, node[2] + value)
            self.change_vertex(node, tuple(new.tolist()[0]))      
        elif direction == "y" or direction == "Y":
            new = np.matrix(node) * np.matrix([[np.cos(value), 0, np.sin(value)], [0, 1, 0], [-np.sin(value), 0, np.cos(value)]])
            node = (node[0] + value, node[1] + value, node[2] + value)
            self.change_vertex(node, tuple(new.tolist()[0]))
        elif direction == "y" or direction == "Y":
            new = np.matrix(node)
        elif direction not in {"x", "X", "y", "Y", "z", "Z"}:
            raise ValueError
        return
    
    def rotate_point (self, point = (mid, mid, mid), direction = "x", value = .1):
        temp = self.G.copy()
        for node in temp:
            node = (node[0] - point[0], node[1] - point[1], node[2] - point[2])
            if direction == "x" or direction == "X":
                new = np.matrix(node) * np.matrix([[1, 0, 0], [0, np.cos(value), -np.sin(value)], [0, np.sin(value), np.cos(value)]])
                new = new.tolist()[0]
                node = (node[0] + point[0], node[1] + point[1], node[2] + point[2])
                new = (new[0] + point[0], new[1] + point[1], new[2] + point[2])
                self.change_vertex(node, new)      
            elif direction == "y" or direction == "Y":
                new = np.matrix(node) * np.matrix([[np.cos(value), 0, np.sin(value)], [0, 1, 0], [-np.sin(value), 0, np.cos(value)]])
                node = (node[0] + point[0], node[1] + point[1], node[2] + point[2])
                new = new.tolist()[0]
                node = (node[0] + point[0], node[1] + point[1], node[2] + point[2])
                new = (new[0] + point[0], new[1] + point[1], new[2] + point[2])
                self.change_vertex(node, new)  
            elif direction == "z" or direction == "Z":
                #NOT IMPLEMENTED
                new = np.matrix(node)
                node = (node[0] + point[0], node[1] + point[1], node[2] + point[2])
                new = new.tolist()[0]
                node = (node[0] + point[0], node[1] + point[1], node[2] + point[2])
                new = (new[0] + point[0], new[1] + point[1], new[2] + point[2])
                self.change_vertex(node, new)  
            elif direction not in {"x", "X", "y", "Y", "z", "Z"}:
                raise ValueError
        return
    
    
    def insert (self, node, connections):
        self.G[node] = set(connections)
        for i in connections:
            self.G[i].add(node)
        return 
    
    def project (self, x, y, z):
        return (mid - (self.focal*(mid - x)/(z+self.focal)), (mid - (self.focal*(mid - y)/(z+self.focal))))
    
    def project_graph(self, G):
        GP = defaultdict(list)
        for i in G:
            for j in G[i]:
                GP[self.project(i[0], i[1], i[2])].append(self.project(j[0], j[1], j[2]))
        return GP
    
    def display (self) :
        global screen
        if not self.screen_on:
            screen = pygame.display.set_mode([window_size, window_size])
            self.screen_on = True
            screen.fill(background)
            
        for i in self.GP:
            for j in self.GP[i]:
                pygame.draw.line(screen, shape_color, i, j, 1)
        
        
                
class RRP ():
    def __init__ (self):
        pass
    
    
def generate_RRP(nodes = {(100, 100, 100), (200, 100, 100), (100, 200, 100), (200, 200, 100), (100, 100, 200), (200, 100, 200), (100, 200, 200), (200, 200, 200)}):
    G = defaultdict(set)
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            dx = i[0] == j[0]
            dy = i[1] == j[1]
            dz = i[2] == j[2]
            if dx and dy:
                G[i].add(j)
            elif dz and dx:
                G[i].add(j)
            elif dy and dz:
                G[i].add(j)
    return G




def rotatingcube (point = (mid, mid, mid), direction = "x", speed = 0.01, size = 300):
    """
    

    Parameters
    ----------
    point : TYPE, optional
        list of the center values of the cube. The default is (mid, mid, mid).
    direction : str: x, y, or z,
        determines the direction of rotation. The default is "x".
    speed : TYPE, optional
        controls the speed of rotation. The default is 0.01.
    size : TYPE, optional
        determines the size of the cube. The default is 300.

    Returns
    -------
    None.

    """
    G = cube(center = point, size = size)
    myclass = rendering(G)
    global screen
    screen = pygame.display.set_mode([window_size, window_size])
    myclass.screen_on = True
    screen.fill(background)
    myclass.display()
    running = True
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(background)
        myclass.rotate_point(point, direction, speed)
        myclass.display()
        pygame.display.update()
    pygame.quit()
    return
    
def cube (center = (mid, mid, mid), size = 10):
    """
    

    Parameters
    ----------
    center : TYPE, optional
        list of the center values of the cube. The default is (mid, mid, mid).
    size : TYPE, optional
        determines the size of the cube. The default is 10.

    Returns
    -------
    defaultdict
        graph representing a cube.

    """
    size = size//2
    low  = (center[0] - size, center[1] - size, center[2] - size)
    high = (center[0] + size, center[1] + size, center[2] + size)
    corners = [(low[0], low[1], low[2]), (low[0], low[1], high[2]), (low[0], high[1], low[2]), (low[0], high[1], high[2]), (high[0], low[1], low[2]), (high[0], low[1], high[2]), (high[0], high[1], low[2]), (high[0], high[1], high[2])]
    return generate_RRP(corners)

def movingcube (nodes = {(100, 100, 100), (200, 100, 100), (100, 200, 100), (200, 200, 100), (100, 100, 200), (200, 100, 200), (100, 200, 200), (200, 200, 200)}):
    G = generate_RRP(nodes)
    myclass = rendering(G)
    global screen
    screen = pygame.display.set_mode([window_size, window_size])
    myclass.screen_on = True
    screen.fill(background)
    
    myclass.display()

    running = True
    count = 0
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        myclass.shift("x", 2*(np.sin(count)))
        myclass.shift("y", 2*(np.cos(count)))
        myclass.shift("z", 20*(np.sin(count)))
        screen.fill(background)
        myclass.display()
        count += .01
        pygame.display.update()
        
    pygame.quit()
    return


def rotate_graph (G, direction = "x", point = (mid, mid, mid), speed = 0.1):
    myclass = rendering(G)
    global screen
    screen = pygame.display.set_mode([window_size, window_size])
    myclass.screen_on = True
    screen.fill(background)
    myclass.display()
    running = True
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        myclass.rotate_point(point, direction, speed)
        screen.fill(background)
        myclass.display()
        pygame.display.update()
    pygame.quit()
    return

def BFS_DFS(G, s, rand = [True]):
    '''
    DFS of Graph G from point s
    return set of all explored nodes
    '''
    Stack = deque()
    Stack.append((s, (s[0]-1, s[1])))
    E = {s}
    T = defaultdict(set)
    count = 0
    p = []
    
    
    while Stack:
        count+=1
        if random.choice(rand):
            v, f = Stack.popleft()
        else:
            v, f = Stack.pop()
        for node in G[v]:
            if node not in E:
                Stack.append((node, v))
        if v not in E:
            T[v].add(f)
            T[f].add(v)
            p.append(v)
        E.add(v)
    return T,  p

def generate_cubes_within_bound (bound = ((0, 0, 0), (window_size, window_size, window_size)), length = 100):
    G = defaultdict(set)
    node_dist = [(bound[1][0] - bound[0][0]) / length, (bound[1][1] - bound[0][1]) / length, (bound[1][2] - bound[0][2]) / length]
    current = list(bound[0])
    def adjacent (node):
        x = node_dist[0]
        y = node_dist[1]
        z = node_dist[2]
        res = []
        res.append((node[0] + x, node[1], node[2]))
        res.append((node[0] - x, node[1], node[2]))
        res.append((node[0], node[1] + y, node[2]))
        res.append((node[0], node[1] - y, node[2]))
        res.append((node[0], node[1], node[2] + z))
        res.append((node[0], node[1], node[2] - z))
        return res
    for x in range(length):
        current[0] += node_dist[0]
        for y in range(length):
            current[1] += node_dist[1]
            for z in range(length):
                current[2] += node_dist[2]
                for i in adjacent(current):
                    G[tuple(current)].add(i)
                    G[i].add(tuple(current))
    return G
                
        

def generate_random_nodes_within_bound (bound = ((0, 0, 0), (window_size, window_size, window_size)), node_num = 100, edge_num = 1000):
    """
    generates random nodes wothin the bound
    """
    G = defaultdict(set)
    nodes = []
    for current_node in range(node_num):
        nodes.append((random.randint(bound[0][0], bound[1][0]), random.randint(bound[0][1], bound[1][1]), random.randint(bound[0][2], bound[1][2])))
    for current_edge in range(edge_num):
        if current_edge < node_num:
            new_node = nodes[current_edge]
            while new_node == nodes[current_edge]:
                new_node = random.choice(nodes)
            G[nodes[current_edge]].add(new_node)
            G[new_node].add(nodes[current_edge])
        else:
            node1 = random.choice(nodes)
            node2 = random.choice(nodes)
            while node1 == node2:
                node1 = random.choice(nodes)
                node2 = random.choice(nodes)
            G[node1].add(node2)
            G[node2].add(node1)
            
    return G


def render_graph (G):
    """
    

    Parameters
    ----------
    G : graph.

    Returns
    -------
    animation renderinbg the graph.

    """
    myclass = rendering(G)
    global screen
    screen = pygame.display.set_mode([window_size, window_size])
    myclass.screen_on = True
    screen.fill(background)
    myclass.display()
    pygame.display.update()
    running = True
    
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    pygame.quit()
    return

def Gsearched (a = 1, b = 1, speed = 0.01, node_num = 50, edge_num = 400):
    G = generate_random_nodes_within_bound(node_num = node_num, edge_num = edge_num)
    T, P = BFS_DFS(G, random.choice(list(G.keys())), [True] * a + [False] * b)
    rotate_graph(T, speed = speed)

def dist(node1, node2):
    return ((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2 + (node1[2]-node2[2])**2)**.5

def trim (G):
    """
    removes teh wieghts
    """
    for i in G:
        temp = G[i].copy()
        for j in G[i]:
            temp.remove(j)
            temp.add(j[0])
        G[i] = temp
    return G

def prims (G, isWeighted = False):
    """
    

    Parameters
    ----------
    G : default dict
        graph of nodes to run prims on.
    isWeighted : Boolean, optional
        describes of the graph is weighted. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if not isWeighted:
        for i in G:
            temp = G[i].copy()
            for j in G[i]:
                temp.remove(j)
                temp.add((j, dist(j, i)))
            G[i] = temp
    #print(G)
    V = list(G.keys())
    s = random.choice(V)
    V = set(V)
    V.remove(s)
    new = (s, 0)
    heap = [(0, (s, 0), (s, 0))]
    E = {s}
    T = defaultdict(set)
    global MST
    MST = 0
    while V:
        for i in G[new[0]]:
            if i[0] not in E:
                heappush(heap, (i[1], new, i))
        low = heappop(heap)
        while low[2][0] in E and heap:
            low = heappop(heap)
        E.add(low[2][0])
        new = low[2]
        if low[2][0] in V:
            V.remove(low[2][0])
        T[low[1][0]].add(low[2])
        T[low[2][0]].add(low[1])
        MST += low[0]
    #print(T)
    return trim(T)
        
def run_prims():
    """
    Returns
    -------
    animation
        Generates a random graph, runs prims algorithm to clean the nodes, and animates the graph rotating.

    """
    return rotate_graph(prims(generate_random_nodes_within_bound(bound = ((0, 0, 0), (window_size, window_size, window_size)))), speed = 0.01)