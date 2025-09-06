# ──────── Libraries ────────────────────────────────────────────────────────────────────────────────────────────────────────────

import time
import random
from z3 import *
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────── 3 Sat Problem ────────────────────────────────────────────────────────────────────────────────────────────────────────

def generate_3sat_instance(num_vars, num_clauses, use_random_weights=True):
    true_probability = random.random()
    false_probability = 1 - true_probability
    if (use_random_weights):
        solution = {i: random.choices([True, False], weights=[true_probability, false_probability])[0] for i in range(1, num_vars + 1)}
    else:
        solution = {i: random.choice([True, False]) for i in range(1, num_vars + 1)}

    clauses = []
    variables = list(range(1, num_vars + 1))

    for _ in range(num_clauses):
        clause_vars = random.sample(variables, 3)
        clause = [v * random.choice([-1, 1]) for v in clause_vars]

        is_satisfied = False
        for literal in clause:
            var_index = abs(literal)
            var_value = solution[var_index]
            if (literal > 0 and var_value is True) or (literal < 0 and var_value is False):
                is_satisfied = True
                break
        
        if is_satisfied:
            clauses.append(clause)
        else:
            literal_to_flip = random.choice([0, 1, 2])
            clause[literal_to_flip] *= -1
            clauses.append(clause)
                
    return clauses, solution

def visualize_3sat_instance(clauses: list[list[int]], solution: dict[int, bool]):
    G = nx.Graph()
    num_vars = len(solution)
    variables = list(range(1, num_vars + 1))
    
    var_nodes = [f"x{i}" for i in variables]
    clause_nodes = [f"c{i}" for i in range(len(clauses))]

    G.add_nodes_from(var_nodes, bipartite=0)
    G.add_nodes_from(clause_nodes, bipartite=1)

    for i, clause in enumerate(clauses):
        for literal in clause:
            var_index = abs(literal)
            edge_style = "solid" if literal > 0 else "dashed"
            edge_color = "limegreen" if literal > 0 else "tomato"
            G.add_edge(f"x{var_index}", f"c{i}", style="solid", color=edge_color)

    plt.figure(figsize=(8, 6))
    pos = dict()
    pos.update((node, (1, i)) for i, node in enumerate(var_nodes))
    pos.update((node, (2, i * (len(var_nodes) / len(clause_nodes)))) for i, node in enumerate(clause_nodes))
    var_colors = ["limegreen" if solution[i] else "tomato" for i in variables]
    nx.draw_networkx_nodes(G, pos, nodelist=var_nodes, node_color=var_colors, node_shape="o", node_size=1000)
    nx.draw_networkx_nodes(G, pos, nodelist=clause_nodes, node_color="skyblue", node_shape="s", node_size=1000)

    limegreen_edges = [(u, v) for u, v, d in G.edges(data=True) if d["color"] == "limegreen"]
    tomato_edges = [(u, v) for u, v, d in G.edges(data=True) if d["color"] == "tomato"]
    nx.draw_networkx_edges(G, pos, edgelist=limegreen_edges, edge_color="limegreen", style="solid", width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=tomato_edges, edge_color="tomato", style="solid", width=1.5)

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")    
    plt.title("3-SAT Problem", size=16)
    
    true_patch = mpatches.Patch(color="limegreen", label="Variable = TRUE")
    false_patch = mpatches.Patch(color="tomato", label="Variable = FALSE")
    clause_patch = mpatches.Patch(color="skyblue", label="Clause Node")
    pos_lit_line = plt.Line2D([0], [0], color="limegreen", lw=1.5, linestyle="-", label="Positive Literal (x)")
    neg_lit_line = plt.Line2D([0], [0], color="tomato", lw=1.5, linestyle="-", label="Negative Literal (~x)")
    plt.legend(handles=[true_patch, false_patch, clause_patch, pos_lit_line, neg_lit_line], loc="best")
    plt.tight_layout()
    plt.show()

def verify_3sat_solution(assignment, clauses):
    for clause in clauses:
        is_clause_satisfied = False
        for literal in clause:
            var_index = abs(literal)
            
            if var_index not in assignment:
                return False, f"Verification FAILED: Variable {var_index} not found in the assignment."

            var_value = bool(assignment[var_index])
            if (literal > 0 and var_value is True) or (literal < 0 and var_value is False):
                is_clause_satisfied = True
                break

        if not is_clause_satisfied:
            return False, f"Verification FAILED: Clause {clause} is not satisfied by the assignment."

    return True, "Verification PASSED: The assignment satisfies all clauses."

def solve_3sat_with_z3(clauses, num_vars):
    start_time = time.time()
    solver = Solver()
    bool_vars = {i: Bool(f"x{i}") for i in range(1, num_vars + 1)}

    for clause in clauses:
        z3_clause = []
        for literal in clause:
            var_index = abs(literal)
            z3_var = bool_vars[var_index]
            
            if literal > 0:
                z3_clause.append(z3_var)
            else:
                z3_clause.append(Not(z3_var))
        
        solver.add(Or(z3_clause))

    check_result = solver.check()
    end_time = time.time()
    execution_time = end_time - start_time

    if check_result == sat:
        model = solver.model()
        solution = {i: bool(model.eval(bool_vars[i])) for i in range(1, num_vars + 1)}
        return solution, execution_time
    else:
        return None, execution_time
    
# ──────── Subset Sum Problem ───────────────────────────────────────────────────────────────────────────────────────────────────

def generate_subset_sum_instance(set_size, subset_ratio, min_val=10, range_val=100):
    subset_size = int(set_size * subset_ratio)
    max_val = min_val + range_val
    
    solution_subset = [random.randint(min_val, max_val) for _ in range(subset_size)]
    target_sum = sum(solution_subset)

    other_count = set_size - subset_size
    others = [random.randint(min_val, max_val) for _ in range(other_count)]

    full_set = solution_subset + others
    random.shuffle(full_set)

    return target_sum, full_set, sorted(solution_subset)

def visualize_subset_sum(full_set, target_sum, solution_subset):
    colors = []
    solution_copy = list(solution_subset) 
    
    for num in full_set:
        if num in solution_copy:
            colors.append("limegreen")
            solution_copy.remove(num) 
        else:
            colors.append("lightgray")

    x_labels = [str(n) for n in full_set]
    x_pos = range(len(full_set))

    plt.figure(figsize=(8, 6))
    plt.bar(x_pos, full_set, color=colors, tick_label=x_labels)
    
    plt.ylabel("Value of Number")
    plt.xlabel("Numbers in the Set")
    plt.title(f"Subset Sum Problem (Target Sum = {target_sum})", size=16)
    
    solution_patch = mpatches.Patch(color="limegreen", label="Numbers in Solution Subset")
    other_patch = mpatches.Patch(color="lightgray", label="Other Numbers")
    plt.legend(handles=[solution_patch, other_patch], loc="best")

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def visualize_subset_sum_comparison(target_sum: int, solution_subset: list[int]):
    subset_sum = sum(solution_subset)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(solution_subset))]

    fig, ax = plt.subplots(figsize=(7, 2))
    ax.barh(y=1, width=target_sum, color="limegreen", edgecolor="black", height=0.5, label="Target Sum", alpha=0.8)

    start = 0
    for i, val in enumerate(solution_subset):
        ax.barh(y=0, width=val, left=start, color=colors[i], edgecolor="black", height=0.5)
        start += val

    ax.text(target_sum + target_sum * 0.02, 1, f"= {target_sum}", va="center", ha="left", fontsize=10)
    ax.text(subset_sum + target_sum * 0.02, 0, f"= {subset_sum}", va="center", ha="left", fontsize=10)


    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Subset Sum", "Target Sum"])
    ax.set_xlim(0, max(target_sum, subset_sum) * 1.15)
    ax.set_xlabel("Sum Value")
    ax.set_title("Subset Sum vs Target Sum", fontsize=16)
    
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def verify_subset_sum_solution(target_sum, solution_subset):
    actual_sum = sum(solution_subset)
    if actual_sum == target_sum:
        return True, f"Verification PASSED: Subset sum {actual_sum} matches target {target_sum}."
    else:
        return False, f"Verification FAILED: Subset sum {actual_sum} does not match target {target_sum}."

def solve_subset_sum_with_z3(target_sum, number_set):
    start_time = time.time()
    solver = Solver()
    bool_vars = [Bool(f"b_{i}") for i in range(len(number_set))]
    sum_constraint = Sum([If(bool_vars[i], number_set[i], 0) for i in range(len(number_set))]) == target_sum
    solver.add(sum_constraint)

    check_result = solver.check()
    end_time = time.time()
    execution_time = end_time - start_time

    if check_result == sat:
        model = solver.model()
        solution_subset = [number_set[i] for i, var in enumerate(bool_vars) if is_true(model.eval(var))]
        return sorted(solution_subset), execution_time
    else:
        return None, execution_time

# ──────── Minimum Vertex Cover Problem ─────────────────────────────────────────────────────────────────────────────────────────

def add_edge(graph, u, v):
    if u not in graph[v]:
        graph[u].append(v)
        graph[v].append(u)

def generate_vc_problem(num_vertices, vc_ratio = 0.4, edge_density = 0.5):
    all_vertices = list(range(num_vertices))
    random.shuffle(all_vertices)
    
    vc_size = max(1,int(num_vertices*vc_ratio))
    min_vc = all_vertices[:vc_size]
    independent_set = all_vertices[vc_size:]
    
    graph = {i: [] for i in range(num_vertices)}
    for u in min_vc:
        v = random.choice(independent_set)
        add_edge(graph, u, v)

    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            if u in independent_set and v in independent_set:
                continue

            if random.random() < edge_density:
                add_edge(graph, u, v)
                
    return graph, min_vc

def visualize_vc_graph(graph, vc_solution):
    G = nx.Graph(graph)
    
    node_colors = []
    color_vc = "lightcoral"
    color_independent = "skyblue"
    
    for node in G.nodes():
        if node in vc_solution:
            node_colors.append(color_vc)
        else:
            node_colors.append(color_independent)
            

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=10, font_weight="bold", edge_color="gray")
    
    vc_patch = mpatches.Patch(color=color_vc, label="Vertex Cover")
    is_patch = mpatches.Patch(color=color_independent, label="Independent Set")
    plt.legend(handles=[vc_patch, is_patch], loc="best")
    plt.title("Minimum Vertex Cover Problem", size=15)
    plt.show()

def verify_vc_solution(solution, graph, expected_size):
    if len(solution) != expected_size:
        return False, f"Verification FAILED: Incorrect size (Expected {expected_size} vertices but got {len(solution)})"
    
    solution_set = set(solution)
    uncovered_edges = []
    visited_edges = set()

    for u in graph:
        for v in graph[u]:
            edge = tuple(sorted((u, v)))
            if edge not in visited_edges:
                if u not in solution_set and v not in solution_set:
                    uncovered_edges.append(edge)
                visited_edges.add(edge)

    if len(uncovered_edges) == 0:
        return True, f"Verification PASSED: The solution is a valid vertex cover."
    else:
        return False, f"Verification FAILED: {len(uncovered_edges)} edges are not covered."

def solve_vertex_cover_with_z3(graph):
    start_time = time.time()
    solver = Optimize()
    vertex_vars = {v: Bool(f"v_{v}") for v in graph.keys()}

    for u in graph:
        for v in graph[u]:
            if u < v:
                solver.add(Or(vertex_vars[u], vertex_vars[v]))

    solver.minimize(Sum([If(vertex_vars[v], 1, 0) for v in graph.keys()]))

    check_result = solver.check()
    end_time = time.time()
    execution_time = end_time - start_time

    if check_result == sat:
        model = solver.model()
        vertex_cover = [v for v in graph.keys() if is_true(model[vertex_vars[v]])]
        return sorted(vertex_cover), execution_time
    else:
        return None, execution_time

# ──────── Maximum Clique Problem ───────────────────────────────────────────────────────────────────────────────────────────────

def add_edge(graph, u, v):
    if u not in graph[v]:
        graph[u].append(v)
        graph[v].append(u)

def generate_clique_problem(num_vertices, clique_ratio, edge_density = 0.5):
    graph = {i: [] for i in range(num_vertices)}
    all_vertices = list(range(num_vertices))
    random.shuffle(all_vertices)
    
    clique_size = max(1,int(num_vertices * clique_ratio))
    max_clique = all_vertices[:clique_size]
    other_vertices = all_vertices[clique_size:]

    clique_edges = list(itertools.combinations(max_clique, 2))
    for u, v in clique_edges:
        add_edge(graph, u, v)

    partitions = [[] for _ in range(clique_size)]
    for i, v in enumerate(other_vertices):
        partitions[i % clique_size].append(v)

    vertex_to_partition = {}
    for i, p in enumerate(partitions):        
        for v in p:
            vertex_to_partition[v] = i

    for u, v in itertools.combinations(other_vertices, 2):
        if vertex_to_partition[u] != vertex_to_partition[v]:
            if random.random() < edge_density:
                add_edge(graph, u, v)

    for i, u in enumerate(max_clique):
        forbidden_partition_index = i % clique_size        
        for v in other_vertices:
            if vertex_to_partition[v] != forbidden_partition_index:
                if random.random() < edge_density:
                    add_edge(graph, u, v)

    return graph, max_clique

def visualize_clique_graph(graph, max_clique_nodes):
    G = nx.Graph(graph)
    node_colors = ['gold' if node in max_clique_nodes else 'lightgray' for node in G.nodes()]
    
    edge_colors = []
    edge_widths = []
    clique_edges = list(itertools.combinations(max_clique_nodes, 2))
    for u, v in G.edges():
        if (u, v) in clique_edges or (v, u) in clique_edges:
            edge_colors.append('red')
            edge_widths.append(2.5)
        else:
            edge_colors.append('gray')
            edge_widths.append(0.8)
            
    # pos = nx.spring_layout(G, seed=42, iterations=100)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, width=edge_widths, node_size=800, font_size=10)
    clique_patch = mpatches.Patch(color='gold', label='Maximum Clique Nodes')
    other_patch = mpatches.Patch(color='lightgray', label='Other Nodes')
    plt.legend(handles=[clique_patch, other_patch], loc='best')
    plt.title("Maximum Clique Problem", size=16)
    plt.show()

def verify_clique_solution(solution, graph, expected_size):
    if len(solution) != expected_size:
        return False, f"Verification FAILED: Incorrect size (Expected {expected_size} vertices but got {len(solution)})"
    if len(solution) > 1:
        missing_edges = []
        for u, v in itertools.combinations(solution, 2):
            if v not in graph[u] and u not in graph[v]:
                missing_edges.append((u, v))

        if len(missing_edges) != 0:
            return False, f"Verification FAILED: {len(missing_edges)} edges are missing."

    return True, f"Verification PASSED: The solution is a valid clique."

def solve_max_clique_with_z3(graph):
    start_time = time.time()
    solver = Optimize()

    vertex_vars = {v: Bool(f"v_{v}") for v in graph.keys()}

    non_adjacent_pairs = []
    all_vertices = list(graph.keys())
    for i in range(len(all_vertices)):
        for j in range(i + 1, len(all_vertices)):
            u, v = all_vertices[i], all_vertices[j]
            if v not in graph[u]:
                non_adjacent_pairs.append((u, v))

    for u, v in non_adjacent_pairs:
        solver.add(Or(Not(vertex_vars[u]), Not(vertex_vars[v])))

    clique_size = Sum([If(vertex_vars[v], 1, 0) for v in graph.keys()])
    solver.maximize(clique_size)

    check_result = solver.check()
    end_time = time.time()
    execution_time = end_time - start_time

    if check_result == sat:
        model = solver.model()
        max_clique = [v for v in graph.keys() if is_true(model.eval(vertex_vars[v]))]
        return sorted(max_clique), execution_time
    else:
        return None, execution_time

# ──────── Hamiltonian Path Problem ─────────────────────────────────────────────────────────────────────────────────────────────

def add_edge(graph, u, v):
    if u not in graph[v]:
        graph[u].append(v)
        graph[v].append(u)

def generate_hamiltonian_path_graph(num_vertices, edge_density = 0.2):
    graph = {i: [] for i in range(num_vertices)}
    nodes = list(range(num_vertices))
    random.shuffle(nodes)
    hamiltonian_path = nodes

    path_edges = set()
    for i in range(num_vertices - 1):
        u = hamiltonian_path[i]
        v = hamiltonian_path[i + 1]
        add_edge(graph, u, v)
        path_edges.add((u, v))

    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            if (u, v) not in path_edges and (v, u) not in path_edges:
                if random.random() < edge_density:
                    add_edge(graph, u, v)
    
    return graph, hamiltonian_path

def visualize_hamiltonian_path_graph(adj_list_graph, path):
    G = nx.Graph(adj_list_graph)
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    
    pos = nx.spring_layout(G, seed=42, iterations=100)    
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='limegreen', width=3.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    plt.title("Hamiltonian Path Problem", size=16)    
    legend_path = mpatches.Patch(color='limegreen', label='Hamiltonian Path Edges')
    legend_other = mpatches.Patch(color='lightgray', label='Other Edges')
    plt.legend(handles=[legend_path, legend_other], loc='best')
    plt.axis('off')
    plt.show()

def verify_hamiltonian_path(path, graph):
    num_vertices = len(graph)
    if len(path) != num_vertices:
        return False, f"Verification FAILED: Incorrect size (Expected {num_vertices} vertices but got {len(path)})"

    if len(set(path)) != num_vertices:
        return False, "Verification FAILED: Path does not visit each vertex exactly once."

    for i in range(num_vertices - 1):
        u = path[i]
        v = path[i + 1]
        if v not in graph[u] and u not in graph[v]:
            return False, f"Verification FAILED: Path is not connected (Missing edge between {u} and {v})"

    return True, "Verification PASSED: The solution is a valid Hamiltonian Path."

def solve_hamiltonian_path_with_z3(graph):
    start_time = time.time()
    num_vertices = len(graph)
    solver = Solver()
    pos_vars = {v: Int(f"pos_{v}") for v in graph.keys()}

    for v in graph.keys():
        solver.add(And(pos_vars[v] >= 0, pos_vars[v] < num_vertices))

    solver.add(Distinct(list(pos_vars.values())))

    all_vertices = list(graph.keys())
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            u, v = all_vertices[i], all_vertices[j]
            if v not in graph[u] and u not in graph[v]:
                solver.add(And(pos_vars[v] != pos_vars[u] + 1, pos_vars[u] != pos_vars[v] + 1))
    
    check_result = solver.check()
    end_time = time.time()
    execution_time = end_time - start_time

    if check_result == sat:
        model = solver.model()
        path_map = {model.eval(pos_vars[v]).as_long(): v for v in graph.keys()}
        hamiltonian_path = [path_map[i] for i in range(num_vertices)]
        return hamiltonian_path, execution_time
    else:
        return None, execution_time

# ──────── Hamiltonian Cycle Problem ────────────────────────────────────────────────────────────────────────────────────────────

def add_edge(graph, u, v):
    if u not in graph[v]:
        graph[u].append(v)
        graph[v].append(u)

def generate_hamiltonian_cycle_graph(num_vertices, edge_density = 0.2):
    graph = {i: [] for i in range(num_vertices)}
    nodes = list(range(num_vertices))
    random.shuffle(nodes)
    hamiltonian_cycle_path = nodes

    cycle_edges = set()
    for i in range(num_vertices):
        u = hamiltonian_cycle_path[i]
        v = hamiltonian_cycle_path[(i + 1) % num_vertices]
        add_edge(graph, u, v)
        cycle_edges.add((u, v))

    for u in range(num_vertices):
        for v in range(u + 1, num_vertices):
            if (u, v) not in cycle_edges and (v, u) not in cycle_edges:
                if random.random() < edge_density:
                    add_edge(graph, u, v)
    
    return graph, hamiltonian_cycle_path

def visualize_hamiltonian_cycle_graph(graph, cycle_path):

    G = nx.Graph(graph)
    hc_edges = []
    for i in range(len(cycle_path)):
        u = cycle_path[i]
        v = cycle_path[(i + 1) % len(cycle_path)]
        hc_edges.append((u, v))

    pos = nx.circular_layout(cycle_path)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=700)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=hc_edges, edge_color="crimson", width=3.0)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    plt.title("Hamiltonian Cycle Problem", size=16)
    legend_cycle = mpatches.Patch(color="crimson", label="Hamiltonian Cycle Edges")
    legend_other = mpatches.Patch(color="lightgray", label="Other Edges")
    plt.legend(handles=[legend_cycle, legend_other], loc="best")
    plt.axis("off")
    plt.show()

def verify_hamiltonian_cycle(path, graph):
    num_vertices = len(graph)
    if len(path) != num_vertices:
        return False, f"Verification FAILED: Incorrect size (Expected {num_vertices} vertices but got {len(path)})"
    
    if len(set(path)) != num_vertices:
        return False, f"Verification FAILED: Path does not visit each vertex exactly once."

    for i in range(num_vertices - 1):
        u = path[i]
        v = path[i + 1]
        if v not in graph[u] and u not in graph[v]:
            return False, f"Verification FAILED: Path is not connected (Missing edge between {u} and {v})"

    last_vertex = path[-1]
    first_vertex = path[0]
    if first_vertex not in graph[last_vertex] and last_vertex not in graph[first_vertex]:
        return False, f"Verification FAILED: Path is not a cycle (Missing closing edge between {last_vertex} and {first_vertex})"

    return True, "Verification PASSED: The solution is a valid Hamiltonian Cycle."

def solve_hamiltonian_cycle_with_z3(graph):
    start_time = time.time()
    num_vertices = len(graph)
    solver = Solver()

    pos_vars = {v: Int(f"pos_{v}") for v in graph.keys()}

    for v in graph.keys():
        solver.add(And(pos_vars[v] >= 0, pos_vars[v] < num_vertices))

    solver.add(Distinct(list(pos_vars.values())))

    all_vertices = list(graph.keys())
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            u, v = all_vertices[i], all_vertices[j]
            if v not in graph[u] and u not in graph[v]:
                solver.add(And((pos_vars[u] + 1) % num_vertices != pos_vars[v], (pos_vars[v] + 1) % num_vertices != pos_vars[u]))

    check_result = solver.check()
    end_time = time.time()
    execution_time = end_time - start_time

    if check_result == sat:
        model = solver.model()
        path_map = {}
        for v in graph.keys():
            position = model.eval(pos_vars[v]).as_long()
            path_map[position] = v
        
        hamiltonian_cycle = [path_map[i] for i in range(num_vertices)]
        return hamiltonian_cycle, execution_time
    else:
        return None, execution_time

# ──────── To Be Continue ───────────────────────────────────────────────────────────────────────────────────────────────────────
