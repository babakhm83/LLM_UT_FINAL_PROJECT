# ──────── Libraries ────────────────────────────────────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import ParameterSampler
import time
from z3 import *
import itertools
import matplotlib.pyplot as plt
import re
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ──────── General Utils ────────────────────────────────────────────────────────────────────────────────────────────────────────


def run_baseline(chat_model, system_message, human_message):
    messages = [system_message, human_message]
    start_time = time.time()
    answer = chat_model.invoke(messages).content
    end_time = time.time()
    execution_time = end_time - start_time
    return answer, execution_time


def add_z3_result(results, z3_solver):
    if not isinstance(results, list):
        results = [results]
    for result in results:
        z3_solution, solve_time = z3_solver(*result['Generated'][:-1])
        result['Z3 Solution'] = z3_solution
        result['Z3 Sovle Time'] = solve_time


def print_one_result(result):
    print("-" * 75)
    for key, val in result.items():
        print(f"{key}:\n{val}")
        print("-" * 75)


class TestProblemArgs():
  def __init__(self, chat_model=None, generator=None, verifier=None, 
               z3_solver=None, answer_extractor=None, problem_formatter=None, 
               prompt_creator=None, generator_args=None):

    if chat_model is not None:
      self.chat_model = chat_model
    if generator is not None:
      self.generator = generator
    if verifier is not None:
      self.verifier = verifier
    if z3_solver is not None:
      self.z3_solver = z3_solver
    if answer_extractor is not None:
      self.answer_extractor = answer_extractor
    if problem_formatter is not None:
      self.problem_formatter = problem_formatter
    if prompt_creator is not None:
      self.prompt_creator = prompt_creator
    if generator_args is not None:
      self.generator_args = generator_args


def test_one_problem(test_args: TestProblemArgs):
    result = {}
    result['Parameters'] = test_args
    result['Generator Parameters'] = test_args.generator_args

    generated = test_args.generator(**test_args.generator_args)
    result['Generated'] = generated

    problem_str = test_args.problem_formatter(*generated[:-1])
    result['Formatted Problem'] = problem_str

    system_message, human_message = test_args.prompt_creator(problem_str)
    result['System Message'] = system_message
    result['Human Message'] = human_message

    llm_response, llm_solve_time = run_baseline(
        test_args.chat_model, system_message, human_message)
    result['LLM Response'] = llm_response
    result['LLM Solve Time'] = llm_solve_time

    llm_answer = test_args.answer_extractor(llm_response)
    result['Extracted Answer'] = llm_answer

    is_correct, message = test_args.verifier(llm_answer, *generated)
    result['Is Correct'] = is_correct
    result['Message'] = message

    return result


def run_baseline_batched(chat_model, system_messages, human_messages):
    message_batches = []
    for system, human in zip(system_messages, human_messages):
        message_batches.append([system, human])

    start_time = time.time()
    responses = chat_model.batch(message_batches)
    end_time = time.time()

    execution_time = (end_time - start_time) / len(system_messages)

    answers = [resp.content for resp in responses]
    times = [execution_time] * len(answers)

    return answers, times


def test_many_problems(args, n_problems, random_state=42):
    results = []
    generator_args_list = list(ParameterSampler(
        param_distributions=args.generator_args,
        n_iter=n_problems,
        random_state=random_state
    ))

    system_messages = []
    human_messages = []

    for generator_args in generator_args_list:
        args.generator_args = generator_args
        result = {}
        result['Parameters'] = args
        result['Generator Parameters'] = generator_args

        generated = args.generator(**generator_args)
        result['Generated'] = generated

        problem_str = args.problem_formatter(*generated[:-1])
        result['Formatted Problem'] = problem_str

        system_message, human_message = args.prompt_creator(problem_str)
        result['System Message'] = system_message
        result['Human Message'] = human_message

        system_messages.append(system_message)
        human_messages.append(human_message)
        results.append(result)

    llm_responses, llm_solve_times = run_baseline_batched(
        args.chat_model, system_messages, human_messages)
    for i, (llm_response, llm_time) in enumerate(zip(llm_responses, llm_solve_times)):
        result = results[i]
        result['LLM Response'] = llm_response
        result['LLM Solve Time'] = llm_time

        llm_answer = args.answer_extractor(llm_response)
        result['Extracted Answer'] = llm_answer

        is_correct, message = args.verifier(llm_answer, *result['Generated'])
        result['Is Correct'] = is_correct
        result['Message'] = message

    return results


def scatter_plot_correctness(X, y):
    """
    Scatterplot the correctness of the solution for every pair of parameters
    """
    for c1,c2 in itertools.combinations(X.columns, 2):
        x1 = X[c1]
        x2 = X[c2]
        z = ['green' if res else 'red' for res in y.values]

        plt.scatter(x1, x2, c=z)
        plt.xlabel(c1)
        plt.ylabel(c2)
        plt.title(f"Problem Success/Failure by {c1} and {c2}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_correctness_distribution(X, y, n_bins=7):
    """
    Plot the proportion of correct solutions for each bin of each feature
    """
    for feature in X.columns:
        plt.figure(figsize=(6, 4))

        feature_values = X[feature]
        bins = np.linspace(feature_values.min(), feature_values.max(), n_bins+1)

        proportions = []
        bin_centers = []

        for i in range(len(bins) - 1):
            lower_bound = bins[i]
            upper_bound = bins[i + 1]
            bin_center = (lower_bound + upper_bound) / 2

            in_bin_mask = (feature_values >= lower_bound) & (feature_values < upper_bound)

            if in_bin_mask.sum() > 0:
                correct_in_bin = y[in_bin_mask].sum()
                total_in_bin = in_bin_mask.sum()
                proportion = correct_in_bin / total_in_bin

                proportions.append(proportion)
                bin_centers.append(bin_center)

        plt.bar(bin_centers, proportions, width=(bins[1]-bins[0])*0.8)
        plt.xlabel(feature)
        plt.ylabel('Proportion Correct')
        plt.title(f'Proportion of Correct Solutions by {feature}')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_feature_importance(X, y):
    """
    Train a classifier and plot the feature importance
    """
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    importance = model.feature_importances_
    indices = np.argsort(importance)

    plt.figure(figsize=(6, len(feature_names)))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

def report_results(results):
    df = pd.DataFrame([res['Generator Parameters'] for res in results]).copy()
    df['Is Correct'] = [res['Is Correct'] for res in results]
    df['Length of prompt'] = [
        len(res['Human Message']['content']) for res in results]

    features = list(key for key in results[0]['Generator Parameters'].keys()
        if type(results[0]['Generator Parameters'][key])!=bool)
    features.append('Length of prompt')

    X = df[features]
    y = df['Is Correct']

    scatter_plot_correctness(X, y)
    plot_correctness_distribution(X, y)

    plot_feature_importance(X.drop(columns=['Length of prompt']), y)

    print("LLM Solve Time (Average):", np.mean([res['LLM Solve Time'] for res in results]))
    print("LLM Solve Time (STD):", np.std([res['LLM Solve Time'] for res in results]))
    if 'Z3 Solve Time' in results[0].keys():
        print("Z3 Solve Time (Average):", np.mean([res.get('Z3 Solve Time',0) for res in results]))
        print("Z3 Solve Time (STD):", np.std([res.get('Z3 Solve Time',0) for res in results]))

def save_results(results, path):
    save_results = [dict() for res in results]

    for i, key in enumerate(results.keys()):
        if key == 'Parameters':
            continue
        save_results[i][key] = results[i][key]

    with open(path, 'wb') as f:
        pickle.dump(save_results, f)


def load_results(path):
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results

# ──────── Baseline ────────────────────────────────────────────────────────────────────────────────────────────────────────

# ──────── 3 Sat Problem ──────────────────────────────────────────────────────────────────────────────────────────────────


def format_sat_problem_for_llm(clauses):
    sat_problem_parts = []
    for clause in clauses:
        temp_clause = [
            f"{'~' if var < 0 else ''}x{abs(var)}" for var in clause]
        str_clause = "(" + " v ".join(temp_clause) + ")"
        sat_problem_parts.append(str_clause)
    return " ∧ ".join(sat_problem_parts)


def create_sat_prompt(formatted_problem_str):
    system_content = (
        "You are an expert logic engine specializing in Boolean Satisfiability "
        "Problems (SAT). You will receive a 3-SAT problem instance, which consists "
        "of a conjunction of clauses, where each clause is a disjunction of three "
        "literals. Your task is to determine if a satisfying assignment of boolean "
        "values (True or False) exists for the variables that makes the entire "
        "formula True. If a solution exists, you must provide a valid satisfying "
        "assignment. If the formula is unsatisfiable, you must state that.\n\n"

        "CRITICAL REQUIREMENTS:\n"
        "- The solution must satisfy all clauses\n"
        "- Show your reasoning process step by step\n"
        "- Return only list all True variables in one line and all False variables in the other or the statement of unsatisfiablity.\n\n"

        "Example Output Format:\n"
        "True:\n"
        "[x1, x4, ...]"
        "False:\n"
        "[x2, x3, ...]"
    )

    human_content = (
        f"Please solve the following 3-SAT problem:\n\n{formatted_problem_str}\n\n"

        "Follow these steps to find a solution:\n"
        "1.  **Analyze the Clauses**\n"
        "    * Identify all unique variables in the clauses.\n"
        "2.  **Initial Solution**\n"
        "    * Start with an initial assignment for the variables. A good starting point is to set all variables to `True`.\n"
        "    * Check each clause to see if it is satisfied by your current assignment. A clause is satisfied if at least one of its literals is `True`.\n"
        "3.  **Iterate and Refine**:\n"
        "    * If all clauses are satisfied, you have found a solution. Present the final assignment.\n"
        "    * If not, you need to adjust your assignment. Identify the unsatisfied clauses and flip the value of variables in to satisfy them.\n"
        "    * Repeat until all clauses are satisfied.\n"
        "5.  **Conclusion**\n"
        "    * Provide the final assignment or state unsatisfiability in the required format without any additional texts.\n"
    )

    sys_msg = SystemMessage(content=system_content)
    human_msg = HumanMessage(content=human_content)

    return sys_msg, human_msg


def extract_assignment_from_output(llm_output):
    assignment = {}
    llm_output = llm_output.lower()

    try:
        conclusion_part = llm_output.split("conclusion")[-1]
        true_part = conclusion_part.split("true:")[1]
        false_part = conclusion_part.split("false:")[1]
    except IndexError:
        return {}

    pattern = re.compile(r"x(\d+)")

    for match in pattern.findall(true_part.strip()):
        var_index = int(match)
        assignment[var_index] = True

    for match in pattern.findall(false_part.strip()):
        var_index = int(match)
        assignment[var_index] = False

    return assignment

# ──────── Subset Sum Problem ───────────────────────────────────────────────────────────────────────────────────────────────────


def format_subset_sum_problem_for_llm(target, number_set):
    numbers_part = " ".join(str(number) for number in number_set)
    prompt = f"Numbers are\n{numbers_part}\nTarget is {target}"
    return prompt


def create_subset_sum_prompt(formatted_problem_str):
    system_content = (
        "You are an expert logic engine specializing in Subset Sum "
        "Problems (SSP). You will receive an SSP instance consisting of a list of "
        "positive numbers and a target sum. Your task is to determine if there exists "
        "a subset of these numbers whose sum equals the target. "
        "If a solution exists, you must provide the subset. If no solution exists, "
        "clearly state so.\n\n"

        "CRITICAL REQUIREMENTS:\n"
        "- The sum of all elements in the subset must be equal to the target\n"
        "- Show your reasoning process step by step\n"
        "- Return only list of selected numbers or state that no solution can be found.\n\n" |

        "Example Output Format:\n"
        "[1, 4, ...]"
    )

    human_content = (
        f"Please solve the following SSP:\n\n{formatted_problem_str}\n\n"

        "Follow these steps to find a solution:\n"
        "1.  **Analyze the Input**\n"
        "    * List all numbers and the target sum.\n"
        "    * Start by identifying potential subsets.\n"
        "2.  **Initial Solution**\n"
        "    * Start with an initial subset of the numbers, such as the k smallest or largest numbers.\n"
        "    * Compute the sum of the selected subset and compare it to the target.\n"
        "3.  **Iterate and Refine Subset**\n"
        "    * If the sum equals the target, you have found a solution. output the subset as the solution.\n"
        "    * If not, you need to adjust the subset. Adjust the subset by adding or removing numbers based on the difference from the target.\n"
        "    * Repeat until a solution is found or all feasible subsets are exhausted.\n"
        "4.  **Conclusion**\n"
        "    * If a subset is found, provide the final subset in the required format without any additional texts. If no subset sums to the target, state 'No solution exists.'\n"
    )

    sys_msg = SystemMessage(content=system_content)
    human_msg = HumanMessage(content=human_content)

    return sys_msg, human_msg


def extract_subset_from_output(llm_output):
    try:
        conclusion_part = llm_output.split("[")[-1]
    except IndexError:
        return {}

    pattern = re.compile(r"\d+", re.IGNORECASE)
    subset = pattern.findall(conclusion_part.strip())
    return [int(number) for number in subset]

# ──────── Minimum Vertex Cover Problem ─────────────────────────────────────────────────────────────────────────────────────────


def format_graph_problem_for_llm(generated_graph):
    prompt = (f"Total vertices: {len(generated_graph)}\n"
              "Generated Graph:\n"
              )
    for vertex, neighbors in sorted(generated_graph.items()):
        prompt += f"   {vertex}: {sorted(neighbors)}\n"
    return prompt


def create_vc_prompt(formatted_problem_str):
    system_content = (
        "You are an expert graph algorithm solver specializing in Minimum "
        "Vertex Cover Problems. You will receive an undirected graph where "
        "each line contains a vertex and its adjacent vertices. Your task "
        "is to find the smallest possible subset of vertices such "
        "that every edge in the graph has at least one endpoint in this subset.\n\n"

        "CRITICAL REQUIREMENTS:\n"
        "- The solution must be minimal (no smaller valid vertex cover exists)\n"
        "- Every edge must be covered (have at least one endpoint in the selected subset)\n"
        "- Show your reasoning process step by step\n"
        "- Return only the final vertex set as a comma-separated list in brackets without any additional texts., e.g., [v1, v3, v5]"
    )

    human_content = (
        f"Please solve the following Minimum Vertex Cover Problem:\n\n{formatted_problem_str}\n\n"

        "Follow these steps to find a solution:\n"
        "1.  **Graph Analysis**\n"
        "    * Identify all vertices and edges from the input.\n"
        "    * Note any isolated vertices (they don't need to be in the cover).\n"
        "    * Identify high-degree vertices (good candidates for the cover).\n"
        "    * Start by identifying potential vertices, beginning with high-degree vertices.\n"
        "2.  **Initial Solution**\n"
        "    * Start with an initial subset of the vertices.\n"
        "    * Ensure all edges are covered by your initial selection.\n"
        "3.  **Iterate and Refine the Subset**\n"
        "    * If all the edges are covered, try to minimize the size of the subset by adjusting it.\n"
        "    * If not, you need to adjust the subset to cover them all.\n"
        "    * Adjust the subset by adding or removing vertices.\n"
        "    * Continue until no further reduction is possible and all the edges are covered.\n"
        "4.  **Conclusion**\n"
        "    * Provide the final set of vertices in the required format without any additional texts.\n"
    )

    sys_msg = SystemMessage(content=system_content)
    human_msg = HumanMessage(content=human_content)

    return sys_msg, human_msg

# ──────── Maximum Clique Problem ───────────────────────────────────────────────────────────────────────────────────────────────


def create_clique_prompt(formatted_problem_str):
    system_content = (
        "You are an expert in graph theory specializing in Maximum "
        "Clique Problems. You will receive an undirected graph where "
        "each line contains a vertex and its adjacent vertices. Your task "
        "is to find the maximum clique in the graph.\n\n"

        "CRITICAL REQUIREMENTS:\n"
        "- The solution must be maximum (no greater clique exists)\n"
        "- Show your reasoning process step by step\n"
        "- Return only the final set of vertices as a comma-separated list in brackets without any additional texts., e.g., [v1, v3, v5]"
    )

    human_content = (
        f"Please solve the following Maximum Clique Problem:\n\n{formatted_problem_str}\n\n"

        "Follow these steps to find a solution:\n"
        "1.  **Graph Analysis**\n"
        "    * Identify all vertices and edges from the input.\n"
        "    * Identify high-degree vertices (good candidates for the cover).\n"
        "    * Start by identifying potential cliques, beginning with high-degree vertices.\n"
        "2.  **Initial Solution**\n"
        "    * Start with an initial subset of the vertices.\n"
        "    * Ensure the subset produces a clique.\n"
        "3.  **Iterate and Refine**\n"
        "    * If the subset produces a clique, try to maxmize the size of the subset by adjusting it.\n"
        "    * If not, you need to add or remove vertices the subset to reach a clique.\n"
        "    * Adjust the subset by adding or removing vertices.\n"
        "    * Repeat until no further refinement is possible and the greatest clique is found.\n"
        "4.  **Conclusion**\n"
        "    * Provide the final set of vertices in the required format without any additional texts.\n"
    )

    sys_msg = SystemMessage(content=system_content)
    human_msg = HumanMessage(content=human_content)

    return sys_msg, human_msg

# ──────── Hamiltonian Path Problem ─────────────────────────────────────────────────────────────────────────────────────────────


def create_ham_path_prompt(formatted_problem_str):
    system_content = (
        "You are an expert in graph theory specializing in Hamiltonian "
        "Path Problem. You will receive an undirected graph where "
        "each line contains a vertex and its adjacent vertices. Your task "
        "is to determine if there exists a path that visits every vertex exactly once."
        "If such a path exists, you must provide the sequence of vertices. If not, "
        "you must state that.\n\n"

        "CRITICAL REQUIREMENTS:\n"
        "- The path must visit every vertex exactly once\n"
        "- Consecutive vertices in the path must be connected by an edge\n"
        "- Show your reasoning process step by step\n"
        "- If a path exists, return only the final sequence of vertices as a comma-separated list in brackets without any additional texts., e.g., [v1, v3, v5]"
        "- If no path exists, return 'NO HAMILTONIAN PATH EXISTS'\n"
    )

    human_content = (
        f"Please solve the following Hamiltonian Path Problem:\n\n{formatted_problem_str}\n\n"

        "Follow these steps to find a solution:\n"
        "1.  **Graph Analysis**\n"
        "    * Identify all vertices and edges from the input.\n"
        "    * Identify high-degree and low-degree vertices.\n"
        "    * Check necessary conditions:\n"
        "        Graph must be connected.\n"
        "        At most 2 vertices can have odd degree.\n"
        "        Vertices with degree 1 can only be endpoints.\n"
        "2.  **Initial Solution**\n"
        "    * Start with an initial sequence of the vertices.\n"
        "    * If exactly 2 vertices have odd degree, start from one of them.\n"
        "    * If all vertices have even degree, start from any vertex.\n"
        "    * Consider vertices with degree 1 as mandatory endpoints.\n"
        "    * Ensure the path is valid that means there is an edge between any two consequent vertices.\n"
        "3.  **Iterate and Refine**\n"
        "    * If a path is found:\n\n"
        "        Verify it contains all vertices exactly once.\n"
        "        Verify each consecutive pair is connected by an edge.\n"
        "        Present the path as [v1, v2, v3, ...].\n"
        "    * If no path exists:\n\n"
        "        Explain why (disconnected graph, degree constraints, etc.).\n"
        "        Return 'NO HAMILTONIAN PATH EXISTS'.\n"
        "4.  **Conclusion**\n"
        "    * Provide the final cycle in the required format without any additional texts or state that no hamiltonian path exists.\n"
    )

    sys_msg = SystemMessage(content=system_content)
    human_msg = HumanMessage(content=human_content)

    return sys_msg, human_msg

# ──────── Hamiltonian Cycle Problem ─────────────────────────────────────────────────────────────────────────────────────────────


def create_ham_cycle_prompt(formatted_problem_str):
    system_content = (
        "You are an expert in graph theory specializing in Hamiltonian "
        "Cycle Problem. You will receive an undirected graph where "
        "each line contains a vertex and its adjacent vertices. Your task "
        "is to determine if there exists a cycle that visits every vertex exactly once "
        "and then returns to the first vertex. If such a path exists, you must provide "
        "the sequence of vertices. If not, you must state that.\n\n"

        "CRITICAL REQUIREMENTS:\n"
        "- The cycle must visit every vertex exactly once except for the first vertex that is also the last vertex\n"
        "- Consecutive vertices in the cycle must be connected by an edge\n"
        "- Show your reasoning process step by step\n"
        "- The first vertex must also be the last vertex\n"
        "- If a Hamiltonian cycle exists, return only the final sequence of vertices as a comma-separated list in brackets without any additional texts., e.g., [v1, v3, v1]"
        "- If no Hamiltonian cycle exists, return 'NO HAMILTONIAN CYCLE EXISTS'.\n"
    )

    human_content = (
        f"Please solve the following Hamiltonian Cycle Problem:\n\n{formatted_problem_str}\n\n"

        "Follow these steps to find a solution:\n"
        "1.  **Graph Analysis**:\n"
        "    * Identify all vertices and edges from the input.\n"
        "    * Identify high-degree and low-degree vertices.\n"
        "2.  **Initial Solution**:\n"
        "    * Start with an initial cycle.\n"
        "    * Ensure the sequence is valid cycle that means there is an edge between any two consequent vertices.\n"
        "3.  **Iterate and Refine**:\n"
        "    * If a cycle is found:\n\n"
        "        Verify each vertex (except the first, which repeats at the end) appears exactly once.\n"
        "        Verify each consecutive pair is connected by an edge.\n"
        "        Present the cycle as [v1, v2, v3, ..., v1].\n"
        "    * If no cycle exists:\n\n"
        "        Explain why (disconnected graph, etc.).\n"
        "        Return 'NO HAMILTONIAN CYCLE EXISTS'.\n"
        "4.  **Conclusion**:\n"
        "    * Provide the final cycle in the required format without any additional texts or state that no hamiltonian cycle exists.\n"
    )

    sys_msg = SystemMessage(content=system_content)
    human_msg = HumanMessage(content=human_content)

    return sys_msg, human_msg
