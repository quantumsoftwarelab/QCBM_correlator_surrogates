import pennylane as qml
from pennylane.pauli import PauliWord
from pennylane import X, Y, Z
import itertools


def generate_valid_subsets(n):
    """
    Generate all subsets of {1, 2, ..., n} such that:
        - The number of odd elements in the subset is even.
        - The number of even elements in the subset is even.
    Excludes the empty subset.
    """
    odds = [x for x in range(n) if x % 2 == 1]
    evens = [x for x in range(n) if x % 2 == 0]

    valid_subsets = []

    # For each even-sized combination of odd indices
    for k in range(0, len(odds) + 1, 2):
        for odd_subset in itertools.combinations(odds, k):
            # For each even-sized combination of even indices
            for j in range(0, len(evens) + 1, 2):
                for even_subset in itertools.combinations(evens, j):
                    # Combine and add to list if not empty
                    combined = tuple(sorted(odd_subset + even_subset))
                    if combined:  # Exclude empty tuple
                        valid_subsets.append(combined)

    return valid_subsets


def get_combinations(n):
    combs = []
    for i in range(1, n + 1):
        combs.extend(list(itertools.combinations(range(n), i)))
    return combs


def generate_pauli_words(subset):
    """
    Generate all possible Pauli words for a given subset of indices
    using Y and Z operators.
    """
    operators = ["Y", "Z"]
    pauli_words = []

    # Generate all combinations of Y and Z for the given subset
    for combo in itertools.product(operators, repeat=len(subset)):
        # Map each index in the subset to its corresponding operator
        word_dict = {index: op for index, op in zip(subset, combo)}
        # Construct the PauliWord
        pauli_word = PauliWord(word_dict)
        pauli_words.append(pauli_word)

    return pauli_words


def remove_X_terms(X_terms, n):
    even_inds_0 = [idx for idx in range(n) if idx % 2 == 0]
    even_inds_1 = [idx for idx in range(n) if idx % 2 != 0]
    all_indcs = list(range(n))
    remove_indcs_list = [even_inds_0, even_inds_1, all_indcs]
    for remove_list in remove_indcs_list:
        X_remove = X(remove_list[0])
        for idx in remove_list[1:]:
            X_remove = X_remove @ X(idx)
        X_terms.remove(X_remove)
    return X_terms


def get_algebra(hamiltonian, n):
    if hamiltonian == "haldane":
        combinations = get_combinations(n)
        X_terms = []
        for combination in combinations:
            Xs = X(combination[0])
            for i in range(1, len(combination)):
                Xs = Xs @ X(combination[i])
            X_terms.append(Xs)
        # list of all pauli strings with even number of Ys and Zs
        indices = generate_valid_subsets(n)
        YZ_terms = []
        for index in indices:
            YZ_terms += generate_pauli_words(index)
        galg = [term.pauli_rep for term in YZ_terms]

        galg += [
            X_term.pauli_rep @ term
            for term in galg
            for X_term in X_terms
            if not any([wire in X_term.wires for wire in term.wires])
        ]
        X_terms = remove_X_terms(X_terms, n)
        galg += X_terms
    elif hamiltonian == "heisenberg":
        indices = [(i, i + 1) for i in range(0, n - 1)]
        XX_terms = [X(idx0) @ X(idx1) for idx0, idx1 in indices]
        YY_terms = [Y(idx0) @ Y(idx1) for idx0, idx1 in indices]
        ZZ_terms = [Z(idx0) @ Z(idx1) for idx0, idx1 in indices]
        group = XX_terms + YY_terms + ZZ_terms
        galg = qml.lie_closure(group)
    elif hamiltonian == "matchgate":
        indices = [(i, i + 1) for i in range(0, n - 1)]
        XX_terms = [X(idx0) @ X(idx1) for idx0, idx1 in indices]
        Z_terms = [Z(idx) for idx in range(0, n)]
        MG = XX_terms + Z_terms
        galg = qml.lie_closure(MG)
    return [list((term).pauli_rep)[0] for term in galg]
