import itertools
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.finite_automaton import State, Symbol


def regex_str_to_dfa(regex_str: str):
    """Convert a regex string to a minimized DFA."""
    regex = Regex(regex_str)
    nfa = regex.to_epsilon_nfa()
    dfa = nfa.minimize()

    return dfa


def generate_strings(dfa, max_length=5):
    """Generate accepted strings up to max_length from a DFA"""
    results = set()
    symbols = list(dfa.symbols)

    for length in range(max_length + 1):
        for combo in itertools.product(symbols, repeat=length):
            if dfa.accepts([*combo]):
                results.add("".join([s.value for s in combo]))

    return sorted(results)


def find_distinguishing_string(dfa1, dfa2, max_length=5):
    """
    Finds a string accepted by one DFA and not the other.
    Returns (string, origin) where origin is 'regex1_only' or 'regex2_only'.
    """
    diff1 = dfa1.get_difference(dfa2)
    diff2 = dfa2.get_difference(dfa1)

    # Generate from diff1: regex1 but not regex2
    for length in range(max_length + 1):
        for combo in itertools.product(dfa1.symbols.union(dfa2.symbols), repeat=length):
            if diff1.accepts([*combo]):
                return "".join(s.value for s in combo), "regex1_only"
            if diff2.accepts([*combo]):
                return "".join(s.value for s in combo), "regex2_only"

    return None, "no_distinguishing_string_found"


def analyze_relationship(dfa1, dfa2):
    """Analyze the relationship between two DFAs"""
    intersection = dfa1.get_intersection(dfa2)
    inter_nonempty = not intersection.is_empty()

    diff1 = dfa1.get_difference(dfa2)
    diff2 = dfa2.get_difference(dfa1)

    if diff1.is_empty() and diff2.is_empty():
        relation = "Equivalent"
    elif diff1.is_empty():
        relation = "Regex1 ⊆ Regex2"
    elif diff2.is_empty():
        relation = "Regex1 ⊇ Regex2"
    elif not inter_nonempty:
        relation = "Disjoint"
    else:
        relation = "Partial Overlap"

    return relation, generate_strings(intersection)


def main():
    regex1 = "a+"
    regex2 = "a*"

    dfa1 = regex_to_dfa(regex1)
    dfa2 = regex_to_dfa(regex2)

    relation, intersection_examples = analyze_relationship(dfa1, dfa2)

    print(f"\nRelation between regex1 and regex2: {relation}")
    print("\nExamples from regex1:")
    for s in generate_strings(dfa1):
        print(f"  {s}")
    print("\nExamples from regex2:")
    for s in generate_strings(dfa2):
        print(f"  {s}")
    if intersection_examples:
        print("\nExamples from intersection:")
        for s in intersection_examples:
            print(f"  {s}")
    else:
        print("\nNo intersection examples found (disjoint or empty intersection).")

    distinguishing_str, origin = find_distinguishing_string(dfa2, dfa1)
    if distinguishing_str:
        print(
            f"\nDistinguishing string: '{distinguishing_str}' is accepted by {origin.replace('_', ' ')}"
        )
    else:
        print(
            "\nNo distinguishing string found (they may be equivalent up to max_length)."
        )


if __name__ == "__main__":
    main()
