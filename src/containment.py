from ltlnode import *
from spotutils import *
from itertools import combinations
import spot

my_formulas = ["G(a)", "F(a)", "X(a)", "a"]
parsed_formulas = [str(parse_ltl_string(f)) for f in my_formulas]

diagram = {}

for f1, f2 in combinations(parsed_formulas, 2):
    # Convert formulas to automata.
    aut1 = spot.translate(f1)
    aut2 = spot.translate(f2)

    # Check equivalence.
    equiv = spot.are_equivalent(aut1, aut2)

    # Check disjointness.
    disjoint = spot.product(aut1, aut2).is_empty()

    # Check subset and superset relationships
    f1_subset_f2 = spot.contains(aut2, aut1)  # f1 ⊆ f2
    f2_subset_f1 = spot.contains(aut1, aut2)  # f2 ⊆ f1

    if equiv:
        print(f"{f1} equiv to {f2}")
        if not diagram.get(f1):
            diagram[f1] = {}

        if not diagram[f1].get("equivalent"):
            diagram[f1]["equivalent"] = []

        diagram[f1]["equivalent"].append(f2)

    elif disjoint:
        print(f"{f1} disjoint from {f2}")
        if not diagram.get(f1):
            diagram[f1] = {}

        if not diagram[f1].get("disjoint"):
            diagram[f1]["disjoint"] = []

        diagram[f1]["disjoint"].append(f2)

    elif f1_subset_f2 and not f2_subset_f1:
        print(f"{f1} subset of {f2}")
        if not diagram.get(f1):
            diagram[f1] = {}

        if not diagram[f1].get("super"):
            diagram[f1]["super"] = []

        diagram[f1]["super"].append(f2)

    elif f2_subset_f1 and not f1_subset_f2:
        print(f"{f1} supserset of {f2}")
        pass

    # Check partial overlap
    elif not disjoint and not equiv:
        print(f"{f1} overlaps {f2}")
        if not diagram.get(f1):
            diagram[f1] = {}

        if not diagram[f1].get("overlap"):
            diagram[f1]["overlap"] = []

        diagram[f1]["overlap"].append(f2)

    else:
        print(f"well shit..., {f1} compared to {f2}")

print("Final Diagram: ")
for formula, relations in diagram.items():
    for relation, forms in relations.items():
        print(f"{formula} has {relation}s: {forms}")

    # areEquivalent
    # isSufficientFor
    # isNecessaryFor
    # areDisjoint
    # generate_traces(f_accepted, f_rejected, max_traces=5)

    # for f in formulas:
    #     for g in formulas:

    test = {
        "(G a)": {"supersets": ["(F a)", "(X a)", "a"], "equivalent": ["(G (G a))"]},
        "(F a)": {"subsets": ["(G a)", "(X a)", "a", "(G (G a))"]},
        "(X a)": {
            "subsets": ["(G a)", "(G (G a))"],
            "supersets": ["(F a)"],
            "overlaps": ["a"],
        },
        "a": {
            "subsets": ["(G a)", "(G (G a))"],
            "supersets": ["(F a)"],
            "overlaps": ["(X a)"],
        },
        "(G (G a))": {"equivalent": ["(G a)"], "supersets": ["(F a)", "(X a)", "a"]},
    }
