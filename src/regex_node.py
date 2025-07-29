import itertools
import random
from collections import deque
from typing import Optional, Iterable
from enum import Enum
import networkx as nx
from cached_method import cached_method
from greenery import parse
from greenery.parse import NoMatch
import copy


class RegexRelationship(str, Enum):
    EQUIVALENT = "Equivalent"
    DISJOINT = "Disjoint"
    SUBSET = "Subset"
    SUPERSET = "Superset"
    OVERLAP = "Partial Overlap"


def fsm_to_digraph(fsm):
    G = nx.DiGraph()
    for state, trans in fsm.map.items():
        for _, target in trans.items():
            G.add_edge(state, target)
    return G


def count_paths_of_len(fsm, max_length):
    """
    Returns a dict: count[state][length] = number of accepted paths of given length from that state.
    """
    count = {state: [0] * (max_length + 1) for state in fsm.states}
    for state in fsm.finals:
        count[state][
            0
        ] = 1  # Base case: accepted path of length 0 ends in a final state

    for length in range(1, max_length + 1):
        for state in fsm.states:
            total = 0
            for charclass, next_state in fsm.map.get(state, {}).items():
                total += count[next_state][length - 1]
            count[state][length] = total
    return count


def get_random_char_from_charclass(charclass, rng):
    possible_chars = list(charclass.get_chars())
    return rng.choice(possible_chars)


def generate_diff_words(
    regex1: str,
    regex2: str,
    num_words: int = 1,
    exclude: list[str] | None = None,
    seed: int | None = None,
) -> list[str]:
    """
    Return up to *num_words* strings accepted by exactly one of the two
    regexes unless the two are equivalent.  All generated strings are of minimum possible length; if a length
    bucket is exhausted we move on to the next-shortest length, and so on.

    Parameters
    ----------
    regex1, regex2 : str
        Regular-expression strings to compare.
    num_words      : int
        How many words to return (default 1).
    exclude        : Iterable[str]
        Words that must *not* be returned.
    seed           : Optional[int]
        Seed forwarded to the internal RNG for reproducibility.

    Returns
    -------
    list[str]
        Up to *num_words* differentiating strings (may be fewer if the
        symmetric difference is finite and smaller than *num_words*).
    """
    if exclude is None:
        exclude = []

    rng = random.Random(seed)

    r1 = RegexNode(regex1)
    r2 = RegexNode(regex2)

    # --- 1.  Compute the *symmetric* difference we need ------------------
    rel = r1.relationship(r2)

    if rel in (
        RegexRelationship.DISJOINT,
        RegexRelationship.OVERLAP,
        RegexRelationship.SUPERSET,
    ):
        diff_node = r1.difference(r2)
    elif rel == RegexRelationship.SUBSET:
        diff_node = r2.difference(r1)
    else:  # Equivalent – nothing can differentiate them
        return []

    if diff_node.empty():
        return []

    # --- 2.  Generate words length-by-length -----------------------------
    words = diff_node.generate_random_words(
        num_words,
        exclude=exclude,
        seed=rng.randint(0, 2**32 - 1),
    )

    return words


class RegexNode:
    """
    A wrapper around Greenery regular expressions that provides regex analysis, comparison,
    and word generation capabilities.
    """

    def __init__(self, regex_str):
        """
        Create a RegexNode from a regex string.

        Parameters
        ----------
        regex_str : str
            The regular expression string.

        Raises
        ------
        ValueError
            If the regex string is invalid and cannot be parsed.
        """
        try:
            self.pattern = parse(regex_str)
        except NoMatch as e:
            raise ValueError(f"Invalid regular expression: {regex_str}") from e

        self.fsm = self.pattern.to_fsm()

    @cached_method
    def _count_table(self, length: int):
        return count_paths_of_len(self.fsm, length)

    @classmethod
    def from_pattern(cls, pattern):
        """
        Create a RegexNode from a greenery pattern object.

        Parameters
        ----------
        pattern : greenery.lego.LEGO
            The parsed regex pattern.

        Returns
        -------
        RegexNode
            A new RegexNode instance.
        """
        node = cls.__new__(cls)  # Bypass __init__
        node.pattern = pattern
        node.fsm = pattern.to_fsm()
        return node

    def __str__(self):
        return str(self.pattern)

    def __eq__(self, other):
        """
        Check whether this regex is equivalent to other.

        Parameters
        ----------
        other : RegexNode

        Returns
        -------
        bool
            True if this language is a strict subset of the other's.
        """
        return self.pattern.equivalent(other.pattern)

    def __lt__(self, other):
        """
        Check whether this regex accepts a proper subset of the other's language.

        Parameters
        ----------
        other : RegexNode

        Returns
        -------
        bool
            True if this language is a strict subset of the other's.
        """
        return self <= other and self != other

    def __le__(self, other):
        """
        Check whether this regex accepts a subset (including equality) of the other's language.

        Parameters
        ----------
        other : RegexNode

        Returns
        -------
        bool
            True if every word accepted by this regex is also accepted by the other.
        """
        return self.pattern.difference(other.pattern).empty()

    def __gt__(self, other):
        """
        Check whether this regex accepts a proper superset of the other's language.

        Parameters
        ----------
        other : RegexNode

        Returns
        -------
        bool
            True if this language is a strict superset of the other's.
        """
        return self >= other and self != other

    def __ge__(self, other):
        """
        Check whether this regex accepts a superset (including equality) of the other's language.

        Parameters
        ----------
        other : RegexNode

        Returns
        -------
        bool
            True if every word accepted by the other regex is also accepted by this one.
        """
        return other.pattern.difference(self.pattern).empty()

    def accepts(self, word):
        """
        Check whether the regex accepts the given word.

        Parameters
        ----------
        word : str

        Returns
        -------
        bool
        """
        return self.pattern.matches(word)

    def empty(self):
        """
        Check whether the regex is empty (accepts no words).

        Returns
        -------
        bool
        """
        return self.pattern.empty()

    def relationship(self, other):
        if self == other:
            return RegexRelationship.EQUIVALENT

        if self.intersect(other).empty():
            return RegexRelationship.DISJOINT

        if self <= other:
            return RegexRelationship.SUBSET

        if self >= other:
            return RegexRelationship.SUPERSET

        return RegexRelationship.OVERLAP

    @cached_method
    def minimum_word_length(self):
        """
        Compute the length of the shortest word accepted by this regex.

        Returns
        -------
        int
            Minimum word length.
        """
        visited = set()
        queue = deque([(self.fsm.initial, 0)])

        while queue:
            state, depth = queue.popleft()
            if state in self.fsm.finals:
                return depth
            if state in visited:
                continue
            visited.add(state)
            for _, next_state in self.fsm.map.get(state, {}).items():
                queue.append((next_state, depth + 1))

        raise ValueError(f"Empty language: {self.pattern}")

    @cached_method
    def maximum_word_length(self):
        """
        Compute the maximum word length accepted by the regex.
        Returns None if the language is infinite.

        Returns
        -------
        Optional[int]
        """
        G = fsm_to_digraph(self.fsm)

        # Check forward and backward reachability
        reachable = nx.descendants(G, self.fsm.initial) | {self.fsm.initial}
        co_reachable = set()
        for final in self.fsm.finals:
            co_reachable |= nx.ancestors(G, final)
            co_reachable.add(final)

        important = reachable & co_reachable
        subgraph = G.subgraph(important)

        # Check for cycles
        try:
            return nx.dag_longest_path_length(subgraph)
        except nx.NetworkXUnfeasible:
            return None  # Contains a cycle → infinite language

    def _sample_branch(self, length: int, rng: random.Random) -> str:
        """One random word of exactly `length`, using the cached table."""
        table = self._count_table(length)
        if table[self.fsm.initial][length] == 0:
            raise ValueError(f"No words of length {length}")
        state = self.fsm.initial
        word = []
        for remaining in range(length, 0, -1):
            total = table[state][remaining]
            choice = rng.randrange(total)
            for charclass, next_state in self.fsm.map[state].items():
                cnt = table[next_state][remaining - 1]
                if choice < cnt:
                    word.append(get_random_char_from_charclass(charclass, rng))
                    state = next_state
                    break
                choice -= cnt
        if state not in self.fsm.finals:
            raise ValueError("FSM ended in non-accepting state")
        return "".join(word)

    def generate_sequential_words(self, num_items=5, exclude=[]):
        """
        Generate a deterministic sequence of accepted words, starting with the shortest word(s).

        Parameters
        ----------
        num_items : int
        exclude : List[str]

        Returns
        -------
        List[str]
        """
        word_generator = self.pattern.strings()
        result = []
        for word in word_generator:
            if word not in exclude:
                result.append(word)
                if len(result) >= num_items:
                    break
        return result

    def generate_random_words(
        self,
        num_words: int,
        exclude: Iterable[str] = (),
        seed: Optional[int] = None,
    ) -> list[str]:
        rng = random.Random(seed)
        forbid = set(exclude)
        generated: list[str] = []

        min_len = self.minimum_word_length()
        max_len = self.maximum_word_length() or (min_len + num_words + 10)

        # Try lengths from min_len upward until we fill our quota
        for length in itertools.count(min_len):
            if length > max_len:
                break

            table = self._count_table(length)
            total_at_length = table[self.fsm.initial][length]
            if total_at_length == 0:
                # If no words of this length, move on.
                continue

            need = min(num_words - len(generated), total_at_length)
            seen_at_len: set[str] = set()
            attempts = 0
            max_attempts = max(need * 10, total_at_length * 2)

            while len(seen_at_len) < need and attempts < max_attempts:
                attempts += 1
                try:
                    w = self._sample_branch(length, rng)
                except ValueError:
                    break  # truly no more words
                if w in forbid or w in seen_at_len:
                    continue
                seen_at_len.add(w)

            generated.extend(seen_at_len)
            if len(generated) >= num_words:
                return generated[:num_words]

        if len(generated) < num_words:
            needed = num_words - len(generated)
            more = self.generate_random_words(
                num_words=needed,
                exclude=list(exclude) + list(generated),
                seed=seed,
            )
            generated.extend(more)

        return generated[:num_words]

    def generate_random_word(self, length, rng):
        """
        Generate one random accepted word of length `k`.

        Parameters
        ----------
        k : int
        seed : Optional[int]

        Returns
        -------
        str
        """
        count = self._count_table(length)

        if count[self.fsm.initial][length] == 0:
            raise ValueError(f"No words of length {length} accepted by FSM")

        state = self.fsm.initial
        word = []

        for remaining in range(length, 0, -1):
            transitions = list(self.fsm.map.get(state, {}).items())
            total = count[state][remaining]
            choice = rng.randint(0, total - 1)

            for charclass, next_state in transitions:
                next_count = count[next_state][remaining - 1]
                if choice < next_count:
                    symbol = get_random_char_from_charclass(charclass, rng)
                    word.append(symbol)
                    state = next_state
                    break
                choice -= next_count
            else:
                raise RuntimeError("FSM traversal logic failed.")

        if state not in self.fsm.finals:
            raise ValueError("Ended on non-accepting state. FSM is likely malformed.")

        return "".join(word)

    def intersect(self, other):
        """
        Compute the intersection of this regex with another.

        Returns
        -------
        RegexNode
        """
        intersect_pattern = self.pattern.intersection(other.pattern)
        return RegexNode.from_pattern(intersect_pattern)

    def union(self, other):
        """
        Compute the union of this regex with another.

        Returns
        -------
        RegexNode
        """
        union_pattern = self.pattern.union(other.pattern)
        return RegexNode.from_pattern(union_pattern)

    def difference(self, other):
        """
        Compute the difference (self - other) of two regexes.

        Returns
        -------
        RegexNode
        """
        difference_pattern = self.pattern.difference(other.pattern)
        return RegexNode.from_pattern(difference_pattern)

    def reduce(self):
        """
        Return a new RegexNode with a reduced (simplified) pattern.

        Returns
        -------
        RegexNode
        """
        return RegexNode.from_pattern(self.pattern.reduce())
