"""Coding tasks shown in the UI. Tasks with `tests` are auto-graded after generation."""

CODE_ONLY = (
    "You are an expert Python programmer. Reply with one complete ```python code block "
    "that implements the requested function or class. Do not include example usage or tests."
)
GENERAL = "You are a helpful, precise coding assistant."

PRESETS = [
    {
        "id": "two_sum",
        "title": "Two Sum",
        "system": CODE_ONLY,
        "prompt": "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that returns "
                  "the indices of the two numbers that add up to `target`, in increasing order. "
                  "Assume exactly one solution exists. Aim for O(n) time.",
        "tests": [
            "assert two_sum([2, 7, 11, 15], 9) == [0, 1]",
            "assert two_sum([3, 2, 4], 6) == [1, 2]",
            "assert two_sum([3, 3], 6) == [0, 1]",
            "assert two_sum([-1, -2, -3, -4, -5], -8) == [2, 4]",
        ],
    },
    {
        "id": "valid_parens",
        "title": "Valid parentheses",
        "system": CODE_ONLY,
        "prompt": "Write a Python function `is_valid(s: str) -> bool` that returns True if the brackets "
                  "'()', '[]' and '{}' in `s` are balanced and correctly nested.",
        "tests": [
            "assert is_valid('()') is True",
            "assert is_valid('()[]{}') is True",
            "assert is_valid('(]') is False",
            "assert is_valid('([)]') is False",
            "assert is_valid('{[]}') is True",
            "assert is_valid('') is True",
            "assert is_valid('((') is False",
        ],
    },
    {
        "id": "roman",
        "title": "Roman numeral to int",
        "system": CODE_ONLY,
        "prompt": "Write a Python function `roman_to_int(s: str) -> int` that converts a Roman numeral "
                  "(1 to 3999) to an integer.",
        "tests": [
            "assert roman_to_int('III') == 3",
            "assert roman_to_int('IV') == 4",
            "assert roman_to_int('LVIII') == 58",
            "assert roman_to_int('MCMXCIV') == 1994",
            "assert roman_to_int('MMMCMXCIX') == 3999",
        ],
    },
    {
        "id": "merge_intervals",
        "title": "Merge intervals",
        "system": CODE_ONLY,
        "prompt": "Write a Python function `merge(intervals: list[list[int]]) -> list[list[int]]` that merges "
                  "all overlapping (or touching) intervals and returns them sorted by start. "
                  "The input may be unsorted and may be empty.",
        "tests": [
            "assert merge([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]",
            "assert merge([[1, 4], [4, 5]]) == [[1, 5]]",
            "assert merge([]) == []",
            "assert merge([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]",
            "assert merge([[1, 10], [2, 3]]) == [[1, 10]]",
        ],
    },
    {
        "id": "lru",
        "title": "LRU cache class",
        "system": CODE_ONLY,
        "prompt": "Implement a Python class `LRUCache` with `__init__(self, capacity: int)`, "
                  "`get(self, key: int) -> int` (return -1 if the key is missing) and "
                  "`put(self, key: int, value: int) -> None`. When capacity is exceeded, evict the "
                  "least recently used key. Both operations must be O(1).",
        "tests": [
            "c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1",
            "c = LRUCache(2); c.put(1, 1); c.put(2, 2); c.get(1); c.put(3, 3); assert c.get(2) == -1",
            "c = LRUCache(2); c.put(1, 1); c.put(2, 2); c.put(1, 10); c.put(3, 3); assert c.get(1) == 10 and c.get(2) == -1",
            "c = LRUCache(1); c.put(1, 1); c.put(2, 2); assert c.get(1) == -1 and c.get(2) == 2",
            "c = LRUCache(2); assert c.get(42) == -1",
        ],
    },
    {
        "id": "top_k_words",
        "title": "Top-k frequent words",
        "system": CODE_ONLY,
        "prompt": "Write a Python function `top_k_words(text: str, k: int) -> list[str]` that returns the `k` "
                  "most frequent words in `text`. Words are case-insensitive and consist of letters only "
                  "(ignore punctuation). Return them lowercase, ordered by frequency descending, "
                  "breaking ties alphabetically.",
        "tests": [
            "assert top_k_words('the cat and the hat', 1) == ['the']",
            "assert top_k_words('b a b a c', 2) == ['a', 'b']",
            "assert top_k_words('Hello, hello! World.', 2) == ['hello', 'world']",
            "assert top_k_words('x y z', 3) == ['x', 'y', 'z']",
        ],
    },
    {
        "id": "flatten",
        "title": "Flatten nested list",
        "system": CODE_ONLY,
        "prompt": "Write a Python function `flatten(items: list) -> list` that flattens an arbitrarily "
                  "nested list of integers into a single flat list, preserving order.",
        "tests": [
            "assert flatten([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]",
            "assert flatten([]) == []",
            "assert flatten([[[]]]) == []",
            "assert flatten([[1, 2], [3], 4]) == [1, 2, 3, 4]",
        ],
    },
    {
        "id": "primes",
        "title": "Sieve of Eratosthenes",
        "system": CODE_ONLY,
        "prompt": "Write a Python function `primes_up_to(n: int) -> list[int]` that returns all primes <= n "
                  "using the Sieve of Eratosthenes.",
        "tests": [
            "assert primes_up_to(10) == [2, 3, 5, 7]",
            "assert primes_up_to(1) == []",
            "assert primes_up_to(2) == [2]",
            "assert len(primes_up_to(1000)) == 168",
        ],
    },
    {
        "id": "find_bug",
        "title": "Find the bug (free-form)",
        "system": GENERAL,
        "prompt": "This function should return the median of a list of numbers, but it is wrong for some "
                  "inputs. Explain the bug briefly and give a fixed version.\n\n"
                  "```python\ndef median(xs):\n    xs.sort()\n    n = len(xs)\n    return xs[n // 2]\n```",
        "tests": [],
    },
    {
        "id": "to_js",
        "title": "Translate Python to JavaScript (free-form)",
        "system": GENERAL,
        "prompt": "Translate this Python function to idiomatic modern JavaScript:\n\n"
                  "```python\ndef group_by_len(words):\n    out = {}\n    for w in words:\n"
                  "        out.setdefault(len(w), []).append(w)\n    return out\n```",
        "tests": [],
    },
]
