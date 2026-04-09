Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals.

Requirements:
- Input: list of tuples like `[(1, 3), (2, 5), (8, 10)]`
- Output: merged list sorted by start
- Handle empty input
- Preserve single-point intervals like `(4, 4)`
- Treat touching intervals as overlapping, e.g. `(1, 3)` and `(3, 5)` should merge to `(1, 5)`
- Include a docstring
- Include exactly 3 small self-contained tests using `assert`
- Return only code
