The following Python function is intended to return the first non-repeating character in a string, or `None` if every character repeats.

```python
def first_unique_char(s):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    for i in range(len(s)):
        if counts[i] == 1:
            return s[i]
    return ""
