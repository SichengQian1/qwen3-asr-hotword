from __future__ import annotations

try:
    from rapidfuzz.distance import Levenshtein as _RapidFuzzLevenshtein
except ImportError:  # pragma: no cover - exercised only in minimal local environments
    _RapidFuzzLevenshtein = None


def sequence_editops(
    reference: tuple[int, ...],
    hypothesis: tuple[int, ...],
) -> list[tuple[str, int, int]]:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distances = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distances[row][0] = row
    for column in range(columns):
        distances[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = reference[row - 1] != hypothesis[column - 1]
            distances[row][column] = min(
                distances[row - 1][column - 1] + substitution_cost,
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
            )

    operations: list[tuple[str, int, int]] = []
    row = len(reference)
    column = len(hypothesis)
    while row or column:
        if (
            row
            and column
            and reference[row - 1] == hypothesis[column - 1]
            and distances[row][column] == distances[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
        elif row and column and distances[row][column] == distances[row - 1][column - 1] + 1:
            operations.append(("replace", row - 1, column - 1))
            row -= 1
            column -= 1
        elif row and distances[row][column] == distances[row - 1][column] + 1:
            operations.append(("delete", row - 1, column))
            row -= 1
        else:
            operations.append(("insert", row, column - 1))
            column -= 1
    operations.reverse()
    return operations


def sequence_edit_distance(
    reference: tuple[int, ...],
    hypothesis: tuple[int, ...],
) -> int:
    if _RapidFuzzLevenshtein is not None:
        return int(_RapidFuzzLevenshtein.distance(reference, hypothesis))
    return len(sequence_editops(reference, hypothesis))


def sequence_edit_distance_backend() -> str:
    return "rapidfuzz" if _RapidFuzzLevenshtein is not None else "python_dynamic_programming"
