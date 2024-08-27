import math


def get_labels(N: int, n_cols: int):
    labels = []
    n_rows = math.ceil(N / n_cols)

    n_row_chars = int(math.ceil(math.log(n_rows, 26)))

    for i in range(N):
        row = i // n_cols
        col = i % n_cols
        row_label = chr(65 + (row % 26))
        row = row // 26
        while row > 0:
            row_label = chr(65 + (row % 26)) + row_label
            row = row // 26

        # pad with As
        if len(row_label) < n_row_chars:
            row_label = "A" * (n_row_chars - len(row_label)) + row_label

        labels.append(f"{row_label}{col + 1}")

    return labels
