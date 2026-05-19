import pandas as pd


def print_class_balance(df: pd.DataFrame, labels_col: str = "accepted_labels") -> None:
    """
    Imprime el balance de clases post-cleaning.

    Args:
        df:         DataFrame con la columna de labels (sets de strings).
        labels_col: Nombre de la columna que contiene los sets de labels.
    """
    label_counts: dict[str, int] = {}
    for label_set in df[labels_col]:
        for label in label_set:
            label_counts[label] = label_counts.get(label, 0) + 1

    total = len(df)
    print("=== Balance de clases (post-cleaning) ===")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label.ljust(25)} {count:>6}  ({100 * count / total:.1f}%)")

    print(f"\nTotal contratos  : {total}")
    avg_labels = sum(len(ls) for ls in df[labels_col]) / total
    print(f"Labels / contrato: {avg_labels:.2f} (promedio)")
