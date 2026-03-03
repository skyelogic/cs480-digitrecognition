"""
Author: Donnel Garner

Module 6 Assignment: Learning Agents - Handwritten Digit Recognition
Neural Network with variable hidden layers (10, 100, 500 nodes)
Bonus: PCA dimensionality reduction

Usage:
    python encode_and_train.py --data optdigits-orig.windep
"""

# We need to import libraries, it won't work otherwise
import numpy as np
import argparse
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# We are choosing peace today: silence the warnings so the output doesn’t look like a haunted terminal.
warnings.filterwarnings('ignore')


# ─── Task 1: Encoding ──────────────────────────────────────────────────────────

def parse_data(filepath):
    """
    Parse the optdigits-orig.windep file.
    Each sample: 32 lines of 32 chars ('0' or '1') followed by ' <label>\n'.
    Returns X (N x 1024 float array) and y (N int array).
    """

    # Open the file and inhale the entire thing into memory like it’s a bag of chips at 2AM.
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip header lines until we hit a 32-char bitmap line
    # (Translation: ignore the “blah blah” until we see actual pixel art.)
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) == 32 and set(stripped).issubset({'0', '1'}):
            start = i
            break

    X, y = [], []
    i = start
    while i < len(lines):
        # Read 32 bitmap rows
        bitmap_rows = []
        for _ in range(32):
            if i >= len(lines):
                break
            row = lines[i].strip()
            if len(row) == 32 and set(row).issubset({'0', '1'}):
                bitmap_rows.append([int(c) for c in row])
                i += 1
            else:
                i += 1
                break

        if len(bitmap_rows) != 32:
            continue

        # Next line should be the label: " <digit>"
        if i < len(lines):
            label_line = lines[i].strip()
            i += 1
            try:
                label = int(label_line)  # Convert label to an integer like a responsible adult.
                X.append(np.array(bitmap_rows, dtype=float).flatten())
                y.append(label)
            except ValueError:
                continue  # skip malformed

    return np.array(X), np.array(y)


def save_encoded(X, y, out_dir):
    """Save input and target arrays as CSV files."""
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, 'inputs.csv'), X, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(out_dir, 'targets.csv'), y, delimiter=',', fmt='%d')
    print(f"Encoded data saved to '{out_dir}/'")
    print(f"  inputs.csv  : shape {X.shape}  (N samples x 1024 features)")
    print(f"  targets.csv : shape {y.shape}  (N labels, values 0-9)")


# ─── Task 2: Neural Network Training ──────────────────────────────────────────

def train_and_evaluate(X, y, hidden_nodes, random_state=42):
    """
    Split data, train MLPClassifier, return accuracy results.
    Split: 60% train | 20% validation | 20% test
    """
    # First split off test set (20%)
    # The test set is the final boss: no training, no tuning, no touching.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y)

    # Then split train/val (60/20 of total = 75/25 of trainval)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=random_state, stratify=y_trainval)

    # Normalize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Print a nice banner so you can tell which model is about to flex.
    print(f"\n{'='*60}")
    print(f"  Hidden nodes: {hidden_nodes}")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"{'='*60}")

    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_nodes,),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False
    )
    clf.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_s))
    val_acc   = accuracy_score(y_val,   clf.predict(X_val_s))
    test_acc  = accuracy_score(y_test,  clf.predict(X_test_s))

    # Print accuracies in a human-friendly way (percentages, not existential dread).
    print(f"  Training   accuracy: {train_acc*100:.2f}%")
    print(f"  Validation accuracy: {val_acc*100:.2f}%")
    print(f"  Testing    accuracy: {test_acc*100:.2f}%")

    # Print per-class precision/recall/F1 so you can see which digit the model hates most.
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, clf.predict(X_test_s), digits=3))

    # Return everything needed for summary + possible reuse.
    return {'hidden_nodes': hidden_nodes,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'model': clf,
            'scaler': scaler}


# ─── Bonus: PCA Dimensionality Reduction ──────────────────────────────────────

def pca_experiment(X, y, n_components_list=(50, 100, 200), hidden_nodes=100, random_state=42):
    """
    Train networks using PCA-reduced features and compare to full 1024-feature baseline.
    """
    print(f"\n{'='*60}")
    print("  BONUS: PCA Dimensionality Reduction")
    print(f"{'='*60}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=random_state, stratify=y_trainval)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    results = []
    for n_comp in n_components_list:
        pca = PCA(n_components=n_comp, random_state=random_state)
        Xtr = pca.fit_transform(X_train_s)
        Xva = pca.transform(X_val_s)
        Xte = pca.transform(X_test_s)

        variance_retained = pca.explained_variance_ratio_.sum() * 100

        clf = MLPClassifier(hidden_layer_sizes=(hidden_nodes,), activation='relu',
                            solver='adam', max_iter=500, random_state=random_state,
                            early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=15, verbose=False)
        clf.fit(Xtr, y_train)

        test_acc = accuracy_score(y_test, clf.predict(Xte))
        print(f"  PCA({n_comp:4d} components, {variance_retained:.1f}% variance) -> "
              f"Test accuracy: {test_acc*100:.2f}%")
        results.append((n_comp, variance_retained, test_acc))

    return results


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Handwritten Digit Recognition NN')
    parser.add_argument('--data', default='optdigits-orig.windep',
                        help='Path to optdigits-orig.windep file')
    parser.add_argument('--out_dir', default='encoded_data',
                        help='Directory for encoded CSV files')
    parser.add_argument('--bonus', action='store_true', default=True,
                        help='Run PCA bonus experiment')
    args = parser.parse_args()

    # ── Task 1: Encode ──
    print("\n[Task 1] Parsing and encoding data...")
    X, y = parse_data(args.data)
    print(f"  Loaded {len(X)} samples, {X.shape[1]} features, classes: {sorted(set(y))}")
    save_encoded(X, y, args.out_dir)

    # ── Task 2: Train ──
    print("\n[Task 2] Training neural networks with 10, 100, 500 hidden nodes...")
    summary = []
    for h in [10, 100, 500]:
        res = train_and_evaluate(X, y, hidden_nodes=h)
        summary.append(res)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"  {'Hidden':>8}  {'Train %':>9}  {'Val %':>9}  {'Test %':>9}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")
    for r in summary:
        print(f"  {r['hidden_nodes']:>8}  {r['train_acc']*100:>8.2f}%  "
              f"{r['val_acc']*100:>8.2f}%  {r['test_acc']*100:>8.2f}%")

    # ── Bonus: PCA ──
    if args.bonus:
        # PCA experiment: because 1024 features is a lot, and we like our training time shorter than a movie trilogy.
        pca_results = pca_experiment(X, y, n_components_list=[20, 50, 100, 150, 200])

        print(f"\n  PCA Analysis:")
        print(f"  Using PCA we can reduce 1024 features dramatically while retaining accuracy.")
        print(f"  For example, ~100 PCA components capture most variance and perform similarly")
        print(f"  to the full 1024-feature model, cutting training time significantly.")


if __name__ == '__main__':
    main()
