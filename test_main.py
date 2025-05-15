import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from model.model_factory import ModelFactory
from data_loader.data_loader import create_dataset_from_config
from trainer import create_trainer
from utils import load_json


# Configurable root directory for all outputs from this script
OUTPUT_ROOT_DIR = "paper"  # ⬅️ CHANGE THIS TO SET THE BASE OUTPUT LOCATION

FOLDERS = [
    # ⬅️ PUT YOUR MODEL CHECKPOINTS HERE
]

model_names = [
    # ⬅️ PUT YOUR MODEL NAMES HERE
        ]

# BASE_DIR = "paper/model_results" # This will be replaced by PLOTS_DIR
NAME = FOLDERS[-1]

# Define specific output directories based on OUTPUT_ROOT_DIR and the experiment NAME
PLOTS_DIR = os.path.join(OUTPUT_ROOT_DIR, NAME, "plots")
SAMPLES_DIR = os.path.join(OUTPUT_ROOT_DIR, NAME, "samples")

def compute_confusion_matrix(errors, shape, scale=1, row_norm=True):
    matrix = torch.zeros(*shape)
    for actual, predicted in errors:
        matrix[actual // scale, predicted // scale] += 1
    if row_norm:
        matrix = matrix / matrix.sum(dim=1, keepdim=True) * 100
    else:
        matrix = matrix / matrix.sum() * 100
    return matrix

def plot_matrix(matrix, filename, annotate=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    if annotate:
        for i in range(matrix.size(0)):
            for j in range(matrix.size(1)):
                plt.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center')
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrices(errors_dict, prefix="val", row=True):
    specs = {
        "cmd": (5, 1, True),
        "param_0": (200, 5, False),
        "param_1": (200, 5, False),
        "param_2": (20, 50, True),
        "param_3": (5, 200, True),
        "param_4": (2, 500, True),
        "param_5": (200, 5, False),
    }
    for key, (dim, scale, annotate) in specs.items():
        matrix = compute_confusion_matrix(errors_dict[key], (dim, dim), scale=scale, row_norm=row)
        # Construct full path for saving
        save_filename = os.path.join(PLOTS_DIR, f"{NAME}_{prefix}_{key}_confusion_matrix.png")
        plot_matrix(matrix, save_filename, annotate=annotate)

def plot_sequence_analysis(data, names, mode="val"):
    seq_lengths = data["Sequence Lengths"]
    first_mis = data["First Mistakes"]
    mistakes = data["Number of Mistakes"]

    plt.figure(figsize=(5, 5))
    plt.scatter([x[1] for x in seq_lengths], [x[0] for x in seq_lengths], alpha=0.1)
    max_len = max([x[1] for x in seq_lengths])
    plt.plot([0, max_len], [0, max_len], color='red')
    plt.ylim(0, max_len + 1)
    plt.xlabel("Actual Sequence Length")
    plt.ylabel("Predicted Sequence Length")
    plt.savefig(os.path.join(PLOTS_DIR, f"{NAME}_{mode}_seq_length_scatter.png"))

    # Calculate and print the number of perfect sequences
    perfect_sequences = sum(1 for x in seq_lengths if x[0] == x[1])
    print(f"Number of perfect sequences ({mode}): {perfect_sequences}")

    prob_dict = {k: len(v) for k, v in first_mis.items()}
    plt.figure(figsize=(7, 5))
    plt.bar(names, [prob_dict.get(k, 0) for k in prob_dict.keys()])
    plt.xticks(rotation=30)
    plt.xlabel("Commands and Parameters")
    plt.ylabel("Frequency of Mistake")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{NAME}_{mode}_prob_histogram.png"))

    mistakes_per_seq = [sum(mistakes[i]) / seq_lengths[i][1] for i in range(len(seq_lengths))]
    plt.figure(figsize=(8, 5))
    plt.hist(mistakes_per_seq, bins=np.linspace(0, 1, 101), edgecolor='black', align='left')
    plt.xlabel("Number of Mistakes per Sequence")
    plt.ylabel("Number of Sequences")
    plt.title("Histogram of Mistakes per Sequence")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{NAME}_{mode}_mistakes_histogram.png"))

    seq_lengths_actual = [x[1] for x in seq_lengths]
    mistakes_per_seq = [sum(mistakes[i]) for i in range(len(seq_lengths))]

    plt.figure(figsize=(8, 5))
    plt.scatter(seq_lengths_actual, mistakes_per_seq, alpha=0.5)
    plt.xlabel("Sequence Length")
    plt.ylabel("Number of Mistakes")
    plt.title("Mistakes as a Function of Sequence Length")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{NAME}_{mode}_mistakes_vs_seq_length.png"))

def plot_accuracy_vs_tolerance(data, max_tol, mode="val"):
    """
    Plots prediction accuracy for each feature as a function of the tolerance.

    Args:
        data: list of length T where data[t]["Memory"] has actual/predicted pairs for tolerance t
        names: list of names for plotting (not used in this plot, but may be for extensions)
        mode: "train", "val", or "test"
    """
    features = ["param_0", "param_1", "param_5"]
    accuracies = {f: [] for f in features}
    tolerances = list(range(max_tol))

    for t in tolerances:
        memory = data[-1]["Memory"]
        for f in features:
            gt_pd_pairs = memory[f]
            correct = sum(1 for gt, pd in gt_pd_pairs if abs(gt - pd) <= t)
            total = len(gt_pd_pairs)
            acc = (correct / total * 100) if total > 0 else 0.0
            accuracies[f].append(acc)

    # Plot
    plt.figure(figsize=(10, 6))
    for f in features:
        plt.plot(tolerances, accuracies[f], label=f)
    plt.xlabel("Tolerance")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Feature Accuracy vs Tolerance ({mode})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{NAME}_{mode}_accuracy_vs_tolerance.png"))
    plt.close()

def plot_perfect_sequence_percentage(data, mode="val"):
    """
    Plots the percentage of perfect sequences as a function of the percentage of the sequence that was given.

    Args:
        data: list of dicts per tolerance value, each containing "Number of Mistakes" and "Sequence Lengths"
        mode: "train", "val", or "test"
    """
    data_dict = data[-1]  # Use last tolerance level for exact errors
    num_mistakes = data_dict["Number of Mistakes"]
    seq_lengths = data_dict["Sequence Lengths"]
    max_percent = 100
    percentages = list(range(max_percent + 1))
    perfect_fractions = []

    for p in percentages:
        frac = p / 100.0
        perfect_count = 0
        total = len(seq_lengths)
        for i in range(total):
            gt_len = seq_lengths[i][1]
            start_idx = int(frac * gt_len)
            if sum(num_mistakes[i][start_idx:]) == 0:
                perfect_count += 1
        perfect_fractions.append(perfect_count / total * 100)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(percentages, perfect_fractions, marker='o')
    plt.xlabel("Percentage of Sequence Given (%)")
    plt.ylabel("Perfect Sequences (%)")
    plt.title(f"Perfect Sequence Rate vs Percentage Given ({mode})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{NAME}_{mode}_perfect_sequence_vs_given.png"))
    plt.close()


def plot_sequence_length_scatter(seq_lengths, output_path_basename, output_dir):
    """Plot scatter plot of actual vs predicted sequence lengths."""
    plt.figure(figsize=(5, 5))
    plt.scatter([x[1] for x in seq_lengths], [x[0] for x in seq_lengths], alpha=0.1)
    plt.plot([0, max([x[1] for x in seq_lengths])], [0, max([x[1] for x in seq_lengths])], color='red')
    plt.ylim(0, max([x[1] for x in seq_lengths]) + 1)
    plt.xlabel("Actual Sequence Length")
    plt.ylabel("Predicted Sequence Length")
    plt.savefig(os.path.join(output_dir, output_path_basename + ".png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="ml_dataset/multi_extrude_resized")
    parser.add_argument("--config_path", type=str, default="ml_dataset/multi_extrude_resized/dataset_split.json")
    parser.add_argument("--multiview_dir", type=str, default="multi_view_images")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--view_ids", type=list, default=["05", "09", "20"])
    parser.add_argument("--image_dir", type=str, default="ml_dataset/multi_extrude/images")
    parser.add_argument("--model_config", type=str, default="model_configs/transformer_experiments.json")
    parser.add_argument("--model_name", type=str, default=model_names[-1])
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/"+FOLDERS[-1]+"/best_model.pt")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--enable_parallel", type=bool, default=False)
    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    device = "cuda:0"
    model_params = load_json(args.model_config)
    model_config = model_params[args.model_name]
    model_config["device"] = device
    num_views = model_config.get("num_views", 0)
    gencad = model_config.get("use_pretrained_cad_model", False)

    if num_views == 0:
        args.view_ids = []

    frame_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    loader_args = {
        'dataset_path': args.dataset_path,
        'config': args.config_path,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'multiview_dir': args.multiview_dir,
        'view_ids': args.view_ids,
        'frame_transform': frame_transform,
        'image_transform': None if gencad else transforms.Normalize([0.5], [0.5]),
        'gencad': gencad,
        'sequence_retriever': "base",
        'image_dir': args.image_dir
    }

    train_loader, test_loader, val_loader = create_dataset_from_config(**loader_args)


    state_dict = torch.load(args.checkpoint_path, map_location=device)['model_state_dict']
    model_factory = ModelFactory()
    model, model_type = model_factory.create_model(
        model_config.get("model_name", "autoregressive"), model_config, device, state_dict)

    training_config = {
        'batch_size': args.batch_size,
        'lr': 1e-5,
        'num_workers': args.num_workers,
        'epochs': args.epochs,
        'enable_parallel': args.enable_parallel,
        'sequential': True,
        'early_stopping_enabled': True,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'early_stopping_metric': 'loss',
        'early_stopping_mode': 'min',
        'use_mse': True
    }

    trainer = create_trainer(train_loader, val_loader, test_loader, model, training_config, device, model_type)
    names = ["Move to", "Press key", "Scroll", "Type", "Click", "x", "y", "Key Pressed", "Times Key Pressed", "Scroll Amount", "Type Amount"]

    print("Test len", len(trainer.test_loader))
    trainer.sample(model, mode="test", folder=SAMPLES_DIR, ablation=False, n=len(trainer.test_loader))
    
    # data = trainer.find_first_mistake(model, mode="train", ablation=False, tol=11)

    # plot_sequence_analysis(data[-1], names, mode="train")

    # plot_confusion_matrices(data[-1]["Memory"], prefix="train")

    # plot_accuracy_vs_tolerance(data, max_tol=20, mode="train")

    # plot_perfect_sequence_percentage(data, mode="train")

    # data = trainer.find_first_mistake(model, mode="val", ablation=False, tol=10)

    # plot_sequence_analysis(data[-1], names, mode="val")

    # plot_confusion_matrices(data[-1]["Memory"], prefix="val")

    # plot_accuracy_vs_tolerance(data, max_tol=20, mode="val")

    # plot_perfect_sequence_percentage(data, mode="val")

    data = trainer.find_first_mistake(model, mode="test", ablation=False, tol=10)

    plot_sequence_analysis(data[-1], names, mode="test")

    plot_confusion_matrices(data[-1]["Memory"], prefix="test")

    plot_accuracy_vs_tolerance(data, max_tol=20, mode="test")

    plot_perfect_sequence_percentage(data, mode="test")

    # print("\nEvaluating on Training Set:")
    # trainer.print_metrics(trainer.evaluate(model, mode="train"))

    print("\nEvaluating on Validation Set:")
    trainer.print_metrics(trainer.evaluate(model, mode="val"))

    print("\nEvaluating on Test Set:")
    trainer.print_metrics(trainer.evaluate(model, mode="test"))

    # visualize_mistakes(model, trainer, names, mode="test")

if __name__ == "__main__":
    main()