import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from jsonargparse import CLI
from typing import List, Optional
import os

def validate_kernel_name(user_name: str, file_path: Path) -> str:
    df = pd.read_csv(file_path)
    kernel_names = df['Fp8 Kernel'].str.lower()
    user_lower = user_name.lower()
    exact_matches = kernel_names[kernel_names == user_lower]
    if not exact_matches.empty:
        return df.loc[exact_matches.index[0], 'Fp8 Kernel']
    partial_matches = kernel_names[kernel_names.str.contains(user_lower, regex=False)]
    if not partial_matches.empty:
        matched_kernel = df.loc[partial_matches.index[0], 'Fp8 Kernel']
        print(f"No exact match found. Using closest match: {matched_kernel}")
        return matched_kernel
    valid_kernels = kernel_names.unique().tolist()
    raise ValueError(f"Invalid kernel name: {user_name}. Valid options are: {', '.join(valid_kernels)}")

def load_and_process_data(file_path: Path, kernel_name: str, k_choices: Optional[List[int]] = None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"Loaded CSV with shape: {df.shape}")
    print(f"Columns: {df.columns}")
    df_filtered = df[df["Fp8 Kernel"] == kernel_name]
    print(f"Filtered data shape for kernel {kernel_name}: {df_filtered.shape}")
    if k_choices:
        df_filtered = df_filtered[df_filtered['K'].isin(k_choices)]
    print(f"Data shape after filtering for specified K values: {df_filtered.shape}")
    print(f"Unique K values: {df_filtered['K'].unique()}")
    print(f"Unique M values: {df_filtered['M'].unique()}")
    print(f"Unique N values: {df_filtered['N'].unique()}")
    return df_filtered

def plot_heatmap(df: pd.DataFrame, kernel_name: str, k: int, output_path: Path, max_cells_for_annotation: int):
    plt.figure(figsize=(12, 10))
    plt.title(f"TFlops Heatmap for {kernel_name}, K={k}", fontsize=16)
    pivot = df[df['K'] == k].pivot(index="N", columns="M", values="FP8 TFLOPS")
    annotate = pivot.size <= max_cells_for_annotation
    sns.heatmap(
        pivot,
        cmap="viridis",
        cbar_kws={"label": "TFlops"},
        annot=annotate,
        fmt=".2f" if annotate else "",
    )
    plt.xlabel("M")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def main(
    file_path: Path,
    kernel_name: str,
    k_choices: Optional[List[int]] = None,
    max_cells: int = 400,
    output_folder: Optional[Path] = None
):
    try:
        validated_kernel_name = validate_kernel_name(kernel_name, file_path)
        print(f"Validated kernel name: {validated_kernel_name}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    df = load_and_process_data(file_path, validated_kernel_name, k_choices)
    
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    else:
        output_folder = file_path.parent

    in_file_name_base = file_path.stem
    for k in df['K'].unique():
        output_path = output_folder / f"{in_file_name_base}_{validated_kernel_name.replace('.', '_')}_K{k}_tflops_heatmap.png"
        plot_heatmap(df, validated_kernel_name, k, output_path, max_cells)
        print(f"Heatmap for K={k} saved to {output_path}")

if __name__ == "__main__":
    CLI(main)
