import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_academic_plot(csv_path="simulation_data.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run app.py first.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Set academic plotting style
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Cell Population Dynamics
    axes[0].plot(df['Step'], df['Cancer_Standard'], color='red', label='Standard Cancer Cells', linewidth=2)
    axes[0].plot(df['Step'], df['Cancer_Mutated'], color='darkorange', label='Mutated/Resistant Cells', linewidth=2, linestyle='--')
    axes[0].plot(df['Step'], df['TCell_Standard'], color='blue', label='Standard T-Cells', linewidth=2)
    axes[0].plot(df['Step'], df['CART_Cell'], color='cyan', label='CAR-T Cells', linewidth=2, linestyle='-.')
    
    axes[0].set_ylabel('Cell Population Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Tumor Microenvironment (TME) Population Dynamics', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')

    # Subplot 2: Chemical Concentrations (IFN-γ vs TGF-β)
    axes[1].plot(df['Step'], df['Total_IFN_Gamma'], color='green', label='IFN-γ (Chemotaxis Signal)', linewidth=2)
    axes[1].plot(df['Step'], df['Total_TGF_Beta'], color='purple', label='TGF-β (Immunosuppression)', linewidth=2, linestyle='--')
    
    axes[1].set_xlabel('Simulation Step (Time)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Total Chemical Concentration', fontsize=12, fontweight='bold')
    axes[1].set_title('Cytokine Milieu & Chemokine Gradient', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("tme_simulation_results.png", dpi=300)
    print("Plot saved successfully as 'tme_simulation_results.png'")
    plt.show()

if __name__ == "__main__":
    generate_academic_plot()