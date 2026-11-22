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
    # Column names match TissueModel's DataCollector (app.py). CAR-T is a global
    # model flag, not a separate headcount, so there is no CAR-T series to plot.
    axes[0].plot(df['Step'], df['Normal Cancer (Red)'], color='red', label='Standard Cancer Cells', linewidth=2)
    axes[0].plot(df['Step'], df['Mutated Cancer (Orange)'], color='darkorange', label='Mutated/Resistant Cells', linewidth=2, linestyle='--')
    axes[0].plot(df['Step'], df['Active T-Cells'], color='blue', label='T-Cells (incl. CAR-T once triggered)', linewidth=2)

    axes[0].set_ylabel('Cell Population Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Tumor Microenvironment (TME) Population Dynamics', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')

    # Subplot 2: Chemical Concentrations (IFN-γ vs TGF-β)
    axes[1].plot(df['Step'], df['Total IFN-g (Signal)'], color='green', label='IFN-γ (Chemotaxis Signal)', linewidth=2)
    axes[1].plot(df['Step'], df['Total TGF-b (Toxin)'], color='purple', label='TGF-β (Immunosuppression)', linewidth=2, linestyle='--')

    axes[1].set_xlabel('Simulation Step (Time)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Total Chemical Concentration', fontsize=12, fontweight='bold')
    axes[1].set_title('Cytokine Milieu & Chemokine Gradient', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')

    # Mark the step where the AI Director triggered combo/CAR-T therapy, if any
    trigger = df[df['Combo Therapy Active (Indicator)'] > 0]['Step']
    if not trigger.empty:
        t0 = trigger.iloc[0]
        for ax in axes:
            ax.axvline(x=t0, color='#2CA02C', linestyle=':', linewidth=2)
        axes[0].text(t0 + 1, axes[0].get_ylim()[1] * 0.9, 'AI Triggered', color='#2CA02C', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig("tme_simulation_results.png", dpi=300)
    print("Plot saved successfully as 'tme_simulation_results.png'")
    plt.show()

if __name__ == "__main__":
    generate_academic_plot()