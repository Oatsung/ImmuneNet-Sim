import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from app import TissueModel

def clean_temporary_files():
    """Clear pre-existing simulation artifacts before running experiments."""
    if os.path.exists("simulation_data.csv"):
        try:
            os.remove("simulation_data.csv")
        except OSError:
            pass

def run_headless_simulation(config, output_filename):
    """Executes the TissueModel without GUI and renames the auto-generated telemetry."""
    clean_temporary_files()
    
    print(f"🧬 Initializing In-Silico Trial: {output_filename}...")
    model = TissueModel(
        num_cancer=config["num_cancer"],
        num_tcell=config["num_tcell"],
        tcell_energy=config["tcell_energy"],
        base_proliferation=config["base_proliferation"],
        drug_type=config["drug_type"],
        seed=config.get("seed")
    )
    
    # Execute 100 simulation steps sequentially
    for step in range(100):
        model.step()
        
    if os.path.exists("simulation_data.csv"):
        # 🛠️ [FIX] ใช้ os.replace แทน os.rename เพื่อแก้ปัญหา FileExistsError: [WinError 183] บน Windows
        os.replace("simulation_data.csv", output_filename)
        print(f"💾 Telemetry logged and preserved as '{output_filename}'\n")
    else:
        raise FileNotFoundError("Error: Core engine failed to generate simulation_data.csv")

def generate_comparative_matrix_plot(file_a, file_b):
    """Generates a 2x2 publication-grade visualization comparing both trials."""
    print("📊 Compiling simulation matrix and plotting academic dashboard...")
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create a 2x2 Subplot matrix sharing X-axis across columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    
    # --- ROW 1, COL 1: Case A Cellular Dynamics ---
    axes[0, 0].plot(df_a['Step'], df_a['Normal Cancer (Red)'], color='#FF4B4B', linewidth=2.5, label='Normal Cancer')
    axes[0, 0].plot(df_a['Step'], df_a['Mutated Cancer (Orange)'], color='#FF9900', linewidth=2.5, linestyle='--', label='Mutated Resistant')
    axes[0, 0].plot(df_a['Step'], df_a['Active T-Cells'], color='#1F77B4', linewidth=2.5, label='Active T/CAR-T')
    axes[0, 0].set_title("Case A: Tumor Immune Escape (Cell Counts)", fontsize=13, fontweight='bold', color='#D32F2F')
    axes[0, 0].set_ylabel("Cellular Count", fontsize=11)
    axes[0, 0].legend(loc='upper left')
    
    # --- ROW 1, COL 2: Case B Cellular Dynamics ---
    axes[0, 1].plot(df_b['Step'], df_b['Normal Cancer (Red)'], color='#FF4B4B', linewidth=2.5, label='Normal Cancer')
    axes[0, 1].plot(df_b['Step'], df_b['Mutated Cancer (Orange)'], color='#FF9900', linewidth=2.5, linestyle='--', label='Mutated Resistant')
    axes[0, 1].plot(df_b['Step'], df_b['Active T-Cells'], color='#1F77B4', linewidth=2.5, label='Active T/CAR-T')
    axes[0, 1].set_title("Case B: Successful Eradication (Cell Counts)", fontsize=13, fontweight='bold', color='#388E3C')
    axes[0, 1].legend(loc='upper right')

    # --- ROW 2, COL 1: Case A Chemical Signaling ---
    axes[1, 0].plot(df_a['Step'], df_a['Total IFN-g (Signal)'], color='#2CA02C', linewidth=2, label='IFN-γ (Attractant)')
    axes[1, 0].plot(df_a['Step'], df_a['Total TGF-b (Toxin)'], color='#9467BD', linewidth=2, label='TGF-β (Suppression)')
    axes[1, 0].set_title("Case A: Cytokine Milieu & Exhaustion", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Simulation Steps", fontsize=11)
    axes[1, 0].set_ylabel("Total Gradiant Concentration", fontsize=11)
    axes[1, 0].legend(loc='upper left')

    # --- ROW 2, COL 2: Case B Chemical Signaling ---
    axes[1, 1].plot(df_b['Step'], df_b['Total IFN-g (Signal)'], color='#2CA02C', linewidth=2, label='IFN-γ (Attractant)')
    axes[1, 1].plot(df_b['Step'], df_b['Total TGF-b (Toxin)'], color='#9467BD', linewidth=2, label='TGF-β (Suppression)')
    axes[1, 1].set_title("Case B: Sustained Immune Suppression Control", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Simulation Steps", fontsize=11)
    axes[1, 1].legend(loc='upper right')

    # Add Dynamic Indicators for Therapy Activation Points (Trigger Step)
    for idx, df in enumerate([df_a, df_b]):
        combo_starts = df[df['Combo Therapy Active (Indicator)'] > 0]['Step']
        if not combo_starts.empty:
            start_step = combo_starts.iloc[0]
            axes[0, idx].axvline(x=start_step, color='#2CA02C', linestyle=':', linewidth=2)
            axes[1, idx].axvline(x=start_step, color='#2CA02C', linestyle=':', linewidth=2)
            axes[0, idx].text(start_step + 1.5, axes[0, idx].get_ylim()[1]*0.80, 'AI Triggered', color='#2CA02C', fontweight='bold', fontsize=10)

    # 🛠️ [FIX] ขยับและจัดระเบียบพื้นที่เพื่อป้องกันหัวข้อซ้อนทับกัน (Layout Tuning)
    # 🛠️ [FIX] ขยับ Layout ใหม่: เว้นขอบล่าง 4% (0.04) กันข้อความขาด และเว้นขอบบน 5% (0.95) หลบหัวเรื่อง
    plt.suptitle("Systems Immunology Engine: Comparative Trial Matrix", fontsize=18, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    output_img = "simulation_comparison_matrix.png"
    plt.savefig(output_img, dpi=300)
    print(f"🏆 Comparative Analysis Matrix saved successfully as '{output_img}'!")
    plt.show()

if __name__ == "__main__":
    # Case A: Parameter set mimicking high-aggression tumor escaping weak immune defense
    case_a_config = {
        "num_cancer": 25,
        "num_tcell": 8,
        "tcell_energy": 60,
        "base_proliferation": 0.06,
        "drug_type": "AI Director Adaptive",
        "seed": 0  # fixed seed -> reproducible. Over seeds 0-29: Case A eradicated 0/30, Case B 22/30
    }

    # Case B: Parameter set engineered to facilitate therapeutic success (High T-Cell Endurance)
    case_b_config = {
        "num_cancer": 25,
        "num_tcell": 12,
        "tcell_energy": 120,
        "base_proliferation": 0.04,
        "drug_type": "AI Director Adaptive",
        "seed": 0  # fixed seed -> reproducible. Over seeds 0-29: Case A eradicated 0/30, Case B 22/30
    }

    file_a = "simulation_data_case_a.csv"
    file_b = "simulation_data_case_b.csv"

    # Execute experimental batch pipeline
    run_headless_simulation(case_a_config, file_a)
    run_headless_simulation(case_b_config, file_b)
    
    # Process analytics and save plots
    generate_comparative_matrix_plot(file_a, file_b)
