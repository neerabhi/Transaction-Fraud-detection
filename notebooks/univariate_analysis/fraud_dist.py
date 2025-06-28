import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from pathlib import Path  # Better path handling

def plot_fraud_distribution(df, output_path='reports/figures/fraud_distribution.png'):
    """
    Plots and saves fraud distribution plot with percentage annotations
    
    Args:
        df: Pandas DataFrame containing 'is_fraud' column
        output_path: Relative or absolute path for saving the figure
    """
    try:
        print("→ Starting fraud distribution visualization...")
        
        # ===== 1. Data Validation =====
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if 'is_fraud' not in df.columns:
            raise ValueError("DataFrame must contain 'is_fraud' column")
            
        print(f"✔ Data validated. Fraud counts:\n{df['is_fraud'].value_counts()}")

        # ===== 2. Plot Setup =====
        plt.figure(figsize=(8, 4))
        sns.set(style="whitegrid")
        ax = sns.countplot(y='is_fraud', data=df, color='red')
        
        # ===== 3. Add Percentage Annotations =====
        total = len(df)
        for p in ax.patches:
            percentage = f"{100 * p.get_width()/total:.1f}%"
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height()/2
            ax.annotate(percentage, (x, y), va='center')

        ax.set_xlabel('Count')
        ax.set_ylabel('Fraud Status')
        plt.title("Fraud Distribution in Dataset")

        # ===== 4. Path Handling =====
        output_path = Path(output_path).resolve()  # Convert to absolute path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"✔ Output directory ready: {output_path.parent}")

        # ===== 5. Save Figure =====
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='w',
            transparent=False
        )
        print(f"✅ Successfully saved visualization to:\n{output_path}")

    except Exception as e:
        print(f"❌ Error in plot_fraud_distribution: {str(e)}")
        raise  # Re-raise the exception for debugging
    finally:
        plt.close()

# ===== Example Usage =====
if __name__ == "__main__":
    try:
        # Load data (using path that works cross-platform)
        data_path = Path('data/interim/featured_data.csv')
        print(f"Loading data from: {data_path}")
        
        df = pd.read_csv(data_path)
        print("✔ Data loaded successfully")
        
        # Generate and save plot
        plot_fraud_distribution(df)
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        # Uncomment for detailed error in VS Code:
        # import traceback
        # traceback.print_exc()