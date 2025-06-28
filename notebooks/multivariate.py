import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_correlation_matrix(df, output_path='reports/figures/correlation_matrix.png'):
    """
    Plots a professional correlation matrix heatmap for numerical variables
    
    Args:
        df: Pandas DataFrame
        output_path: Path to save the visualization
    """
    try:
        print("→ Starting correlation matrix visualization...")
        
        # ===== 1. Data Preparation =====
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        num_attributes = df.select_dtypes(include=[np.number])
        if len(num_attributes.columns) < 2:
            raise ValueError("Need at least 2 numerical columns for correlation matrix")
            
        print(f"✔ Calculating correlations for {len(num_attributes.columns)} numerical features")

        # ===== 2. Compute Correlation Matrix =====
        corr = num_attributes.corr()
        
        # ===== 3. Create Mask for Upper Triangle =====
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # ===== 4. Plot Setup =====
        plt.figure(figsize=(12, 10))
        
        # Custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # ===== 5. Create Heatmap =====
        with sns.axes_style("white"):
            ax = sns.heatmap(
                corr,
                mask=mask,
                cmap=cmap,
                vmin=-1,
                vmax=1,
                center=0,
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .8},
                annot_kws={"size": 9}
            )
            
            # Improve readability
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                horizontalalignment='right'
            )
            
            ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=0
            )
            
            plt.title("Feature Correlation Matrix\n", fontsize=14, pad=20)

        # ===== 6. Save Figure =====
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='w',
            transparent=False
        )
        print(f"✅ Successfully saved visualization to:\n{output_path}")

    except Exception as e:
        print(f"❌ Error in plot_correlation_matrix: {str(e)}")
        raise
    finally:
        plt.close()

# ===== Example Usage =====
if __name__ == "__main__":
    try:
        # Load your data
        data_path = Path('data/interim/featured_data.csv')
        print(f"Loading data from: {data_path}")
        
        df = pd.read_csv(data_path)
        print("✔ Data loaded successfully")
        
        # Generate and save correlation matrix
        plot_correlation_matrix(df)
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        import traceback
        traceback.print_exc()