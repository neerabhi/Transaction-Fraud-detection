import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os

def plot_numerical_distributions(df, output_path=Path('reports/figures/numerical_distribution.png')
, max_categories=10):
    """
    Plots distributions of all numerical variables in a DataFrame and saves to file
    
    Args:
        df: Pandas DataFrame
        output_path: Path to save the visualization
    """
    try:
        print("→ Starting numerical distributions visualization...")
        
        # ===== 1. Data Validation =====
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        num_attributes = df.select_dtypes(exclude='object')
        if num_attributes.empty:
            raise ValueError("No numerical columns found in DataFrame")
            
        columns = num_attributes.columns.tolist()
        print(f"✔ Found {len(columns)} numerical columns: {columns}")

        # ===== 2. Plot Setup =====
        plt.figure(figsize=(20, 10))
        plt.suptitle("Numerical Variables Distribution", y=1.02, fontsize=16)
        
        # Determine grid size dynamically
        n_cols = 5
        n_rows = (len(columns) + n_cols - 1) // n_cols  # Round up division
        
        # ===== 3. Create Subplots =====
        for j, column in enumerate(columns, 1):
            plt.subplot(n_rows, n_cols, j)
            
            # Use histplot instead of deprecated distplot
            sns.histplot(
                num_attributes[column],
                kde=True,
                color='skyblue',
                edgecolor='navy',
                linewidth=0.5
            )
            
            plt.title(f"{column}", pad=10)
            plt.xlabel('')
            plt.ylabel('')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')

        # ===== 4. Adjust Layout =====
        plt.tight_layout(pad=2.0)
        
        # ===== 5. Save Figure =====
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='w'
        )
        print(f"✅ Successfully saved visualization to:\n{output_path}")

    except Exception as e:
        print(f"❌ Error in plot_numerical_distributions: {str(e)}")
        raise
    finally:
        plt.close()

# ===== Example Usage =====
if __name__ == "__main__":
    try:
        # Load your data
        data_path = Path('data/interim/featured_data.csv')  # Update path
        print(f"Loading data from: {data_path}")
        
        df = pd.read_csv(data_path)
        print("✔ Data loaded successfully")
        
        # Generate and save plots
        plot_numerical_distributions(df)
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        # For detailed debugging in VS Code:
        import traceback
        traceback.print_exc()