import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

def plot_categorical_distributions(df, output_path=Path('reports/figures/categorical_distribution.png')
, max_categories=10):
    """
    Plots distributions of all categorical variables in a DataFrame and saves to file
    
    Args:
        df: Pandas DataFrame
        output_path: Path to save the visualization
        max_categories: Maximum categories to show per plot (avoids overcrowding)
    """
    try:
        print("→ Starting categorical distributions visualization...")
        
        # ===== 1. Data Validation =====
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        cat_attributes = df.select_dtypes(include=['object', 'category'])
        if cat_attributes.empty:
            raise ValueError("No categorical columns found in DataFrame")
            
        columns = cat_attributes.columns.tolist()
        print(f"✔ Found {len(columns)} categorical columns: {columns}")

        # ===== 2. Plot Setup =====
        n_cols = 2
        n_rows = (len(columns) + n_cols - 1) // n_cols  # Round up division
        plt.figure(figsize=(15, 5 * n_rows))
        plt.suptitle("Categorical Variables Distribution", y=1.02, fontsize=16)
        
        # ===== 3. Create Subplots =====
        for j, column in enumerate(columns, 1):
            plt.subplot(n_rows, n_cols, j)
            
            # Get value counts and limit categories
            value_counts = cat_attributes[column].value_counts().nlargest(max_categories)
            
            # Create horizontal bar plot
            ax = sns.countplot(
                y=column,
                data=cat_attributes,
                order=value_counts.index,
                # palette="Blues_r",
                # edgecolor="black"
            )
            
            # Add percentage annotations
            total = len(cat_attributes[column].dropna())
            for p in ax.patches:
                width = p.get_width()
                percentage = f"{100 * width/total:.1f}%"
                x = width + 0.01 * total  # Offset from bar
                y = p.get_y() + p.get_height()/2
                ax.annotate(percentage, (x, y), va='center')
            
            plt.title(f"{column}", pad=10)
            plt.xlabel('Count')
            plt.ylabel('')
            
            # Add grid for better readability
            plt.grid(axis='x', alpha=0.3)

        # ===== 4. Adjust Layout =====
        plt.tight_layout(pad=3.0)
        
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
        print(f"❌ Error in plot_categorical_distributions: {str(e)}")
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
        plot_categorical_distributions(df)
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        # For detailed debugging in VS Code:
        import traceback
        traceback.print_exc()