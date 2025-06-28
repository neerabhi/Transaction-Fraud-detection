import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def save_plot(fig, filename, output_dir='reports/figures/bivariate_analysis'):
    """Helper function to save plots consistently"""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_fraud_analysis(df):
    """Generate all fraud analysis visualizations"""
    try:
        # 1. Validate and prepare data
        if 'is_fraud' not in df.columns:
            raise KeyError("'is_fraud' column not found in DataFrame")
        
        # Convert is_fraud to boolean if needed
        if df['is_fraud'].dtype == 'object':
            df['is_fraud'] = df['is_fraud'].str.lower().map({'yes': True, 'no': False, 'true': True, 'false': False})
        elif df['is_fraud'].dtype == 'int':
            df['is_fraud'] = df['is_fraud'].astype(bool)
        
        fraud_df = df[df['is_fraud'] == True]
        legit_df = df[df['is_fraud'] == False]
        
        if len(fraud_df) == 0:
            raise ValueError("No fraudulent transactions found in dataset")

        # 2. Origin/Destination Analysis
        plt.figure(figsize=(12, 6))
        
        # Top 10 Fraudulent Origins
        plt.subplot(1, 2, 1)
        top_origins = fraud_df['name_orig'].value_counts().nlargest(10)
        sns.barplot(y=top_origins.index, x=top_origins.values, palette='Blues_r')
        plt.title('Top 10 Fraudulent Origin Accounts')
        plt.xlabel('Count')
        plt.ylabel('Origin Account')
        
        # Top 10 Fraudulent Destinations
        plt.subplot(1, 2, 2)
        top_dests = fraud_df['name_dest'].value_counts().nlargest(10)
        sns.barplot(y=top_dests.index, x=top_dests.values, palette='Reds_r')
        plt.title('Top 10 Fraudulent Destination Accounts')
        plt.xlabel('Count')
        plt.ylabel('Destination Account')
        
        plt.tight_layout()
        save_plot(plt.gcf(), 'origin_dest_analysis.png')

        # 3. Amount Analysis
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_fraud', y='amount', data=df, showfliers=False)
        plt.title('Transaction Amount Distribution (Without Outliers)')
        plt.ylabel('Amount (log scale)')
        plt.xlabel('Is Fraud')
        plt.yscale('log')
        save_plot(plt.gcf(), 'amount_distribution.png')

        # 4. Transaction Type Analysis
        plt.figure(figsize=(12, 6))
        
        # Fraud percentage by type
        type_counts = df.groupby(['type', 'is_fraud']).size().unstack()
        type_counts['fraud_pct'] = type_counts[True] / (type_counts[True] + type_counts[False]) * 100
        
        sns.barplot(y=type_counts.index, x=type_counts['fraud_pct'], palette='viridis')
        plt.title('Fraud Percentage by Transaction Type')
        plt.xlabel('Fraud Percentage (%)')
        plt.ylabel('Transaction Type')
        
        # Add percentage labels
        for i, pct in enumerate(type_counts['fraud_pct']):
            plt.text(pct + 0.5, i, f'{pct:.2f}%', va='center')
        
        save_plot(plt.gcf(), 'fraud_by_type.png')

        # 5. Combined Type Distribution
        plt.figure(figsize=(12, 6))
        type_dist = df.groupby(['type', 'is_fraud']).size().unstack()
        type_dist.plot(kind='barh', stacked=True, color=['#1f77b4', '#ff7f0e'])
        plt.title('Transaction Type Distribution by Fraud Status')
        plt.xlabel('Count')
        plt.ylabel('Transaction Type')
        plt.legend(title='Is Fraud', labels=['Legitimate', 'Fraud'])
        save_plot(plt.gcf(), 'type_distribution.png')

    except Exception as e:
        print(f"Error in fraud analysis: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load data - adjust path as needed
        data_path = Path('data/interim/featured_data.csv')
        print(f"Loading data from: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Generate plots
        plot_fraud_analysis(df)
        
    except Exception as e:
        print(f"Script failed: {str(e)}")
        import traceback
        traceback.print_exc()