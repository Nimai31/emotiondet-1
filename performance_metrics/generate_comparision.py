import matplotlib.pyplot as plt
import pandas as pd

# Define the data with all requested columns and values
data = {
    "System": [
        "Our Model",
        "WSCNet",
        "CNN Custom",
        "AffectGPT",
        "ResNet-50",
        "VGG-16",
        "AlexNet"
    ],
    "Dataset": [
        "Custom Dataset",
        "FI",
        "Kaggle FER",
        "MER-UniBench",
        "FI",
        "FER-2013",
        "FI"
    ],
    "Precision": [0.89, 0.70, 0.69, 0.76, 0.94, 0.66, 0.60],
    "Recall": [0.90, 0.68, 0.67, 0.74, 0.86, 0.65, 0.60],
    "F1-Score": [0.89, 0.70, 0.69, 0.75, 0.89, 0.66, 0.60],
    "Accuracy": [0.8916, 0.70, 0.69, 0.76, 0.858, 0.6552, 0.5985],
    "Source": [
        "",
        "Viso Suite, 2025 (FI dataset)",
        "GitHub: prathmesh444/Emotion-Detection-using-Face-Recognition (Kaggle FER)",
        "AffectGPT: arXiv:2501.16566v1 (MER-UniBench)",
        "ETASR 2024 (ResNet-50, FER)",
        "Viso Suite 2025 (VGG-16, FER-2013)",
        "Viso Suite 2025 (AlexNet, FI)"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plot the table
def generate_comparison_table():
    table_df = df[["System", "Dataset", "Precision", "Recall", "F1-Score", "Accuracy", "Source"]]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(len(table_df.columns))))

    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f1f1f2')

    # Title and source note
    plt.title("Comparison of Emotion Recognition Models", fontsize=16, pad=20)
    plt.figtext(0.5, 0.01, "Metrics and sources are drawn from user input and documented studies.",
                wrap=True, horizontalalignment='center', fontsize=10, style='italic')
    plt.tight_layout()
    plt.savefig('comparison_table_final.png', dpi=300)
    plt.show()

# Generate the table
generate_comparison_table()
