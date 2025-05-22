import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# 1. Create the Confusion Matrix
def create_confusion_matrix():
    
    confusion_data = np.array([
        [615, 3, 9, 15, 60, 15, 12],
        [9, 66, 15, 12, 6, 3, 15],
        [9, 24, 3960, 15, 96, 60, 33],
        [15, 12, 3, 75, 6, 3, 36],
        [45, 12, 30, 12, 1875, 75, 15],
        [6, 3, 3, 12, 60, 345, 123],
        [3, 3, 3, 3, 6, 6, 519]
    ])
    
    # Define emotion labels
    emotions = ['focused', 'neutral', 'bored', 'surprised', 'confused', 'frustrated', 'happy']
    
    # Create a new figure with specified size
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with custom color scaling
    ax = sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotions, yticklabels=emotions,
                    linewidths=0.5)
    
    # Set labels and title
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title('Confusion Matrix ', fontsize=18)
    
    # Improve y-axis labels format
    ax.set_yticklabels([f"{emotion} -" for emotion in emotions])
    
    # Save and display
    plt.tight_layout()
    plt.savefig('confusion_matrix_generated.png', dpi=300)
    plt.show()


# 2. Create the Performance Metrics Bar Chart
def create_performance_chart():
    # Create DataFrame from the provided data
    data = {
        'Emotion': ['Focused', 'Neutral', 'Bored', 'Surprised', 'Confused', 'Frustrated', 'Happy'],
        'Precision': [0.87, 1.0, 0.92, 0.76, 0.88, 0.9, 0.89],
        'Recall': [0.91, 0.31, 0.97, 0.62, 0.9, 0.7, 1.0],
        'F1-Score': [0.88, 0.39, 0.95, 0.68, 0.89, 0.78, 0.95]
    }
    df = pd.DataFrame(data)
    
    # Create the bar chart
    plt.figure(figsize=(12, 7))
    
    # Set the positions for the bars
    bar_width = 0.25
    r1 = np.arange(len(df['Emotion']))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    plt.bar(r1, df['Precision'], width=bar_width, color='skyblue', label='Precision')
    plt.bar(r2, df['Recall'], width=bar_width, color='lightgreen', label='Recall')
    plt.bar(r3, df['F1-Score'], width=bar_width, color='salmon', label='F1-Score')
    
    # Add labels and title
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Emotion Detection Performance Metrics', fontsize=18)
    plt.xticks([r + bar_width for r in range(len(df['Emotion']))], df['Emotion'])
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper left', bbox_to_anchor=(0.05, 1))
    
    # Save and display
    plt.tight_layout()
    plt.savefig('performance_metrics_generated.png', dpi=300)
    plt.show()

# 3. Create the Classification Report Table
def create_classification_table():
    
    data = {
        'Emotion': ['Focused', 'Neutral', 'Bored', 'Surprised', 'Confused', 'Frustrated', 'Happy'],
        'Precision': [0.87, 1.0, 0.92, 0.76, 0.88, 0.9, 0.89],
        'Recall': [0.91, 0.31, 0.97, 0.62, 0.9, 0.7, 1.0],
        'F1-Score': [0.88, 0.39, 0.95, 0.68, 0.89, 0.78, 0.95]
    }
    df = pd.DataFrame(data)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')  # Hide axes
    
    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    
    # Style the header row
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f1f1f2')
    
    plt.title('Classification Report', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('classification_report_table.png', dpi=300)
    plt.show()

# Run all functions to generate the visualizations
if __name__ == "__main__":
    create_confusion_matrix()
    create_performance_chart()
    create_classification_table()

