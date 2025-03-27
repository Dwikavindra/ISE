import pandas as pd

df_baseline_performance = pd.read_csv('base_line_tf/base_data_tensorflow_batchnorm.csv')
df_baseline_no_iteration = df_baseline_performance.drop(columns=['iteration'])
mean_values = df_baseline_no_iteration.mean()
std_values = df_baseline_no_iteration.std()
summary_stats = pd.DataFrame({
    'Mean': mean_values,
    'Standard Deviation': std_values
})

metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
means = df_baseline_performance[metric_cols].mean()

# Compute Euclidean distance from mean for each row
df_baseline_performance['distance_to_mean'] = (
    df_baseline_performance[metric_cols].sub(means).pow(2).sum(axis=1).pow(0.5)
)

closest_row = df_baseline_performance.loc[df_baseline_performance['distance_to_mean'].idxmin()]

print(f"Best representative iteration: is models/baseline_batchnorm/baseline_model_tensorflow_batchnorm{int(closest_row['iteration'])}_iteration.pt ")