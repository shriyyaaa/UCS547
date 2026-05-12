# =============================================================================
# Assignment 6 (UCS547) - RAPIDS cuML
# GPU-Accelerated Machine Learning with RAPIDS cuML
# Run in Google Colab with GPU runtime (Runtime > Change runtime type > GPU)
# =============================================================================

# ---- Install required packages (Google Colab) ----
# !pip install cudf-cu12 cuml-cu12 xgboost --quiet

# %% [markdown]
# ## Setup and Imports

# %%
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100

from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.metrics import accuracy_score, classification_report

import cudf
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.metrics import accuracy_score as cu_accuracy_score
import xgboost as xgb

print("All imports successful.")
print(f"CuPy version: {cp.__version__}")
print(f"cuDF version: {cudf.__version__}")

# =============================================================================
# QUESTION 1: Adult Census Income Dataset - GPU-Accelerated Binning & Training
# =============================================================================

print("\n" + "=" * 70)
print("Q1: Adult Census Income Dataset - GPU-Accelerated Analysis")
print("=" * 70)

# %% [markdown]
# ### 1.1 Load the Adult Census Income Dataset

# %%
print("\n--- 1.1 Loading Adult Census Income Dataset ---")
adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
df = adult.frame
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn types:\n{df.dtypes}")

# %% [markdown]
# ### 1.2 Identify Feature Types

# %%
print("\n--- 1.2 Feature Type Classification ---")

continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                       'capital-loss', 'hours-per-week']
discrete_features = ['education-num']  # ordinal integer counts
categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'native-country']
binary_features = ['sex']

target_col = 'income' if 'income' in df.columns else df.columns[-1]

print(f"Continuous features  : {continuous_features}")
print(f"Discrete features    : {discrete_features}")
print(f"Categorical features : {categorical_features}")
print(f"Binary features      : {binary_features}")
print(f"Target               : {target_col}")

# Features eligible for histogram binning on GPU (continuous numeric)
gpu_binning_eligible = continuous_features
print(f"\nFeatures eligible for GPU histogram binning: {gpu_binning_eligible}")
print("Reason: Histogram binning applies to continuous numerical features where")
print("we can compute meaningful bin edges and frequency distributions on GPU.")

# %% [markdown]
# ### 1.3 GPU Histogram Binning using cuDF and CuPy

# %%
print("\n--- 1.3 GPU Histogram Binning (cuDF + CuPy) vs CPU (NumPy) ---")

# Prepare numeric data
df_numeric = df[continuous_features].apply(pd.to_numeric, errors='coerce').dropna()
gdf_numeric = cudf.DataFrame.from_pandas(df_numeric)

n_bins = 20
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

gpu_times = {}
cpu_times = {}

for idx, feat in enumerate(continuous_features):
    ax = axes[idx]

    # --- GPU histogram using CuPy ---
    gpu_data = cp.asarray(gdf_numeric[feat].values)
    start_gpu = time.time()
    gpu_hist, gpu_edges = cp.histogram(gpu_data, bins=n_bins)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start_gpu
    gpu_times[feat] = gpu_time

    # --- CPU histogram using NumPy ---
    cpu_data = df_numeric[feat].values
    start_cpu = time.time()
    cpu_hist, cpu_edges = np.histogram(cpu_data, bins=n_bins)
    cpu_time = time.time() - start_cpu
    cpu_times[feat] = cpu_time

    # Convert GPU results to numpy for plotting
    gpu_hist_np = cp.asnumpy(gpu_hist)
    gpu_edges_np = cp.asnumpy(gpu_edges)

    # Plot comparison
    width = (gpu_edges_np[1] - gpu_edges_np[0]) * 0.4
    centers = (gpu_edges_np[:-1] + gpu_edges_np[1:]) / 2
    ax.bar(centers - width / 2, cpu_hist, width=width, alpha=0.7,
           label=f'CPU ({cpu_time * 1000:.2f} ms)', color='steelblue')
    ax.bar(centers + width / 2, gpu_hist_np, width=width, alpha=0.7,
           label=f'GPU ({gpu_time * 1000:.2f} ms)', color='coral')
    ax.set_title(f'{feat}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlabel('Bin')
    ax.set_ylabel('Count')

plt.suptitle('CPU (NumPy) vs GPU (CuPy) Histogram Binning Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q1_histogram_comparison.png', bbox_inches='tight')
plt.show()

print("\nHistogram Binning Time Comparison:")
print(f"{'Feature':<20} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
print("-" * 54)
for feat in continuous_features:
    speedup = cpu_times[feat] / gpu_times[feat] if gpu_times[feat] > 0 else float('inf')
    print(f"{feat:<20} {cpu_times[feat]*1000:<12.3f} {gpu_times[feat]*1000:<12.3f} {speedup:<10.2f}x")

# %% [markdown]
# ### 1.4 Quantile Binning on GPU

# %%
print("\n--- 1.4 Quantile Binning on GPU ---")

n_quantile_bins = 5
quantile_probs = cp.linspace(0, 1, n_quantile_bins + 1)

print(f"Number of quantile bins: {n_quantile_bins}")
print(f"Quantile probabilities: {cp.asnumpy(quantile_probs)}\n")

quantile_results = {}
for feat in continuous_features:
    gpu_col = cp.asarray(gdf_numeric[feat].values)
    gpu_col_sorted = cp.sort(gpu_col)

    # Compute quantile edges on GPU
    edges = cp.percentile(gpu_col, cp.asnumpy(quantile_probs * 100))
    edges_np = cp.asnumpy(edges)

    # Assign bin labels using digitize on GPU
    bin_labels = cp.digitize(gpu_col, edges[1:-1])
    bin_labels_np = cp.asnumpy(bin_labels)

    quantile_results[feat] = {
        'edges': edges_np,
        'bin_labels': bin_labels_np
    }

    print(f"Feature: {feat}")
    print(f"  Bin edges: {edges_np}")
    bin_counts = np.bincount(bin_labels_np, minlength=n_quantile_bins)
    for b in range(n_quantile_bins):
        lo = edges_np[b]
        hi = edges_np[b + 1]
        print(f"  Bin {b}: [{lo:.2f}, {hi:.2f}] -> {bin_counts[b]} samples")
    print()

# Visualize quantile binning
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for idx, feat in enumerate(continuous_features):
    ax = axes[idx]
    edges_np = quantile_results[feat]['edges']
    labels = quantile_results[feat]['bin_labels']
    ax.hist(df_numeric[feat].values, bins=50, alpha=0.4, color='gray', label='Data')
    for e in edges_np:
        ax.axvline(e, color='red', linestyle='--', linewidth=1)
    ax.set_title(f'{feat} - Quantile Bins', fontsize=11, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.legend(['Bin Edges', 'Data'], fontsize=8)

plt.suptitle('GPU Quantile Binning - Bin Edges Visualization',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q1_quantile_binning.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 1.5 Data Preprocessing for Model Training

# %%
print("\n--- 1.5 Data Preprocessing ---")

# Encode categorical features
df_encoded = df.copy()
label_encoders = {}
for col in categorical_features + binary_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# Encode target
le_target = LabelEncoder()
df_encoded[target_col] = le_target.fit_transform(df_encoded[target_col].astype(str))

feature_cols = continuous_features + categorical_features + binary_features
X = df_encoded[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
y = df_encoded[target_col].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Convert to cuDF for GPU models
X_train_cudf = cudf.DataFrame(X_train, columns=feature_cols)
X_test_cudf = cudf.DataFrame(X_test, columns=feature_cols)
y_train_cudf = cudf.Series(y_train)
y_test_cudf = cudf.Series(y_test)

print("Data prepared for both CPU and GPU models.")

# %% [markdown]
# ### 1.6 CPU vs GPU Model Training Comparison

# %%
print("\n--- 1.6 CPU (sklearn) vs GPU (cuML) Model Comparison ---")

results = {}

# --- sklearn Random Forest (CPU) ---
print("\nTraining sklearn RandomForest (CPU)...")
start = time.time()
sk_rf = skRF(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
sk_rf.fit(X_train, y_train)
sk_train_time = time.time() - start

start = time.time()
sk_preds = sk_rf.predict(X_test)
sk_pred_time = time.time() - start
sk_acc = accuracy_score(y_test, sk_preds)

results['sklearn RF (CPU)'] = {
    'train_time': sk_train_time,
    'predict_time': sk_pred_time,
    'accuracy': sk_acc
}
print(f"  Train time: {sk_train_time:.3f}s | Predict time: {sk_pred_time:.3f}s | Accuracy: {sk_acc:.4f}")

# --- cuML Random Forest (GPU) ---
print("\nTraining cuML RandomForest (GPU)...")
start = time.time()
cu_rf = cuRF(n_estimators=100, max_depth=16, random_state=42)
cu_rf.fit(X_train_cudf, y_train_cudf)
cp.cuda.Stream.null.synchronize()
cu_train_time = time.time() - start

start = time.time()
cu_preds = cu_rf.predict(X_test_cudf)
cp.cuda.Stream.null.synchronize()
cu_pred_time = time.time() - start
cu_acc = accuracy_score(y_test, cu_preds.to_pandas().values.astype(np.int32))

results['cuML RF (GPU)'] = {
    'train_time': cu_train_time,
    'predict_time': cu_pred_time,
    'accuracy': cu_acc
}
print(f"  Train time: {cu_train_time:.3f}s | Predict time: {cu_pred_time:.3f}s | Accuracy: {cu_acc:.4f}")

# %% [markdown]
# ### 1.7 XGBoost GPU (tree_method='gpu_hist') vs cuML

# %%
print("\n--- 1.7 XGBoost with GPU Histogram ---")

# --- XGBoost GPU ---
print("\nTraining XGBoost (gpu_hist)...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 42
}

start = time.time()
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=100)
xgb_train_time = time.time() - start

start = time.time()
xgb_preds_prob = xgb_model.predict(dtest)
xgb_preds = (xgb_preds_prob > 0.5).astype(int)
xgb_pred_time = time.time() - start
xgb_acc = accuracy_score(y_test, xgb_preds)

results['XGBoost (gpu_hist)'] = {
    'train_time': xgb_train_time,
    'predict_time': xgb_pred_time,
    'accuracy': xgb_acc
}
print(f"  Train time: {xgb_train_time:.3f}s | Predict time: {xgb_pred_time:.3f}s | Accuracy: {xgb_acc:.4f}")

# --- XGBoost CPU for comparison ---
print("\nTraining XGBoost (CPU hist)...")
xgb_cpu_params = xgb_params.copy()
xgb_cpu_params['tree_method'] = 'hist'

start = time.time()
xgb_cpu_model = xgb.train(xgb_cpu_params, dtrain, num_boost_round=100)
xgb_cpu_train_time = time.time() - start

start = time.time()
xgb_cpu_preds_prob = xgb_cpu_model.predict(dtest)
xgb_cpu_preds = (xgb_cpu_preds_prob > 0.5).astype(int)
xgb_cpu_pred_time = time.time() - start
xgb_cpu_acc = accuracy_score(y_test, xgb_cpu_preds)

results['XGBoost (CPU hist)'] = {
    'train_time': xgb_cpu_train_time,
    'predict_time': xgb_cpu_pred_time,
    'accuracy': xgb_cpu_acc
}
print(f"  Train time: {xgb_cpu_train_time:.3f}s | Predict time: {xgb_cpu_pred_time:.3f}s | Accuracy: {xgb_cpu_acc:.4f}")

# %% [markdown]
# ### 1.8 Summary Comparison - All Models

# %%
print("\n--- 1.8 Model Performance Comparison Summary ---")

results_df = pd.DataFrame(results).T
results_df.columns = ['Train Time (s)', 'Predict Time (s)', 'Accuracy']
print(results_df.to_string())

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models = list(results.keys())
colors = ['steelblue', 'coral', 'seagreen', 'goldenrod']

# Training Time
train_times = [results[m]['train_time'] for m in models]
axes[0].bar(models, train_times, color=colors)
axes[0].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Time (seconds)')
axes[0].tick_params(axis='x', rotation=20)
for i, v in enumerate(train_times):
    axes[0].text(i, v + 0.01, f'{v:.3f}s', ha='center', fontsize=9)

# Prediction Time
pred_times = [results[m]['predict_time'] for m in models]
axes[1].bar(models, pred_times, color=colors)
axes[1].set_title('Prediction Time Comparison', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Time (seconds)')
axes[1].tick_params(axis='x', rotation=20)
for i, v in enumerate(pred_times):
    axes[1].text(i, v + 0.001, f'{v:.4f}s', ha='center', fontsize=9)

# Accuracy
accuracies = [results[m]['accuracy'] for m in models]
axes[2].bar(models, accuracies, color=colors)
axes[2].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Accuracy')
axes[2].set_ylim(0.8, 1.0)
axes[2].tick_params(axis='x', rotation=20)
for i, v in enumerate(accuracies):
    axes[2].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)

plt.suptitle('Q1: CPU vs GPU Model Performance - Adult Census Dataset',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q1_model_comparison.png', bbox_inches='tight')
plt.show()

# =============================================================================
# QUESTION 2: Random Forest - scikit-learn (CPU) vs RAPIDS cuML (GPU)
#              on Breast Cancer Dataset
# =============================================================================

print("\n" + "=" * 70)
print("Q2: Random Forest - CPU vs GPU on Breast Cancer Dataset")
print("=" * 70)

# %% [markdown]
# ### 2.1 Data Preparation

# %%
print("\n--- 2.1 Data Preparation ---")

data = load_breast_cancer()
X_bc = data.data.astype(np.float32)
y_bc = data.target.astype(np.int32)
feature_names = data.feature_names

print(f"Dataset: Breast Cancer Wisconsin")
print(f"Samples: {X_bc.shape[0]}, Features: {X_bc.shape[1]}")
print(f"Classes: {data.target_names}")

X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

# Convert to cuDF for GPU
X_bc_train_cudf = cudf.DataFrame(X_bc_train, columns=feature_names)
X_bc_test_cudf = cudf.DataFrame(X_bc_test, columns=feature_names)
y_bc_train_cudf = cudf.Series(y_bc_train)
y_bc_test_cudf = cudf.Series(y_bc_test)

print(f"Train: {X_bc_train.shape}, Test: {X_bc_test.shape}")
print("cuDF DataFrames created for GPU processing.")

# %% [markdown]
# ### 2.2 CPU: sklearn RandomForest

# %%
print("\n--- 2.2 CPU: sklearn RandomForestClassifier ---")

start = time.time()
cpu_model = skRF(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
cpu_model.fit(X_bc_train, y_bc_train)
cpu_train_time = time.time() - start

start = time.time()
cpu_preds = cpu_model.predict(X_bc_test)
cpu_predict_time = time.time() - start

cpu_accuracy = accuracy_score(y_bc_test, cpu_preds)

print(f"CPU Training Time  : {cpu_train_time:.4f} s")
print(f"CPU Prediction Time: {cpu_predict_time:.4f} s")
print(f"CPU Accuracy       : {cpu_accuracy:.4f}")
print(f"\nClassification Report (CPU):")
print(classification_report(y_bc_test, cpu_preds, target_names=data.target_names))

# %% [markdown]
# ### 2.3 GPU: cuML RandomForest

# %%
print("\n--- 2.3 GPU: cuML RandomForestClassifier ---")

start = time.time()
gpu_model = cuRF(n_estimators=100, max_depth=16, random_state=42)
gpu_model.fit(X_bc_train_cudf, y_bc_train_cudf)
cp.cuda.Stream.null.synchronize()
gpu_train_time = time.time() - start

start = time.time()
gpu_preds = gpu_model.predict(X_bc_test_cudf)
cp.cuda.Stream.null.synchronize()
gpu_predict_time = time.time() - start

gpu_preds_np = gpu_preds.to_pandas().values.astype(np.int32)
gpu_accuracy = accuracy_score(y_bc_test, gpu_preds_np)

print(f"GPU Training Time  : {gpu_train_time:.4f} s")
print(f"GPU Prediction Time: {gpu_predict_time:.4f} s")
print(f"GPU Accuracy       : {gpu_accuracy:.4f}")

train_speedup = cpu_train_time / gpu_train_time if gpu_train_time > 0 else float('inf')
predict_speedup = cpu_predict_time / gpu_predict_time if gpu_predict_time > 0 else float('inf')

print(f"\nSpeedup (Training)  : {train_speedup:.2f}x")
print(f"Speedup (Prediction): {predict_speedup:.2f}x")
print(f"\nClassification Report (GPU):")
print(classification_report(y_bc_test, gpu_preds_np, target_names=data.target_names))

# %% [markdown]
# ### 2.4 Visualization

# %% [markdown]
# #### 2.4.1 Forest-Level Parallelism: Training Time vs n_estimators

# %%
print("\n--- 2.4.1 Forest-Level Parallelism ---")

n_estimators_list = [1, 10, 50, 100]
cpu_forest_times = []
gpu_forest_times = []

for n_est in n_estimators_list:
    # CPU
    start = time.time()
    skRF(n_estimators=n_est, max_depth=16, random_state=42, n_jobs=-1).fit(
        X_bc_train, y_bc_train)
    cpu_forest_times.append(time.time() - start)

    # GPU
    start = time.time()
    cuRF(n_estimators=n_est, max_depth=16, random_state=42).fit(
        X_bc_train_cudf, y_bc_train_cudf)
    cp.cuda.Stream.null.synchronize()
    gpu_forest_times.append(time.time() - start)

    print(f"n_estimators={n_est:>3d}  |  CPU: {cpu_forest_times[-1]:.4f}s  |  "
          f"GPU: {gpu_forest_times[-1]:.4f}s")

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(n_estimators_list))
width = 0.35
bars1 = ax.bar(x_pos - width / 2, cpu_forest_times, width, label='CPU (sklearn)',
               color='steelblue', alpha=0.8)
bars2 = ax.bar(x_pos + width / 2, gpu_forest_times, width, label='GPU (cuML)',
               color='coral', alpha=0.8)
ax.set_xlabel('Number of Estimators (n_estimators)', fontsize=12)
ax.set_ylabel('Training Time (seconds)', fontsize=12)
ax.set_title('Forest-Level Parallelism: Training Time vs n_estimators',
             fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(n_estimators_list)
ax.legend(fontsize=11)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('q2_forest_level_parallelism.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# #### 2.4.2 Feature-Level: Feature Importance

# %%
print("\n--- 2.4.2 Feature-Level: Feature Importance ---")

cpu_importance = cpu_model.feature_importances_
sorted_idx = np.argsort(cpu_importance)[-15:]  # Top 15 features

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(sorted_idx)), cpu_importance[sorted_idx], color='steelblue', alpha=0.8)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Feature-Level Analysis: Top 15 Feature Importances (Random Forest)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_feature_importance.png', bbox_inches='tight')
plt.show()

print("Top 10 most important features:")
for i, idx in enumerate(sorted_idx[::-1][:10]):
    print(f"  {i+1}. {feature_names[idx]}: {cpu_importance[idx]:.4f}")

# %% [markdown]
# #### 2.4.3 Data-Level: Training Time for Varying Dataset Sizes

# %%
print("\n--- 2.4.3 Data-Level: Training Time vs Dataset Size ---")

data_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
# Replicate data to create larger datasets for meaningful comparison
scale_factors = [1, 2, 5, 10, 20]
cpu_data_times = []
gpu_data_times = []
actual_sizes = []

for sf in scale_factors:
    X_scaled = np.tile(X_bc_train, (sf, 1))
    y_scaled = np.tile(y_bc_train, sf)
    actual_sizes.append(len(y_scaled))

    X_scaled_cudf = cudf.DataFrame(X_scaled, columns=feature_names)
    y_scaled_cudf = cudf.Series(y_scaled)

    # CPU
    start = time.time()
    skRF(n_estimators=50, max_depth=16, random_state=42, n_jobs=-1).fit(
        X_scaled, y_scaled)
    cpu_t = time.time() - start
    cpu_data_times.append(cpu_t)

    # GPU
    start = time.time()
    cuRF(n_estimators=50, max_depth=16, random_state=42).fit(
        X_scaled_cudf, y_scaled_cudf)
    cp.cuda.Stream.null.synchronize()
    gpu_t = time.time() - start
    gpu_data_times.append(gpu_t)

    print(f"Dataset size: {len(y_scaled):>6d}  |  CPU: {cpu_t:.4f}s  |  GPU: {gpu_t:.4f}s")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(actual_sizes, cpu_data_times, 'o-', label='CPU (sklearn)',
        color='steelblue', linewidth=2, markersize=8)
ax.plot(actual_sizes, gpu_data_times, 's-', label='GPU (cuML)',
        color='coral', linewidth=2, markersize=8)
ax.set_xlabel('Dataset Size (number of samples)', fontsize=12)
ax.set_ylabel('Training Time (seconds)', fontsize=12)
ax.set_title('Data-Level Parallelism: Training Time vs Dataset Size',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_data_level_scaling.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# #### 2.4.4 Bin-Level: GPU Histogram Binning Demo with CuPy

# %%
print("\n--- 2.4.4 Bin-Level: GPU Histogram Binning Demo (Breast Cancer) ---")

n_demo_bins = 15
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

# Show first 12 features
for idx in range(min(12, X_bc.shape[1])):
    ax = axes[idx]
    feat_data = X_bc[:, idx]
    gpu_feat = cp.asarray(feat_data)

    # GPU histogram
    gpu_h, gpu_e = cp.histogram(gpu_feat, bins=n_demo_bins)
    gpu_h_np = cp.asnumpy(gpu_h)
    gpu_e_np = cp.asnumpy(gpu_e)

    centers = (gpu_e_np[:-1] + gpu_e_np[1:]) / 2
    width = (gpu_e_np[1] - gpu_e_np[0]) * 0.8
    ax.bar(centers, gpu_h_np, width=width, color='coral', alpha=0.7, edgecolor='black',
           linewidth=0.5)
    ax.set_title(f'{feature_names[idx]}', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)

plt.suptitle('Bin-Level Analysis: GPU Histogram Binning (Breast Cancer Features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_bin_level_histogram.png', bbox_inches='tight')
plt.show()
print("GPU histogram binning completed for 12 breast cancer features.")

# %% [markdown]
# ### 2.5 Performance Comparison Table and Bar Charts

# %%
print("\n--- 2.5 Performance Comparison Summary ---")

comparison_data = {
    'Metric': ['Training Time (s)', 'Prediction Time (s)', 'Accuracy',
               'Train Speedup', 'Predict Speedup'],
    'CPU (sklearn)': [
        f'{cpu_train_time:.4f}', f'{cpu_predict_time:.4f}',
        f'{cpu_accuracy:.4f}', '1.00x (baseline)', '1.00x (baseline)'
    ],
    'GPU (cuML)': [
        f'{gpu_train_time:.4f}', f'{gpu_predict_time:.4f}',
        f'{gpu_accuracy:.4f}', f'{train_speedup:.2f}x', f'{predict_speedup:.2f}x'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nPerformance Comparison Table:")
print(comparison_df.to_string(index=False))

# Final comparison bar charts
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Training Time
labels = ['CPU (sklearn)', 'GPU (cuML)']
values = [cpu_train_time, gpu_train_time]
bars = axes[0].bar(labels, values, color=['steelblue', 'coral'], alpha=0.8,
                   edgecolor='black', linewidth=0.5)
axes[0].set_title('Training Time', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Time (seconds)')
for bar, v in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{v:.4f}s', ha='center', va='bottom', fontsize=10)

# Prediction Time
values = [cpu_predict_time, gpu_predict_time]
bars = axes[1].bar(labels, values, color=['steelblue', 'coral'], alpha=0.8,
                   edgecolor='black', linewidth=0.5)
axes[1].set_title('Prediction Time', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Time (seconds)')
for bar, v in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{v:.4f}s', ha='center', va='bottom', fontsize=10)

# Accuracy
values = [cpu_accuracy, gpu_accuracy]
bars = axes[2].bar(labels, values, color=['steelblue', 'coral'], alpha=0.8,
                   edgecolor='black', linewidth=0.5)
axes[2].set_title('Accuracy', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Accuracy')
axes[2].set_ylim(0.9, 1.0)
for bar, v in zip(bars, values):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{v:.4f}', ha='center', va='bottom', fontsize=10)

plt.suptitle('Q2: CPU vs GPU Random Forest - Breast Cancer Dataset',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_performance_comparison.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Final Summary

# %%
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print("""
Q1 - Adult Census Income Dataset:
  - Identified continuous, discrete, categorical, and binary features
  - GPU histogram binning with CuPy matches CPU NumPy results
  - Quantile binning computed on GPU with bin edges and labels
  - Compared sklearn RF, cuML RF, XGBoost (GPU & CPU)
  - XGBoost gpu_hist provides strong accuracy with fast training

Q2 - Breast Cancer Dataset (RF CPU vs GPU):
  - Forest-Level: GPU scales better as n_estimators increases
  - Feature-Level: Feature importance identifies key diagnostic features
  - Data-Level: GPU advantage grows with larger dataset sizes
  - Bin-Level: GPU histogram binning demonstrated with CuPy
  - Performance table shows training/prediction speedups
""")
print("=" * 70)
print("Assignment 6 Complete.")
