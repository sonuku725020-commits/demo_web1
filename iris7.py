import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

# -----------------------------------------------------------
# 🧠 Title and Description
# -----------------------------------------------------------
st.set_page_config(page_title="Bagging Ensemble Dashboard", layout="wide")
st.title("🎯 Bagging Ensemble Classifier Dashboard")
st.write(
    """This dashboard trains a **Bagging Ensemble** using `scikit-learn`.
    It visualizes the decision surface, confusion matrix, and individual trees."""
)

# -----------------------------------------------------------
# ⚙️ Sidebar Controls
# -----------------------------------------------------------
st.sidebar.header("Configure Model")

n_estimators = st.sidebar.slider("Number of Trees", 5, 50, 10, 1)
max_samples = st.sidebar.slider("Max Samples (fraction)", 0.1, 1.0, 0.8, 0.1)
max_depth = st.sidebar.slider("Max Tree Depth", 1, 10, 3, 1)
noise = st.sidebar.slider("Data Noise", 0.0, 0.5, 0.1, 0.05)
random_state = st.sidebar.number_input("Random State", 0, 999, 42)

# -----------------------------------------------------------
# 📊 Generate Synthetic Data
# -----------------------------------------------------------
X, y = make_classification(
    n_samples=400,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.5 - noise,
    random_state=random_state
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)

# -----------------------------------------------------------
# 🔍 Train Bagging Classifier
# -----------------------------------------------------------
base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
bag_clf = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=n_estimators,
    max_samples=max_samples,
    bootstrap=True,
    random_state=random_state
)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -----------------------------------------------------------
# 🎯 Performance Metrics
# -----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Model Performance")
    st.metric("Accuracy", f"{acc:.3f}")

with col2:
    st.subheader("Configuration")
    st.write(f"Number of Trees: {n_estimators}")
    st.write(f"Max Samples: {max_samples}")
    st.write(f"Max Depth: {max_depth}")

# -----------------------------------------------------------
# 🧩 Create Tabs for Visualizations
# -----------------------------------------------------------
tab1, tab2 = st.tabs(["Decision Boundary & Confusion Matrix", "Tree Visualization"])

# Tab 1: Decision Boundary & Confusion Matrix
with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Decision Boundary
    DecisionBoundaryDisplay.from_estimator(
        bag_clf,
        X_train,
        response_method="predict",
        cmap="RdBu",
        alpha=0.6,
        ax=axes[0]
    )
    axes[0].scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        edgecolor="k",
        cmap="RdBu",
    )
    axes[0].set_title("Decision Boundary")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted labels")
    axes[1].set_ylabel("True labels")

    st.pyplot(fig)

# Tab 2: Tree Visualization
with tab2:
    st.subheader("Tree Visualization")

    # Let user select which tree to visualize
    if n_estimators > 0:
        tree_idx = st.selectbox("Select Tree to Visualize", range(n_estimators), index=0)

        # Using matplotlib for tree visualization
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            bag_clf.estimators_[tree_idx],
            filled=True,
            rounded=True,
            feature_names=[f"Feature {i}" for i in range(X.shape[1])],
            class_names=[f"Class {i}" for i in np.unique(y)],
            ax=ax,
            fontsize=8
        )
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No trees available for visualization. Please increase the number of trees.")

# -----------------------------------------------------------
# 📊 Additional Metrics
# -----------------------------------------------------------
st.subheader("Tree Performance Distribution")

# Calculate accuracy of each individual tree
individual_accs = []
for i, tree in enumerate(bag_clf.estimators_):
    individual_accs.append(tree.score(X_test, y_test))

# Plot individual tree accuracies
if individual_accs:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(individual_accs)), individual_accs)
    ax.axhline(y=acc, color='r', linestyle='--', label='Ensemble Accuracy')
    ax.set_title("Accuracy of Individual Trees")
    ax.set_xlabel("Tree Index")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("No trees available for performance analysis.")

# -----------------------------------------------------------
# 📜 Summary Report
# -----------------------------------------------------------
st.subheader("Summary")
st.write(f"""
- **Training Samples:** {len(X_train)}  
- **Test Samples:** {len(X_test)}  
- **Number of Base Estimators:** {n_estimators}  
- **Max Samples per Estimator:** {max_samples}  
- **Max Tree Depth:** {max_depth}  
- **Random State:** {random_state}  
""")

st.markdown("---")
st.info("🌟 Tip: Use the tabs above to switch between different visualizations!")
st.write("🌳 *Each tree in the forest sees the world differently, but together they make better decisions!*")