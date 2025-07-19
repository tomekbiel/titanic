import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ------------- LOAD -------------
def load_titanic():
    df = sns.load_dataset("titanic")
    print(f"Loaded Titanic dataset shape = {df.shape}")
    return df

# ------------- PLOTS -------------
def plot_class_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="pclass", palette="viridis")
    plt.title("Passenger class distribution")
    plt.xlabel("Pclass")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_survival_by_sex_class(df):
    plt.figure(figsize=(8,5))
    pivot = pd.crosstab(index=df["sex"], columns=df["pclass"], values=df["survived"], aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Survival rate by Sex and Pclass")
    plt.ylabel("Sex"); plt.xlabel("Pclass")
    plt.tight_layout()
    plt.show()

def plot_age_distribution(df):
    g = sns.FacetGrid(df, col="survived", height=4, aspect=1.2)
    g.map_dataframe(sns.histplot, x="age", bins=20, kde=True, color="steelblue")
    g.set_titles(col_template="Survived = {col_name}")
    g.set_axis_labels("Age", "Count")
    plt.tight_layout()
    plt.show()

def plot_fare_by_class(df):
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="pclass", y="fare", palette="Set2")
    plt.yscale("log")      # rozrzut jest spory
    plt.title("Fare distribution by Pclass (log scale)")
    plt.tight_layout()
    plt.show()

def plot_embarked(df):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="embarked", palette="pastel")
    plt.title("Embarkation port counts")
    plt.tight_layout()
    plt.show()

def plot_family_size(df):
    df = df.copy()
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    plt.figure(figsize=(7,4))
    sns.countplot(data=df, x="family_size", palette="crest")
    plt.title("Family size distribution")
    plt.tight_layout()
    plt.show()

def plot_corr_heatmap(df):
    numeric = df.select_dtypes(include=["number"])
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric.corr(), cmap="coolwarm", annot=False, center=0)
    plt.title("Correlation matrix (numeric features)")
    plt.tight_layout()
    plt.show()

# ------------- MAIN -------------
def main():
    df = load_titanic()

    # przykładowy pipeline
    plot_class_distribution(df)
    plot_survival_by_sex_class(df)
    plot_age_distribution(df)
    plot_fare_by_class(df)
    plot_embarked(df)
    plot_family_size(df)
    plot_corr_heatmap(df)

if __name__ == "__main__":
    main()