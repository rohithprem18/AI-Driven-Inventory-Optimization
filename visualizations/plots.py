import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    sns.set_theme(style="whitegrid")


def plot_sales_trends(df, product_id, output_path):
    set_style()
    subset = df[df["product_id"] == product_id].copy()
    subset = subset.sort_values("date")
    subset["rolling_30"] = subset["daily_sales"].rolling(30).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(subset["date"], subset["daily_sales"], label="Daily Sales", alpha=0.5)
    plt.plot(subset["date"], subset["rolling_30"], label="30-Day Rolling Avg", linewidth=2)
    plt.title(f"Sales Trend for {product_id}")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_actual_vs_pred(actual, pred, output_path, title):
    set_style()
    plt.figure(figsize=(12, 5))
    plt.plot(actual.index, actual.values, label="Actual")
    plt.plot(actual.index, pred.values, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Daily Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_inventory_levels(before_df, after_df, output_path):
    set_style()
    plt.figure(figsize=(12, 5))
    plt.plot(before_df.index, before_df["inventory_level"], label="Before Optimization")
    plt.plot(after_df.index, after_df["inventory_level"], label="After Optimization")
    plt.title("Inventory Level Comparison")
    plt.xlabel("Date")
    plt.ylabel("Inventory Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
