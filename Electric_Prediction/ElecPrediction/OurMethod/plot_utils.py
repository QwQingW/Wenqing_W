import matplotlib.pyplot as plt


def plot_components(true_vals, pred_vals, trend_vals, seasonal_vals, title="Prediction Breakdown"):
    plt.figure(figsize=(16, 6))
    plt.plot(true_vals, label='True', linewidth=2)
    plt.plot(pred_vals, label='Predicted', linestyle='--')
    plt.plot(trend_vals, label='Trend', linestyle=':')
    plt.plot(seasonal_vals, label='Seasonal', linestyle='-.')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Global Active Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()