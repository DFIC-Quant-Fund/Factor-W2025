import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm

# Step 1: Collect Monthly Returns of QQQ
qqq_daily = yf.download("QQQ", start="2006-01-01", end="2023-12-31")
qqq_monthly = qqq_daily["Close"].resample("ME").mean()
qqq_monthly.name = "QQQ"
qqq_monthly = pd.DataFrame(qqq_monthly)
qqq_monthly["Return"] = qqq_monthly["QQQ"].pct_change() * 100  # Calculate percentage change
qqq_monthly.dropna(inplace=True)  # Drop rows with NaN values
qqq_monthly.index = qqq_monthly.index.to_period("M")  # Convert index to year-month format

# Step 2: Load Fama-French Three-Factor Data
ff_factors_monthly = pd.read_csv("../data/F-F_Research_Data_Factors_Jer.CSV", index_col=0)
ff_factors_monthly = ff_factors_monthly[ff_factors_monthly.index.notna()]
ff_factors_monthly = ff_factors_monthly[ff_factors_monthly.index.str.match(r"^\d{6}$")]
ff_factors_monthly.index.names = ["Date"]
ff_factors_monthly.index = pd.to_datetime(ff_factors_monthly.index, format="%Y%m")
ff_factors_monthly.index = ff_factors_monthly.index.to_period("M")


# Step 3: Match factor dates to match the asset (Put returns onto ff_factors_monthly)
ff_factors_subset = ff_factors_monthly.loc[ff_factors_monthly.index.isin(qqq_monthly.index)].copy()
ff_factors_subset["RF"] = pd.to_numeric(ff_factors_subset["RF"], errors="coerce")
aligned_returns = qqq_monthly.loc[qqq_monthly.index.isin(ff_factors_subset.index), "Return"]
aligned_returns = pd.to_numeric(aligned_returns, errors="coerce")
ff_factors_subset["Excess_Return"] = aligned_returns.values - ff_factors_subset["RF"]
ff_factors_subset = ff_factors_subset[["Mkt-RF", "SMB", "HML", "RF", "Excess_Return"]]

#Step 4: Regression

# Ensure there are no NaN values in X or y and convert to numeric
X = sm.add_constant(ff_factors_subset[["Mkt-RF", "SMB", "HML"]]) 
X = X.apply(pd.to_numeric, errors="coerce")  # Ensure all values are numeric
X = X.dropna() 

y = ff_factors_subset["Excess_Return"]  
y = pd.to_numeric(y, errors="coerce")  
y = y.loc[X.index]

if X.empty or y.empty:
    raise ValueError("X or y is empty after cleaning. Check your data for issues.")

# Run the regression
model = sm.OLS(y, X).fit()
print(model.summary())

# Step 5: plot
factors = model.params.index[1:]  # ['Mkt_Rf', 'SMB', 'HML']
coefficients = model.params.values[1:]
confidence_intervals = model.conf_int().diff(axis=1).iloc[1]

# Create a DataFrame
ols_data = pd.DataFrame(
    {
        "Factor": factors,
        "Coefficient": coefficients,
        "Confidence_Lower": confidence_intervals[0],
        "Confidence_Upper": confidence_intervals[1],
    }
)

# Step 5.2: Plot the coefficients and their confidence intervals
factors = model.params.index[1:]  # ['Mkt-RF', 'SMB', 'HML']
coefficients = model.params.values[1:]
confidence_intervals = model.conf_int().iloc[1:, :]  # Confidence intervals for factors

ols_data = pd.DataFrame(
    {
        "Factor": factors,
        "Coefficient": coefficients,
        "Confidence_Lower": confidence_intervals[0].values,
        "Confidence_Upper": confidence_intervals[1].values,
    }
)

# Plotting
plt.figure(figsize=(8, 5))
sns.barplot(
    x="Factor",
    y="Coefficient",
    hue="Factor", 
    data=ols_data,
    capsize=0.2,
    errorbar=None,  # Replace deprecated `ci` parameter
    palette="coolwarm",
    legend=False, 
)

# Add error bars for confidence intervals
for i, row in ols_data.iterrows():
    plt.plot(
        [i, i],
        [row["Confidence_Lower"], row["Confidence_Upper"]],
        color="black",
        linewidth=1.5,
    )

# Add the p-value for each factor to the plot
for i, factor in enumerate(factors):
    p_value = model.pvalues[factor]
    plt.text(
        i,
        coefficients[i] + 0.1,  # Adjust position above the bar
        f"p={p_value:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
    )

plt.title("Impact of Fama-French Factors on QQQ Monthly Returns (2006-2023)")
plt.xlabel("Factor")
plt.ylabel("Coefficient Value")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()

# Save the plot to a file instead of showing it interactively
plt.savefig("imgs/fama_french_coefficients.png")
print("Plot saved as 'fama_french_coefficients.png'")