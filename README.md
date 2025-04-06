# Factor-W2025 

## Parameters Description and Example
    ticker = "QQQ"  # The stock or ETF ticker symbol (e.g., "QQQ" for the Nasdaq-100 ETF)
    start_date = "2006-01-01"  # Start date for the analysis (format: YYYY-MM-DD)
    end_date = "2023-12-31"  # End date for the analysis (format: YYYY-MM-DD)
    ff_data_path = "../data/F-F_Research_Data_Factors_Jer.CSV"  # Path to the Fama-French factors CSV file
    factors = ["Mkt-RF"]  # List of factors to include in the regression (e.g., "Mkt-RF", "SMB", "HML")