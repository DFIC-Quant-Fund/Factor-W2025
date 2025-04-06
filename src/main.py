from fama_french_regression import FamaFrenchRegression

def main():
    ticker = "QQQ"  
    start_date = "2006-01-01"  
    end_date = "2023-12-31"  
    ff_data_path = "../data/F-F_Research_Data_Factors_Jer.CSV"
    factors = ["Mkt-RF"]  
    
    regression = FamaFrenchRegression(ticker, start_date, end_date, ff_data_path, factors)
    regression.run_fama_fetch()

if __name__ == "__main__":
    main()
