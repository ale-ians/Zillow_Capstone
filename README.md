Forecasting Colorado Home Values by Zip Code

1. Quick Results Summary
   - Overall Median Mean Absolute Error (MAE) across all ZIPS: $8,893
   - Best-performing ZIP: 81003 (MAE = $ 206)
   - Worst-performing ZIP: 81615 (MAE = $426,541)
   - Forecast Horizon: 6 Months
   - Visualization Highlights
     - Line Charts comparing actual vs predicted values
     - Heatmap of MAE by Zip for geographical insight
     
2. Project Overview
   -This project forecasts home values at the ZIP code level for the state of Colorado using historical Zillow Home Value
      index data. The model outputs predictions that can help real estate professionals, buyers, sellers, and investors
      make better informed decisions in a market where trends vary greatly by location.
   - Research Question:
     - "Can historical price trends be used to reliably forecast future home values at the ZIP code level in Colorado"
     
3. Data Sources
   -Primary Dataset [Zillow Research ZHVI Data](https://www.zillow.com/research/data/) -Single Family, All Homes, Smoothed
      & Seasonally Adjusted.
   -Geospacial Data: USPS ZIP centroid file for latitude/longitude mapping. 
   -Preprocessing:
        -Converted to long format (melted)
        -Filtered for Colorado ZIP Codes (prefix 80-81)
        -Removed null and invalid dates
        -Ensured ZIP Codes were stored as 5 digit strings

4. Methodology/Pipeline
    -The workflow follows a reproducible ML pipeline:
        1. Data ingestion
            Load raw Zillow CSV and USPS centroid file
        2. Data Cleaning
            Remove missing values, convert dates, filter for Colorado ZIP Codes
        3. Feature Engineering
            Create lag features (value_lag_1, value_lag_2, value_lag_3) for the time series modeling
        4. Model Training
            Train models for each zip separately:
            - RandomForestRegressor
            - XGBoost
        5. Evaluation
            Use the last 12 months as a test set and calculate Mean Absolute Error (MAE)
        6. Visualization
            - Plot actual vs. predicted values for individual ZIP Codes
            - Create MAE heatmaps by ZIP location
        7.Forecasting
            Generates forward-looking predictions for 6 months
 
5. Installation Requirements
     git clone https://github.com/ale-ians/Zillow_Capstone
     cd Zillow_Capstone
     pip install -r requirements.txt
   
    Python Version 3.10+
    Required Packages:
    - pandas
    - scikit-learn
    - matplotlib
    - xgboost
    - streamlit 

6. How to Run
   Launch Streamlit App
        streamlit run dashboard.py
        exit with ^c
    
    Run notebooks Capstone.ipynb

7. Results
    Example output for ZIP 80134:
    Mean Absolute Error (Test): $8,732
    Key Findings:
    Median MAE across all ZIPs: ~$12,000
    Highest MAE ZIP Codes appeared to be in smaller ZIP Codes to the West, in the mountains, or newer communities along the Front Range.
    Areas like North Field in Denver which has a large number of new builds over a smaller area.
    
8. Limitations and Future Work
    - Models were trained per ZIP; pooling data and creating areas like West Mountains, Front Range, and Eastern Plains
        may improve the results.
    - Access to greater detail data may help create a better all around tool. Using the Multiple Listing Service (MLS) to access home size,
        number of bedrooms, number of bathrooms could create a more accurate picture for individual homes.
    - Having more macroeconomic variables available would also help. Interest rates and population growth could give better insight.
    - Future work could explore:
      - Cross-validation for a more robust evaluation
      - Adding exogenous variables
      - Switching to a global time-series model like LightGBM with ZIP as a categorical feature
9. Repository Structure

    data/
        
    Generate_MAE_by_ZIP.py
    forecast_model.py
    notebooks/
        Capstone.ipynb
    dashboard.py
    requirements.txt
    README.md

10. References

    Zillow Research Data: https://www.zillow.com/research/data/
    USPS ZIP Centroid data: https://geodata.colorado.gov/datasets/fedmaps::census-zip-code-tabulation-areas/about
    