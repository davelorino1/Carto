#!/usr/bin/env python3

from google.cloud import bigquery
import pandas as pd 
import chime   
import seaborn as sns
import pymc as pm
import numpy as np
import arviz as az
import subprocess # uses the subprocess module to call the C++ 'global vs greedy' matching process 
import os
import matplotlib.pyplot as plt
client = bigquery.Client("gcp-wow-rwds-ai-checkout-dev")

# Bakery Campaign Digital Screen Assets 
check = client.query(
    """          
    CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1 AS 
        with booking_count AS (
            SELECT DISTINCT
                booking_number,
                line_name,
                CASE WHEN media_start_date = "2024-09-18" THEN 1 ELSE 2 END AS campaign_id_count
            FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns`
            WHERE quoteline_sku LIKE '%278258%' 
            AND media_type = 'Digital Screens Supers' 
            AND store_list IS NOT NULL 
        )
        SELECT 
            CONCAT(CONCAT(campaigns.booking_number, "_"), booking_count.campaign_id_count) AS booking_and_asset_number,
            campaigns.*
        FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns` campaigns
        LEFT JOIN booking_count 
            ON booking_count.booking_number = campaigns.booking_number 
            AND booking_count.line_name = campaigns.line_name
        WHERE quoteline_sku LIKE '%278258%' 
        AND media_type = 'Digital Screens Supers' 
        AND store_list IS NOT NULL   
        ;
    SELECT * FROM gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1
"""
).result()
check_df = check.to_dataframe()
check_df

# Campaign IDs to analyze
check = client.query(
    """
    SELECT DISTINCT booking_and_asset_number 
    FROM gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1
    WHERE booking_and_asset_number = "WOW20014359_2"
    """
).result()
campaign_ids_df = check.to_dataframe()

# Campaign Ids to list
campaign_ids_list = campaign_ids_df['booking_and_asset_number'].tolist()

# # Matching program executable - not running as test-control matching has already beeen done at the campaign setup level
# # cpp_executable = "./Match Maker/campaign_match_maker"

# # run the loop over each campaign to get the posterior distributions in sum_of_sales, convert to z_scores and then get
# # posterior distributions in percentage differences, regularized by: 
# #     - group posterior mean (not empirical mean) and, 
# #     - weight of evidence for each store pair
for campaign_id in campaign_ids_list:
    print(f"Processing campaign_id: {campaign_id}")
    check = client.query(
    f"""
        CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.unique_skus_2 AS (
            SELECT DISTINCT 
                booking_and_asset_number, 
                sku 
            FROM gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1, 
            UNNEST(SPLIT(quoteline_sku, ",")) AS sku 
            WHERE booking_and_asset_number = '{campaign_id}'
            AND sku IS NOT NULL
            AND LOWER(sku) <> "npd"
            AND sku <> ""
        );

        CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.test_stores_2 AS (
            SELECT DISTINCT 
                booking_number, 
                booking_and_asset_number,
                test_store 
            FROM gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1, 
            UNNEST(SPLIT(store_list, ",")) AS test_store 
            WHERE booking_and_asset_number = '{campaign_id}'
            AND test_store IS NOT NULL
        );
    """
    ).result()

    check_df = check.to_dataframe()

    ## Adding calculation of baseline_statistics_with_campaign for these campaigns
    check = client.query(
        f""" 
        DECLARE media_start_date_global_var DATE;
        DECLARE media_end_date_global_var DATE;
        SET media_start_date_global_var = (SELECT DISTINCT media_start_date FROM gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1 WHERE booking_and_asset_number = '{campaign_id}');
        SET media_end_date_global_var = (SELECT DISTINCT media_end_date FROM gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1 WHERE booking_and_asset_number = '{campaign_id}');
        CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_macro_bakery_live_test_1 AS 
            with step_one AS (
                SELECT 
                    ass_pre_campaign.Article, 
                    MIN(TXNStartDate) AS earliest_date, 
                    MAX(TXNStartDate) AS latest_date, 
                    trading.booking_and_asset_number AS campaign_id,
                    trading.media_start_date,
                    trading.media_end_date,
                    ass_pre_campaign.Site, 
                    DATE_TRUNC(ass_pre_campaign.TXNStartDate, WEEK(WEDNESDAY)) AS sales_week, 
                    SUM(ass_pre_campaign.TotalAmountIncldTax) AS total_sales_amount

                FROM `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` ass_pre_campaign
                
                LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1 trading
                    ON ass_pre_campaign.TXNStartDate >= DATE_ADD(trading.media_start_date, INTERVAL -12 WEEK)
                    AND ass_pre_campaign.TXNStartDate <= DATE_ADD(trading.media_end_date, INTERVAL -1 WEEK)
                
                INNER JOIN gcp-wow-cart-data-dev-d4d7.davide.unique_skus_2 skus 
                    ON TRIM(skus.sku) = TRIM(ass_pre_campaign.Article)
                    
                WHERE trading.booking_and_asset_number = '{campaign_id}' 
                AND LOWER(ass_pre_campaign.SalesChannelDescription) <> "online"
                AND ass_pre_campaign.TXNStartDate >= DATE_ADD(media_start_date_global_var, INTERVAL -12 WEEK)
                AND ass_pre_campaign.TXNStartDate <= DATE_ADD(media_end_date_global_var, INTERVAL -1 WEEK)
                AND ass_pre_campaign.SalesOrg = 1005
                GROUP BY ALL
            )  
            SELECT 
                campaign_id, 
                Site, 
                COUNT(DISTINCT sales_week) AS weeks_count,
                AVG(total_sales_amount) AS weekly_avg_sales_amount, 
                STDDEV(total_sales_amount) AS stddev_sales_amount 
            FROM step_one 
            GROUP BY ALL
        ; 
        SELECT * FROM gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_macro_bakery_live_test_1;
        """
    ).result()
    check_df = check.to_dataframe()
    print(check_df)

    ## Percentage of Stores Having Fewer than 13 weeks historical sales data pre-campaign
    check = client.query(
        f""" 
        with step_one AS (
            SELECT 
                baseline.*,
                CASE WHEN test.test_store IS NULL THEN "Control" ELSE "Test" END AS test_or_control
            FROM gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_macro_bakery_live_test_1 baseline
            LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.test_stores_2 test
                ON TRIM(test.booking_and_asset_number) = TRIM(baseline.campaign_id)
                AND TRIM(CAST(test.test_store AS STRING)) = TRIM(CAST(baseline.Site AS STRING))
            WHERE baseline.campaign_id = '{campaign_id}'
        ),
        step_two AS (
            SELECT 
                test_or_control,
                COUNT(DISTINCT Site) AS n_stores, 
                COUNT(DISTINCT CASE WHEN weeks_count = 12 THEN Site ELSE NULL END) AS has_12_wks_historical_sales
            FROM step_one 
            GROUP BY 1
        )

        SELECT 
            test_or_control, 
            SAFE_DIVIDE(CAST(has_12_wks_historical_sales AS FLOAT64) , CAST(n_stores AS FLOAT64)) AS perc_of_stores_with_12_wks_historical_sales 
        FROM step_two
        """
    ).result()

    check_df = check.to_dataframe()

    # Check if any group has less than 90% stores with 13 weeks of historical sales data
    if any(check_df['perc_of_stores_with_12_wks_historical_sales'] < 0.9):
        print(f"Skipping campaign_id {campaign_id} as less than 90% of stores in either group have 12 weeks of historical sales data.")
        continue

    print(f"Processing campaign period transactions: {campaign_id}")
    check = client.query(
        f""" 
        DECLARE media_start_date_global_var DATE; 
        DECLARE media_end_date_global_var DATE; 
        SET media_start_date_global_var = (SELECT DISTINCT media_start_date FROM gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1 WHERE booking_and_asset_number = '{campaign_id}');
        SET media_end_date_global_var = (SELECT DISTINCT media_end_date FROM gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1 WHERE booking_and_asset_number = '{campaign_id}');
        
        CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.regularization_test_campaign_period_transactions AS 
            with step_one AS (
                SELECT DISTINCT
                    trading.booking_and_asset_number AS campaign_id,
                    trading.media_start_date,
                    trading.media_end_date,
                    ass_campaign_period.Site, 
                    CASE WHEN test_stores.test_store IS NOT NULL THEN "Test" ELSE "Control" END AS test_or_control, 
                    --ass_campaign_period.Article,
                    ass_campaign_period.TXNStartDate,
                    ass_campaign_period.BasketKey,
                    TotalAmountIncldTax

                FROM `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` ass_campaign_period
                LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.macro_bakery_live_test_1 trading
                    ON ass_campaign_period.TXNStartDate >= trading.media_start_date 
                    AND ass_campaign_period.TXNStartDate <= trading.media_end_date 
                INNER JOIN gcp-wow-cart-data-dev-d4d7.davide.unique_skus_2 skus 
                    ON skus.sku = ass_campaign_period.Article 
                LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.test_stores_2 test_stores 
                    ON CAST(test_stores.test_store AS INT64) = CAST(ass_campaign_period.Site AS INT64)
                    
                WHERE trading.booking_and_asset_number = '{campaign_id}' 
                AND LOWER(ass_campaign_period.SalesChannelDescription) <> "online"
                AND ass_campaign_period.TXNStartDate >= media_start_date_global_var
                AND ass_campaign_period.TXNStartDate <= media_end_date_global_var
                AND ass_campaign_period.SalesOrg = 1005
            ) 
            SELECT *, SUM(TotalAmountIncldTax) AS sales_amount FROM step_one GROUP BY ALL
        ;
        
            

        SELECT * FROM gcp-wow-cart-data-dev-d4d7.davide.regularization_test_campaign_period_transactions;
      """
    ).result()
    transactions_df = check.to_dataframe()
    chime.success() 

    
    print(f"Processing baseline historical data: {campaign_id}")
    check = client.query(
    f"""
        with step_one AS (
        SELECT baseline.* , campaign.test_or_control
        FROM gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_macro_bakery_live_test_1 baseline
        LEFT JOIN (SELECT DISTINCT campaign_id, Site, test_or_control FROM gcp-wow-cart-data-dev-d4d7.davide.regularization_test_campaign_period_transactions) campaign
            ON baseline.campaign_id = campaign.campaign_id 
            AND baseline.Site = campaign.Site
        WHERE baseline.campaign_id = '{campaign_id}'
        AND weeks_count = 12
        ) 
        SELECT 
            test.campaign_id AS study_id, 
            test.Site AS test_store, 
            control.Site AS control_store, 
            ABS(test.weekly_avg_sales_amount / control.weekly_avg_sales_amount - 1) + ABS(test.stddev_sales_amount / control.stddev_sales_amount - 1) AS abs_perc_diff
        FROM step_one test
        LEFT JOIN step_one control 
            ON test.Site <> control.Site 
        WHERE test.test_or_control = "Test" 
        AND control.test_or_control = "Control" 
    """
    ).result()
    historical_performance_df = check.to_dataframe()
    #input_csv = f"./Match Maker/inputs/{campaign_id}_Store_Pairs.csv"
    #historical_performance_df.to_csv(input_csv, index = False)
    chime.success() 
    
# #     #print(f"Creating matched pairs: {campaign_id}")
# #     # Run the C++ program with the campaign_id
# #     #subprocess.run([cpp_executable, input_csv, campaign_id], check=True)
    
# #     # Check for the output file from the C++ program
# #     #output_file = f"./Match Maker/outputs/{campaign_id}_global_matching.csv"
# #     #if os.path.exists(output_file):
# #     #    print(f"Output file for {campaign_id} exists.")
# #         # Proceed with processing the output file
# #     #    output_df = pd.read_csv(output_file)
# #     #else:
# #     #    print(f"Output file for {campaign_id} does not exist.")

    check = client.query(
    """      
        SELECT DISTINCT CAST(Site AS INT64) Site, test_or_control
        FROM gcp-wow-cart-data-dev-d4d7.davide.regularization_test_campaign_period_transactions 
        ;
    """
    ).result()
    check_df = check.to_dataframe()
    check_df['Site'] = check_df['Site'].astype(int)
    matched_pairs = pd.read_csv(f"./Match Maker/outputs/{campaign_id}_global_matching.csv")

# #     # Filter check_df to get stores that are actually test stores
    test_stores = check_df[check_df['test_or_control'] == 'Test']

# #     # Identify the control stores in matched_pairs that ended up being test stores
    invalid_control_stores = matched_pairs[matched_pairs['control_store'].isin(test_stores['Site'])]
    print("Invalid control stores:\n")
    print(invalid_control_stores)
# #     # Now remove these invalid control stores from matched_pairs
    matched_pairs = matched_pairs[~matched_pairs['control_store'].isin(invalid_control_stores['control_store'])]
# #     # matched_pairs = matched_pairs[matched_pairs['abs_perc_diff'] <= 0.05]

# #     # Combine 'test_store' and 'control_store' into a single series
    combined_stores = pd.concat([matched_pairs['test_store'], matched_pairs['control_store']])

# #     # Filter transactions_df to only those where 'Site' (store number) is in the test or control stores
    filtered_transactions = transactions_df[transactions_df['Site'].isin(combined_stores)]

    print(f"Processing posterior distributions of sum_of_sales: {campaign_id}")

#     ## Posterior Distributions for total sum of sales 
    df = filtered_transactions

#     # Convert sales_amount to numeric, forcing any errors to NaN and dropping them
    df['sales_amount'] = pd.to_numeric(df['sales_amount'], errors='coerce')
    df.dropna(subset=['sales_amount'], inplace=True)

#     # Group transactions by store (Site) for posterior modeling
    grouped = df.groupby(['Site', 'test_or_control'])

#     # Initialize an empty dictionary to store full posterior samples for each store
    posterior_samples = {}
    count = 0

#     # Iterate through each group (store-level) and model the sum of sales as a posterior distribution
    for store, group in grouped:
        sales = group['sales_amount'].values  # extract individual transaction sales for the store
        count = count + 1

        # Check if sales is empty or non-numeric (additional guard)
        if len(sales) == 0 or not np.issubdtype(sales.dtype, np.number):
            
            print(f"Skipping store {store} due to invalid sales data")
            continue

        # Fitting the model for the posterior distribution of the sum of sales
        with pm.Model() as model:

            # Total number of transactions
            n_sales = len(sales)

            # Flat prior for the mean of the total sales across all transactions (meaning no prior belief that any value on the number line is more likely than any other value)
            total_sum_sales = pm.Uniform("total_sum_sales", lower=n_sales * sales.min(), upper=n_sales * sales.max())

            # Per-transaction mean derived from the total sum
            per_transaction_mu = total_sum_sales / n_sales
            
            # Flat prior for the standard deviation of the sales per transaction (meaning no prior belief that any value on the number line is more likely than any other value)
            sigma = pm.Uniform("sigma", lower=0, upper=sales.std() * 2)
            
            # Likelihood of observing sales per transaction
            sales_obs = pm.Normal("sales_obs", mu=per_transaction_mu, sigma=sigma, observed=sales)
            
            # Sampling from the posterior with increased tuning and sample size
            trace = pm.sample(2000, tune=2000, target_accept=0.95, return_inferencedata=True, progressbar=False)

            # Store the full posterior samples for the total sum of sales
            posterior_samples[store] = trace.posterior['total_sum_sales'].to_dataframe()
            print(f"{count} stores fit with posteriors\n")

    # Convert the posterior samples into a dictionary for each store
    # The key will be the store and the value will be a DataFrame of posterior samples
    posterior_samples_dict = {store: df.reset_index(drop=True) for store, df in posterior_samples.items()}

#     # Example output for a specific store (the first one in the dictionary)
    store_name = list(posterior_samples_dict.keys())[0]
    chime.success() 
    print(f"Posterior samples for store {store_name}:\n", posterior_samples_dict[store_name].head())

    # Flatten the posterior_samples_dict into a DataFrame
    # Stacking the store IDs and their corresponding posterior samples

    flattened_samples = []
    for store, posterior_df in posterior_samples.items():
        store_id = store[0]  # Extract the store ID
        test_or_control = store[1]  # Extract test or control group info
        posterior_df = posterior_df.reset_index(drop=True)  # Reset the index of the posterior samples
        posterior_df['store'] = store_id  # Add the store ID as a column
        posterior_df['test_or_control'] = test_or_control  # Add the test or control info
        flattened_samples.append(posterior_df)

    # Concatenate all store samples into a single DataFrame
    flattened_samples_df = pd.concat(flattened_samples, ignore_index=True)

    # Merging posterior samples of sum_of_sales with historical performance in order to produce z-score distributions
    # flattened_samples_df: Contains store, test_or_control, total_sum_sales
    # campaign_data_df: Contains campaign_id, Site (store), stddev_sales_amount, weekly_avg_sales_amt
    check = client.query(
    f"""
        SELECT 
            baseline.*, 
            campaign.test_or_control 
        FROM gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_macro_bakery_live_test_1 baseline
        LEFT JOIN (SELECT DISTINCT campaign_id, Site, test_or_control FROM gcp-wow-cart-data-dev-d4d7.davide.regularization_test_campaign_period_transactions) campaign
            ON baseline.campaign_id = campaign.campaign_id 
            AND baseline.Site = campaign.Site
        WHERE baseline.campaign_id = '{campaign_id}'
        AND weeks_count = 12
    """
    ).result()
    campaign_data_df = check.to_dataframe()

    # Perform a left join
    merged_df = pd.merge(flattened_samples_df, 
                        campaign_data_df[['campaign_id', 'Site', 'stddev_sales_amount', 'weekly_avg_sales_amount']],
                        left_on='store', right_on='Site', how='inner')

    # Drop the redundant 'Site' column (if needed)
    merged_df = merged_df.drop(columns=['Site'])

    # Output the merged DataFrame
    print(merged_df.head(5))

    print(f"Processing z-scores from posterior distribution of sum_of_sales: {campaign_id}")
    # Convert 'total_sum_sales', 'weekly_avg_sales_amt', and 'stddev_sales_amount' to float
    merged_df['total_sum_sales'] = merged_df['total_sum_sales'].astype(float)
    merged_df['weekly_avg_sales_amount'] = merged_df['weekly_avg_sales_amount'].astype(float)
    merged_df['stddev_sales_amount'] = merged_df['stddev_sales_amount'].astype(float)

    # Ensure there are no zero values in stddev_sales_amt to avoid division errors
    merged_df['stddev_sales_amount'].replace(0, np.nan, inplace=True)

    # Calculate the campaign_z_score and create a new column
    merged_df['campaign_z_score'] = (merged_df['total_sum_sales'] - merged_df['weekly_avg_sales_amount']) / merged_df['stddev_sales_amount']

    # Compute 'percentage_change' as the relative change from the baseline weekly average sales amount
    merged_df['percentage_change'] = (merged_df['campaign_z_score'] * merged_df['stddev_sales_amount']) / merged_df['weekly_avg_sales_amount']
    merged_df.to_csv(f"./Match Maker/outputs/{campaign_id}_merged_df.csv")
#     # Output the resulting DataFrame with the new column
    print(merged_df.head(5))
    chime.success() 

    # 'merged_df' contains columns: 'store', 'test_or_control', 'campaign_z_score'
    # 'matched_pairs' contains columns: 'test_store', 'control_store'

    # merged_df = pd.read_csv(f"./Match Maker/outputs/{campaign_id}_merged_df.csv")
    # Ensure 'merged_df' is sorted by 'store' and 'test_or_control'
    merged_df = merged_df.sort_values(by=['store', 'test_or_control'])

    def run_model(sigma_value, merged_df, matched_pairs):
    
        # Initialize a list to store the differences in percentage changes
        percentage_change_differences = []

        print(f"Processing differences in percentage changes: {campaign_id}")
        # Iterate through each pair of test and control stores in 'matched_pairs'
        for _, pair in matched_pairs.iterrows():
            test_store = pair['test_store']
            control_store = pair['control_store']
            
            # Extract percentage_change samples for test and control stores
            test_store_pc = merged_df[merged_df['store'] == test_store]['percentage_change'].values
            control_store_pc = merged_df[merged_df['store'] == control_store]['percentage_change'].values
            
            test_store_z_score = merged_df[merged_df['store'] == test_store]['campaign_z_score'].values
            control_store_z_score = merged_df[merged_df['store'] == control_store]['campaign_z_score'].values
            z_score_perc_diff = (test_store_z_score - control_store_z_score) / np.abs(control_store_z_score)
            test_store_stddev = merged_df[merged_df['store'] == test_store]['stddev_sales_amount'].values
            test_store_weekly_avg_sales = merged_df[merged_df['store'] == test_store]['weekly_avg_sales_amount'].values

            z_score_delta = test_store_z_score - control_store_z_score
            z_score_delta_times_stddev = z_score_delta * test_store_stddev
            test_store_performance_relative_to_control_store = z_score_delta_times_stddev / test_store_weekly_avg_sales
            
            # Compute the difference in percentage changes
            pc_diff = test_store_pc - control_store_pc
            
            # Store the differences along with the test and control store ids
            percentage_change_differences.append(pd.DataFrame({
                'test_store': test_store,
                'control_store': control_store,
                'z_score_perc_diff': z_score_perc_diff
            }))

        # Concatenate all the difference DataFrames
        pc_differences_df = pd.concat(percentage_change_differences, ignore_index=True)

        # Create an index that identifies which pair each difference belongs to
        pair_labels = pc_differences_df.apply(lambda row: (row['test_store'], row['control_store']), axis=1)

        # Create a categorical variable to get codes for each unique pair
        pair_categories = pd.Categorical(pair_labels)
        pair_indices = pair_categories.codes  # This will map each observation to a store pair index
        unique_pairs = pair_categories.categories

        with pm.Model() as model:
            # Prior for the global mean difference in percentage changes (mu_pop)
            #mu_pop = pm.HalfNormal('mu_pop', sigma=0.05)  # Adjust sigma based on prior belief

            # Skewed normal prior - low probability of negative effects
            # mu_pop = pm.SkewNormal('mu_pop', mu=0, sigma=0.005, alpha=0.5)

            # Non skewed normal prior 
            mu_pop = pm.Normal('mu_pop', mu=0, sigma=0.0025)

            # Prior for the population standard deviation (sigma_pop)
            sigma_pop = pm.HalfNormal('sigma_pop', sigma=sigma_value)

            # Store-pair-specific effects
            store_pair_effects = pm.Normal('store_pair_effects', mu=mu_pop, sigma=sigma_pop, shape=len(unique_pairs))
            
            # Assign the correct store_pair_effect to each percentage difference using pair_indices
            store_pair_effects_repeated = store_pair_effects[pair_indices]
            
            # Prior for the observational standard deviation (sigma_obs)
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.005)
            
            # Likelihood: Using normal distribution
            observed_pc_diff = pm.Normal(
                'observed_pc_diff',
                mu=store_pair_effects_repeated,
                sigma=sigma_obs,
                observed=pc_differences_df['z_score_perc_diff'].values
            )
            
            # Sample from the posterior distribution
            trace = pm.sample(
                draws=2000,
                tune=2000,
                target_accept=0.95,
                return_inferencedata=True
            )
        
        # Summarize and analyze the trace
        summary_df = az.summary(
            trace,
            var_names=['mu_pop', 'sigma_pop', 'sigma_obs', 'store_pair_effects'],
            hdi_prob=0.95
        )

        return trace, summary_df, unique_pairs


    # def evaluate_model(summary_df):
    #     population_mean = summary_df.loc['mu_pop', 'mean']
    #     group_estimates = summary_df.loc[summary_df.index.str.contains('store_pair_effects'), 'mean']
        
    #     # Calculate the standard deviation of group estimates
    #     group_std = np.std(group_estimates)
        
    #     # Calculate the percentage of group estimates within ±1 standard deviation of the population mean
    #     within_range = np.abs(group_estimates - population_mean) <= group_std
    #     percentage_within_range = np.mean(within_range) * 100
        
    #     # Calculate the percentage of estimates very close to the population mean (potential underfitting)
    #     percentage_close = np.mean(np.abs(group_estimates - population_mean) < 0.1 * group_std) * 100
        
    #     return percentage_within_range, percentage_close

    def evaluate_model(summary_df):
        population_mean = summary_df.loc['mu_pop', 'mean']
        group_estimates = summary_df.loc[summary_df.index.str.contains('store_'), 'mean']
        
        # Calculate the percentage of group estimates within ±0.2 of the population mean
        within_range = np.abs(group_estimates - population_mean) <= 0.3
        percentage_within_range = np.mean(within_range) * 100
        
        # Calculate the percentage of estimates exactly equal to the population mean (underfitting)
        percentage_exact = np.mean(np.abs(group_estimates - population_mean) < 1e-6) * 100
        
        return percentage_within_range, percentage_exact

    
    def optimize_sigma(merged_df, matched_pairs, initial_sigma=0.0005, initial_step=0.0005, max_iterations=80):
        current_sigma = round(initial_sigma, 5)
        step_size = round(initial_step, 5)
        results = []
        tested_sigmas = set()
        previous_direction = None
        optimal_trace = None
        optimal_summary = None
        unique_pairs = None

        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            print(f"Testing sigma: {current_sigma:.5f}")

            if current_sigma in tested_sigmas:
                print("Arrived at a previously tested sigma. Taking a half step in the opposite direction.")
                step_size = round(step_size / 2, 5)
                if previous_direction == 'loosen':
                    current_sigma = round(current_sigma - step_size, 5)
                else:
                    current_sigma = round(current_sigma + step_size, 5)
                print(f"New sigma: {current_sigma:.5f}, New step size: {step_size:.5f}")
                continue

            tested_sigmas.add(current_sigma)

            trace, summary_df, unique_pairs = run_model(current_sigma, merged_df, matched_pairs)
            #percentage_within_range, percentage_close = evaluate_model(summary_df)  # Corrected variable names
            percentage_within_range, percentage_exact = evaluate_model(summary_df)  # Corrected variable names
            print(summary_df)
            results.append({
                'iteration': iteration + 1,
                'sigma': current_sigma,
                'percentage_within_range': percentage_within_range,
                'percentage_exact': percentage_exact  # Updated variable name
            })

            print(f"Percentage within range: {percentage_within_range:.2f}%")
            print(f"Percentage exact to mean: {percentage_exact:.2f}%")  # Updated print statement

            if percentage_within_range >= 46 and percentage_exact < 5:  # Adjusted criteria
                print("Acceptance criteria met. Stopping optimization.")
                optimal_trace = trace
                optimal_summary = summary_df
                break

            is_underfitting = percentage_exact >= 5  # Updated variable name
            is_overfitting = percentage_within_range < 46

            if is_underfitting and is_overfitting:
                print("Both underfitting and overfitting detected. This is an unexpected state.")
                break

            current_direction = 'loosen' if is_underfitting else 'tighten'

            if previous_direction is not None and ((current_direction != previous_direction) or (current_direction == "tighten" and current_sigma <= 0.0005)):
                step_size = round(step_size / 2, 5)
                print(f"Direction changed. New step size: {step_size:.5f}")

            if (current_direction == "tighten" and current_sigma <= 0.0005):
                step_size = round(step_size / 2, 5)
                print(f"Decreasing from .0005% - new step size: {step_size:.5f}")

            if is_underfitting:
                current_sigma = round(current_sigma + step_size, 5)
                print(f"Underfitting. Increasing sigma to {current_sigma:.5f} for next iteration.")
                previous_direction = 'loosen'
            else:
                current_sigma = round(max(0, current_sigma - step_size), 5)
                print(f"Overfitting. Decreasing sigma to {current_sigma:.5f} for next iteration.")
                previous_direction = 'tighten'

            if current_sigma <= 0:
                print("Sigma reached zero or became negative. Stopping optimization.")
                break

        results_df = pd.DataFrame(results)

        if len(results_df) == max_iterations:
            print("Reached maximum iterations without finding optimal sigma.")
        elif optimal_trace is None:
            print("No optimal sigma found.")
        else:
            optimal_sigma = results_df.loc[results_df['iteration'].idxmax(), 'sigma']
            print(f"\nOptimization complete.")
            print(f"Optimal sigma: {optimal_sigma:.5f}")
            print(f"Final percentage within range: {results_df['percentage_within_range'].iloc[-1]:.2f}%")
            print(f"Final percentage close to mean: {results_df['percentage_exact'].iloc[-1]:.2f}%")  # Updated print statement

        return results_df, optimal_trace, optimal_summary, unique_pairs


    # Usage example (assuming merged_df and matched_pairs are already prepared)
    results_df, optimal_trace, optimal_summary, unique_pairs = optimize_sigma(merged_df, matched_pairs)

    # If you want to display the final results:
    if optimal_summary is not None:
        print("\nFinal Model Summary:")
        print(optimal_summary[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']])

    # Plot the optimization results
    def plot_optimization_results(results_df):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.plot(results_df['iteration'], results_df['sigma'])
        plt.xlabel('Iteration')
        plt.ylabel('Sigma')
        plt.title('Sigma Value Over Iterations')

        plt.subplot(2, 2, 3)
        plt.plot(results_df['iteration'], results_df['percentage_within_range'])
        plt.xlabel('Iteration')
        plt.ylabel('Percentage Within Range')
        plt.title('Percentage Within ±0.05 of Population Mean')

        plt.subplot(2, 2, 4)
        plt.plot(results_df['iteration'], results_df['percentage_exact'])
        plt.xlabel('Iteration')
        plt.ylabel('Percentage Exact Match')
        plt.title('Percentage Exactly Equal to Population Mean')

        plt.tight_layout()
        plt.show()

    # Plot results
    #plot_optimization_results(results_df)

    # Map store pairs to store_pair_effects for easier interpretation
    store_pair_mapping = {f"store_pair_effects[{i}]": unique_pairs[i] for i in range(len(unique_pairs))}

    summary_df = optimal_summary
    # Replace the index in summary_df with the store pair labels
    summary_df = summary_df.rename(index=store_pair_mapping)

    # Convert 'mu_pop' and 'store_pair_effects' to percentages
    summary_df[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']] *= 100

    # Since store pair labels are tuples, format them as strings for readability
    summary_df.index = summary_df.index.map(lambda x: f"{int(x[0])}_{int(x[1])}" if isinstance(x, tuple) else x)

    # Multiply store_pair_effects by 100 to get percentages
    store_pair_indices = [i for i in summary_df.index if i not in ['mu_pop', 'sigma_pop', 'sigma_obs']]
    #summary_df.loc[store_pair_indices, ['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']] *= 100

    # Output the summary DataFrame with relevant columns
    print(summary_df[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']])

    # Convert the index into a DataFrame for easier handling
    summary_df = summary_df.reset_index()

    # Separate rows where the index is not a store pair (e.g., 'mu_pop', 'sigma_pop')
    non_store_rows = summary_df[summary_df['index'].isin(['mu_pop', 'sigma_pop', 'sigma_obs'])]

    # Filter out only the store pair rows for processing
    store_pair_rows = summary_df[~summary_df['index'].isin(['mu_pop', 'sigma_pop', 'sigma_obs'])]

    # Extract test_store and control_store from the store pair rows
    store_pair_rows[['test_store', 'control_store']] = store_pair_rows['index'].str.extract(r'(\d+\.?\d*)\_(\d+\.?\d*)')

    # Convert store IDs to integers
    store_pair_rows['test_store'] = store_pair_rows['test_store'].astype(float).astype(int)
    store_pair_rows['control_store'] = store_pair_rows['control_store'].astype(float).astype(int)

    # Drop the original 'index' column from store pair rows
    store_pair_rows.drop(columns=['index'], inplace=True)

    # Combine the population-level rows back with the processed store pair rows
    summary_df_processed = pd.concat([non_store_rows, store_pair_rows], ignore_index=True)

    # Display the processed DataFrame
    print(summary_df_processed.head(20))
    chime.success() 

    print(f"Adding empirical calculation of percentage difference in z-scores for comparison: {campaign_id}")
    check = client.query(
        f"""
        with step_one AS (
            SELECT 
                test.campaign_id, 
                test.Site AS test_store, 
                control.Site AS control_store, 
                test.total_sales_campaign_period AS test_campaign_sales,
                control.total_sales_campaign_period AS control_campaign_sales, 
                SAFE_DIVIDE(test.mean_sales_amount , test.weeks_count) AS test_weekly_avg_sales_amt, 
                SAFE_DIVIDE(control.mean_sales_amount , control.weeks_count) AS control_weekly_avg_sales_amt, 
                test.stddev_sales_amount AS test_stddev_sales_amt, 
                control.stddev_sales_amount AS control_stddev_sales_amt, 
                SAFE_DIVIDE((test.total_sales_campaign_period - SAFE_DIVIDE(test.mean_sales_amount , test.weeks_count)) , test.stddev_sales_amount) AS test_z_score, 
                SAFE_DIVIDE((control.total_sales_campaign_period - SAFE_DIVIDE(control.mean_sales_amount , control.weeks_count)) , control.stddev_sales_amount) AS control_z_score
            FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_pre_vs_during_period_plus_baseline_4 test 
            LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_pre_vs_during_period_plus_baseline_4 control
                ON test.Site <> control.Site 
            WHERE test.campaign_id = '{campaign_id}'
            AND control.campaign_id = '{campaign_id}'
            AND test.test_or_control = "Test" 
            AND control.test_or_control = "Control" 
            AND test.weeks_count = 13
            AND control.weeks_count = 13
        ) 
        SELECT 
            *, 
            CASE WHEN test_z_score < 0 OR control_z_score < 0 THEN ((SAFE_DIVIDE(test_z_score , control_z_score) - 1) * -1) * 100 ELSE (SAFE_DIVIDE(test_z_score , control_z_score) -1) * 100 END AS z_score_perc_diff_old,
            (test_z_score - control_z_score) / ABS(control_z_score) AS z_score_perc_diff_equivalent,
            (test_z_score - control_z_score) / (ABS(test_z_score) + ABS(control_z_score)) AS z_score_perc_diff_new 
        FROM step_one
    """
    ).result()
    check_df = check.to_dataframe()
    chime.success() 
    result_df = check_df.merge(matched_pairs, on=['test_store', 'control_store'], how='inner')
    print(result_df)

    print(f"Saving final results: {campaign_id}")
    # Ensure that 'test_store' and 'control_store' in merged_df are integers for joining
    result_df['test_store'] = result_df['test_store'].astype(int)
    result_df['control_store'] = result_df['control_store'].astype(int)
    result_df['z_score_perc_diff_equivalent'] = result_df['z_score_perc_diff_equivalent'] * 100
    pd.set_option('display.float_format', '{:,.2f}'.format)
    # Join the processed summary dataframe with the merged_df on test_store and control_store
    comparison_df = pd.merge(
        summary_df_processed[['test_store', 'control_store', 'mean']],
        result_df[['test_store', 'control_store', 'z_score_perc_diff_equivalent']],  # Keep the relevant columns from merged_df
        on=['test_store', 'control_store'],
        how='left'
    )

    # Rename the columns for clarity
    comparison_df = comparison_df.rename(columns={
        'mean': 'pymc_percentage_diff',
        'empirical_percentage_diff': 'empirical_percentage_diff'
    })

    # Display the merged dataframe for comparison
    comparison_df['pymc_percentage_diff'] = comparison_df['pymc_percentage_diff'].round(2)
    # Adding the campaign_id column to the DataFrame
    comparison_df.insert(0, 'campaign_id', campaign_id)

    # Display the updated DataFrame
    print(comparison_df)
    comparison_df.to_csv(f"~/Documents/Cartology/Incrementality/Regularization/outputs/{campaign_id}_comparison.csv", index=False)
    chime.success() 
    chime.success() 
    chime.success() 

#campaign_ids_df.to_csv(f"/Regularization/outputs/{campaign_id}_comparison.csv", index=False)

