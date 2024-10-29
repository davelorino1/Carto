#!/usr/bin/env python3

# This script produces regularized estimates of test vs control z-score percentage differences for 
# individual cartology campaign assets

# TODO: 
    ## Add user-specific query execution logging
    ## Add hyperparameter tuning logging 
    ## Eventually remove gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2 
        ### currently it is used exactly 1 time at the start of the script to select a campaign to analyze
        ### this is just the campaign assets belonging to digital screens campaigns 
        ### using the predefined list of campaigns is just a time saving thing so we dont have to constantly 
        ### run the query that identifies the relevant assets 
        ### the roadmap here is to replace that with a query that pulls relevant assets according to which 
        ### live test is being analyzed - not sure yet whether implementation should work off a command line argument or something else 
    
    ## Transform z-score percentage diffs to the log scale for the sampling process
    ## Transform priors and hyperparameters to the log scale 
    ## Un-transform model results from log to natural scale 

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
import sys 

client = bigquery.Client("cart-dai-sandbox-nonprod-c3e7")
data_scientist = sys.argv[1]

# Match Maker Python Implementation (replaces the C++ version)
def store_matching(df):
    # Helper function for greedy matching
    def greedy_matching(store_pairs):
        matched_pairs = []
        used_test_stores = set()
        used_control_stores = set()

        for _, row in store_pairs.iterrows():
            if row['test_store'] not in used_test_stores and row['control_store'] not in used_control_stores:
                matched_pairs.append(row)
                used_test_stores.add(row['test_store'])
                used_control_stores.add(row['control_store'])

        return pd.DataFrame(matched_pairs)

    # Helper function for global matching
    def global_matching(store_pairs):
        store_pairs_sorted = store_pairs.sort_values('abs_perc_diff')
        return greedy_matching(store_pairs_sorted)

    # Calculate total difference
    def calculate_total_difference(matched_pairs):
        return matched_pairs['abs_perc_diff'].sum()

    # Perform greedy matching
    greedy_result = greedy_matching(df)
    greedy_total_diff = calculate_total_difference(greedy_result)

    # Perform global matching
    global_result = global_matching(df)
    global_total_diff = calculate_total_difference(global_result)

    # Print results
    print(f"Greedy Matching Total Difference: {greedy_total_diff}")
    print(f"Global Matching Total Difference: {global_total_diff}")

    # Return the global matching result as a DataFrame
    return global_result



# Campaign IDs to analyze
check = client.query(
    f"""
    SELECT DISTINCT booking_and_asset_number 
    # TODO cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info
    FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2 
    WHERE --booking_and_asset_number NOT IN ("WOW20001912_1", "WOW20000860_1", "WOW20001833_1", "WOW20000487_1")
    booking_and_asset_number = 'WOW20000860_1'
    """
).result()
campaign_ids_df = check.to_dataframe()
print("Campaign ids:")
print(campaign_ids_df)
# Campaign Ids to list
campaign_ids_list = campaign_ids_df['booking_and_asset_number'].tolist()
print(campaign_ids_list)

# run the loop over each campaign to get the posterior distributions in sum_of_sales, convert to z_scores and then get
# posterior distributions in percentage differences, regularized by: 
#     - group posterior mean (not empirical mean) and, 
#     - weight of evidence for each store pair
for campaign_id in campaign_ids_list:
    # print(f"Processing campaign_id: {campaign_id}")

    # # Bakery Campaign Digital Screen Assets 
    # check = client.query(
    #     f"""          
    #     CREATE OR REPLACE TABLE cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info AS 
    #         with booking_count AS (
    #             SELECT DISTINCT
    #                 booking_number,
    #                 line_name,
    #                 ROW_NUMBER() OVER(PARTITION BY booking_number ORDER BY media_start_date) AS asset_id_count
    #             FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns`
    #             WHERE booking_number = 'WOW20000860'  # Placeholder for '{campaign_id}'
    #             AND store_list IS NOT NULL 
    #         )
    #         SELECT 
    #             CONCAT(CONCAT(campaigns.booking_number, "_"), booking_count.asset_id_count) AS booking_and_asset_number,
    #             campaigns.*
    #         FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns` campaigns
    #         LEFT JOIN booking_count 
    #             ON booking_count.booking_number = campaigns.booking_number 
    #             AND booking_count.line_name = campaigns.line_name
    #         WHERE campaigns.booking_number = 'WOW20000860'  # Placeholder for '{campaign_id}'
    #         AND store_list IS NOT NULL   
    #         ;

    #     SELECT * FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info ORDER BY media_start_date;
    # """
    # ).result()
    # check_df = check.to_dataframe()
    # print("Campaign Asset Info: ")
    # print(check_df[['booking_number', 'booking_and_asset_number', 'line_name', 'media_start_date']])


    # check = client.query(
    # f"""
    #     CREATE OR REPLACE TABLE cart-dai-sandbox-nonprod-c3e7.{data_scientist}.unique_skus_2 AS (
    #         SELECT DISTINCT 
    #             booking_and_asset_number, 
    #             sku 
    #         # TODO  cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info        
    #         FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info,
    #         UNNEST(SPLIT(quoteline_sku, ",")) AS sku 
    #         WHERE booking_and_asset_number = '{campaign_id}'
    #         AND sku IS NOT NULL
    #         AND LOWER(sku) <> "npd"
    #         AND sku <> ""
    #     );

    #     CREATE OR REPLACE TABLE cart-dai-sandbox-nonprod-c3e7.{data_scientist}.test_stores_2 AS (
    #         SELECT DISTINCT 
    #             booking_number, 
    #             booking_and_asset_number,
    #             test_store 
    #         # TODO cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info
    #         FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info,
    #         UNNEST(SPLIT(store_list, ",")) AS test_store 
    #         WHERE booking_and_asset_number = '{campaign_id}'
    #         AND test_store IS NOT NULL
    #     );
    # """
    # ).result()

    # check_df = check.to_dataframe()

    # ## Adding calculation of baseline_statistics_with_campaign for these campaigns
    # check = client.query(
    #     f""" 
    #     DECLARE media_start_date_global_var DATE;
    #     DECLARE media_end_date_global_var DATE;
    #     SET media_start_date_global_var = (SELECT DISTINCT media_start_date FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info WHERE booking_and_asset_number = '{campaign_id}');
    #     SET media_end_date_global_var = (SELECT DISTINCT media_end_date FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info WHERE booking_and_asset_number = '{campaign_id}');
    #     CREATE OR REPLACE TABLE cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_baseline_statistics_with_campaign AS 
    #         with step_one AS (
    #             SELECT 
    #                 ass_pre_campaign.Article, 
    #                 MIN(BusinessDate) AS earliest_date, 
    #                 MAX(BusinessDate) AS latest_date, 
    #                 trading.booking_and_asset_number AS campaign_id,
    #                 trading.media_start_date,
    #                 trading.media_end_date,
    #                 ass_pre_campaign.SiteNumber AS Site, 
    #                 DATE_TRUNC(ass_pre_campaign.BusinessDate, WEEK(WEDNESDAY)) AS sales_week, 
    #                 SUM(ass_pre_campaign.TotalAmountIncludingGST) AS total_sales_amount,
    #                 COUNT(DISTINCT ass_pre_campaign.BasketKey) AS total_baskets

    #             FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` ass_pre_campaign
                
    #             # TODO cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info
    #             LEFT JOIN cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info trading
    #                 ON ass_pre_campaign.BusinessDate >= DATE_ADD(trading.media_start_date, INTERVAL -12 WEEK)
    #                 AND ass_pre_campaign.BusinessDate <= DATE_ADD(trading.media_end_date, INTERVAL -1 WEEK)
                
    #             INNER JOIN cart-dai-sandbox-nonprod-c3e7.{data_scientist}.unique_skus_2 skus 
    #                 ON TRIM(skus.sku) = TRIM(ass_pre_campaign.Article)
                    
    #             WHERE trading.booking_and_asset_number = '{campaign_id}' 
    #             AND LOWER(ass_pre_campaign.Channel) = "in store"
    #             AND ass_pre_campaign.BusinessDate >= DATE_ADD(media_start_date_global_var, INTERVAL -12 WEEK)
    #             AND ass_pre_campaign.BusinessDate <= DATE_ADD(media_end_date_global_var, INTERVAL -1 WEEK)
    #             AND ass_pre_campaign.SalesOrganisation = '1005'
    #             GROUP BY ALL
    #         )  
    #         SELECT 
    #             campaign_id, 
    #             Site, 
    #             COUNT(DISTINCT sales_week) AS weeks_count,
    #             AVG(total_sales_amount) AS weekly_avg_sales_amount, 
    #             STDDEV(total_sales_amount) AS stddev_sales_amount 
    #         FROM step_one 
    #         GROUP BY ALL
    #     ; 
    #     SELECT campaign_id, COUNT(DISTINCT CASE WHEN weeks_count = 12 THEN Site ELSE NULL END) / COUNT(DISTINCT Site) AS perc_of_stores_with_12_wks_historical_sales
    #     FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_baseline_statistics_with_campaign
    #     GROUP BY 1
    #     ;
    #     """
    # ).result()
    # check_df = check.to_dataframe()
    # print(check_df)

    # # Check if any group has less than 90% stores with 12 weeks of historical sales data
    # if any(check_df['perc_of_stores_with_12_wks_historical_sales'] < 0.9):
    #     print(f"Skipping campaign_id {campaign_id} as less than 90% of stores in either group have 13 weeks of historical sales data.")
    #     continue

    # print(f"Processing campaign period transactions: {campaign_id}")
    # check = client.query(
    #     f""" 
    #     DECLARE media_start_date_global_var DATE; 
    #     DECLARE media_end_date_global_var DATE; 
    #     SET media_start_date_global_var = (SELECT DISTINCT media_start_date FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info WHERE booking_and_asset_number = '{campaign_id}');
    #     SET media_end_date_global_var = (SELECT DISTINCT media_end_date FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info WHERE booking_and_asset_number = '{campaign_id}');

    #     CREATE OR REPLACE TABLE cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_period_transactions AS 
    #         SELECT 
    #             trading.booking_and_asset_number AS campaign_id,
    #             trading.media_start_date,
    #             trading.media_end_date,
    #             ass_campaign_period.SiteNumber AS Site, 
    #             CASE WHEN test_stores.test_store IS NOT NULL THEN "Test" ELSE "Control" END AS test_or_control, 
    #             --ass_campaign_period.Article,
    #             ass_campaign_period.BusinessDate,
    #             ass_campaign_period.BasketKey,
    #             SUM(ass_campaign_period.TotalAmountIncludingGST) AS sales_amount

    #         FROM `gcp-wow-wiq-ca-prod.wiqIN_DataAssets.CustomerBaseTransaction_v` ass_campaign_period
    #         LEFT JOIN cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_asset_info trading
    #             ON ass_campaign_period.BusinessDate >= trading.media_start_date 
    #             AND ass_campaign_period.BusinessDate <= trading.media_end_date 
    #         INNER JOIN cart-dai-sandbox-nonprod-c3e7.{data_scientist}.unique_skus_2 skus 
    #             ON TRIM(skus.sku) = ass_campaign_period.Article 
    #         LEFT JOIN cart-dai-sandbox-nonprod-c3e7.{data_scientist}.test_stores_2 test_stores 
    #             ON CAST(test_stores.test_store AS INT64) = CAST(ass_campaign_period.SiteNumber AS INT64)
                
    #         WHERE trading.booking_and_asset_number = '{campaign_id}' 
    #         AND LOWER(ass_campaign_period.Channel) = "in store"
    #         AND ass_campaign_period.BusinessDate >= media_start_date_global_var
    #         AND ass_campaign_period.BusinessDate <= media_end_date_global_var
    #         AND ass_campaign_period.SalesOrganisation = '1005'
    #         GROUP BY ALL
    #     ;

    #     SELECT * FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_period_transactions;
    # """
    # ).result()
    # transactions_df = check.to_dataframe()
    # chime.success() 

    # print(f"Processing baseline historical data: {campaign_id}")
    # check = client.query(
    # f"""
    #     with step_one AS (
    #     SELECT baseline.*, campaign.test_or_control 
    #     FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_baseline_statistics_with_campaign baseline
    #     LEFT JOIN (SELECT DISTINCT campaign_id, Site, test_or_control FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_period_transactions) campaign
    #         ON baseline.campaign_id = campaign.campaign_id 
    #         AND baseline.Site = campaign.Site
    #     WHERE baseline.campaign_id = '{campaign_id}'
    #     AND weeks_count = 12
    #     ) 
    #     SELECT 
    #         test.campaign_id AS study_id, 
    #         test.Site AS test_store, 
    #         control.Site AS control_store, 
    #         ABS(test.weekly_avg_sales_amount / control.weekly_avg_sales_amount - 1) + ABS(test.stddev_sales_amount / control.stddev_sales_amount - 1) AS abs_perc_diff
    #     FROM step_one test
    #     LEFT JOIN step_one control 
    #         ON test.Site <> control.Site 
    #     WHERE test.test_or_control = "Test" 
    #     AND control.test_or_control = "Control" 
    # """
    # ).result()
    # historical_performance_df = check.to_dataframe()
    
    # chime.success() 
    
    # print(f"Creating matched pairs: {campaign_id}")
    # matched_pairs = store_matching(historical_performance_df)
    # print("Matched Pairs: ")
    # print(matched_pairs)
    # # Combine 'test_store' and 'control_store' into a single series
    # combined_stores = pd.concat([matched_pairs['test_store'], matched_pairs['control_store']])
    # print("All Stores ")
    # print(combined_stores)
    # # Filter transactions_df to only those where 'Site' (store number) is in the test or control stores
    # filtered_transactions = transactions_df[transactions_df['Site'].isin(combined_stores)]
    # print("Filtered Tranactions: ")
    # print(filtered_transactions)
    # print(f"Processing posterior distributions of sum_of_sales: {campaign_id}")

    # ## Begin the process of calculating posterior distributions for total sum of sales in each store
    # df = filtered_transactions

    # # Convert sales_amount to numeric, forcing any errors to NaN and dropping them
    # df['sales_amount'] = pd.to_numeric(df['sales_amount'], errors='coerce')
    # df.dropna(subset=['sales_amount'], inplace=True)

    # # Group transactions by store (Site) 
    # grouped = df.groupby(['Site', 'test_or_control'])

    # # Initialize an empty dictionary to store full posterior samples for each store
    # posterior_samples = {}
    # count = 0

    # # Iterate through each group (store-level) and model the sum of sales as a posterior distribution
    # for store, group in grouped:
    #     sales = group['sales_amount'].values  # extract individual transaction sales for the store
    #     count = count + 1

    #     # Check if sales is empty or non-numeric (additional guard)
    #     if len(sales) == 0 or not np.issubdtype(sales.dtype, np.number):
            
    #         print(f"Skipping store {store} due to invalid sales data")
    #         continue

    #     # Fitting the model for the posterior distribution of the sum of sales
    #     with pm.Model() as model:

    #         # Total number of transactions
    #         n_sales = len(sales)

    #         # Flat prior for the mean of the total sales across all transactions (meaning no prior belief that any value on the number line is more likely than any other value)
    #         total_sum_sales = pm.Uniform("total_sum_sales", lower=n_sales * sales.min(), upper=n_sales * sales.max())

    #         # Per-transaction mean derived from the total sum
    #         per_transaction_mu = total_sum_sales / n_sales
            
    #         # Flat prior for the standard deviation of the sales per transaction (meaning no prior belief that any value on the number line is more likely than any other value)
    #         sigma = pm.Uniform("sigma", lower=0, upper=sales.std() * 2)
            
    #         # Likelihood of observing sales per transaction
    #         sales_obs = pm.Normal("sales_obs", mu=per_transaction_mu, sigma=sigma, observed=sales)
            
    #         # Sampling from the posterior with increased tuning and sample size
    #         trace = pm.sample(2000, tune=2000, target_accept=0.95, return_inferencedata=True, progressbar=False)

    #         # Store the full posterior samples for the total sum of sales
    #         posterior_samples[store] = trace.posterior['total_sum_sales'].to_dataframe()
    #         print(f"{count} stores fit with posteriors\n")

    # # Convert the posterior samples into a dictionary for each store
    # # The key will be the store and the value will be a DataFrame of posterior samples
    # posterior_samples_dict = {store: df.reset_index(drop=True) for store, df in posterior_samples.items()}

    # # Example output for a specific store (the first one in the dictionary)
    # store_name = list(posterior_samples_dict.keys())[0]
    # chime.success() 
    # print(f"Posterior samples for store {store_name}:\n", posterior_samples_dict[store_name].head())

    # # Flatten the posterior_samples_dict into a DataFrame
    # # Stacking the store IDs and their corresponding posterior samples

    # flattened_samples = []
    # for store, posterior_df in posterior_samples.items():
    #     store_id = store[0]  # Extract the store ID
    #     test_or_control = store[1]  # Extract test or control group info
    #     posterior_df = posterior_df.reset_index(drop=True)  # Reset the index of the posterior samples
    #     posterior_df['store'] = store_id  # Add the store ID as a column
    #     posterior_df['test_or_control'] = test_or_control  # Add the test or control info
    #     flattened_samples.append(posterior_df)

    # # Concatenate all store samples into a single DataFrame
    # flattened_samples_df = pd.concat(flattened_samples, ignore_index=True)

    # # Merging posterior samples of sum_of_sales with historical performance in order to produce z-score distributions from the raw sales distributions
    # # flattened_samples_df: Contains store, test_or_control, total_sum_sales
    # # campaign_data_df: Contains campaign_id, Site (store), stddev_sales_amount, weekly_avg_sales_amt
    # check = client.query(
    # f"""
    #     SELECT 
    #         baseline.*, 
    #         campaign.test_or_control 
    #     FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_baseline_statistics_with_campaign baseline
    #     LEFT JOIN (SELECT DISTINCT campaign_id, Site, test_or_control FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_period_transactions) campaign
    #         ON baseline.campaign_id = campaign.campaign_id 
    #         AND baseline.Site = campaign.Site
    #     WHERE baseline.campaign_id = '{campaign_id}'
    #     AND weeks_count = 12
    # """
    # ).result()
    # campaign_data_df = check.to_dataframe()

    # # Join the historical metrics to the posterior sum of sales distributions by store
    # merged_df = pd.merge(flattened_samples_df, 
    #                     campaign_data_df[['campaign_id', 'Site', 'stddev_sales_amount', 'weekly_avg_sales_amount']],
    #                     left_on='store', right_on='Site', how='left')

    # # Drop the redundant 'Site' column 
    # merged_df = merged_df.drop(columns=['Site'])

    # # Output the merged DataFrame just to have visibility that everything went fine with the join
    # print(merged_df.head(5))

    # print(f"Processing z-scores from posterior distribution of sum_of_sales: {campaign_id}")
    # # Convert 'total_sum_sales', 'weekly_avg_sales_amt', and 'stddev_sales_amount' to float
    # merged_df['total_sum_sales'] = merged_df['total_sum_sales'].astype(float)
    # merged_df['weekly_avg_sales_amount'] = merged_df['weekly_avg_sales_amount'].astype(float)
    # merged_df['stddev_sales_amount'] = merged_df['stddev_sales_amount'].astype(float)

    # # Calculate the campaign_z_score posterior distributions
    # merged_df['campaign_z_score'] = (merged_df['total_sum_sales'] - merged_df['weekly_avg_sales_amount']) / merged_df['stddev_sales_amount']

    # # Output the result
    # print(merged_df.head(5))
    # chime.success() 

    # # 'merged_df' contains columns: 'store', 'test_or_control', 'campaign_z_score'
    # # 'matched_pairs' contains columns: 'test_store', 'control_store'

    # # Ensure 'merged_df' is sorted by 'store' and 'test_or_control'
    # merged_df = merged_df.sort_values(by=['store', 'test_or_control'])
    # merged_df.to_csv("./WOW20000860_1_merge_df.csv")
    # matched_pairs.to_csv("./WOW20000860_1_matched_pairs.csv")
    merged_df = pd.read_csv("./WOW20000860_1_merge_df.csv")
    matched_pairs = pd.read_csv("./WOW20000860_1_matched_pairs.csv")

    def run_model(sigma_value, merged_df, matched_pairs):
    
        # Initialize a list to store the posterior samples of percentage differences
        posterior_percentage_differences = []

        # Set a small constant to handle division by zero in percentage difference calculation
        epsilon = 1e-6

        print(f"Processing posterior distribution of percentage difference in z-scores: {campaign_id}")
        # Iterate through each pair of test and control stores in 'matched_pairs'
        for _, pair in matched_pairs.iterrows():
            test_store = pair['test_store']
            control_store = pair['control_store']
            
            # Extract posterior samples of z-scores for test and control stores
            test_store_posteriors = merged_df[merged_df['store'] == test_store]['campaign_z_score'].values
            control_store_posteriors = merged_df[merged_df['store'] == control_store]['campaign_z_score'].values
            
            # Ensure that the number of posterior samples matches between test and control stores
            if len(test_store_posteriors) != len(control_store_posteriors):
                raise ValueError(f"Mismatch in posterior sample sizes between test store {test_store} and control store {control_store}")
            
            # Compute the percentage difference using the original formula
            # Handle small denominators by adding epsilon
            with np.errstate(divide='ignore', invalid='ignore'):
                perc_diff = (test_store_posteriors - control_store_posteriors) / (np.abs(control_store_posteriors))
            
            # Store the percentage difference samples along with the test and control store ids
            posterior_percentage_differences.append(pd.DataFrame({
                'test_store': test_store,
                'control_store': control_store,
                'posterior_perc_diff': perc_diff
            }))

        # Concatenate all the percentage difference DataFrames
        posterior_percentage_differences_df = pd.concat(posterior_percentage_differences, ignore_index=True)

        # Create an index that identifies which pair each percentage difference belongs to
        pair_labels = posterior_percentage_differences_df.apply(lambda row: (row['test_store'], row['control_store']), axis=1)

        # Create a categorical variable to get codes for each unique pair
        pair_categories = pd.Categorical(pair_labels)
        pair_indices = pair_categories.codes  # This will map each observation to a store pair index
        unique_pairs = pair_categories.categories

        with pm.Model() as model:
            # Prior for the global mean percentage difference (mu_pop)
            # Normal Centered at 0.0 to allow for both positive and negative effects
            mu_pop = pm.Normal('mu_pop', mu=0.0, sigma=0.0035)

            # current trial sigma_pop 
            sigma_pop = pm.HalfNormal('sigma_pop', sigma=sigma_value)

            # store-pair-specific effects (using centered parameterization)
            store_pair_effects = pm.Normal('store_pair_effects', mu=mu_pop, sigma=sigma_pop, shape=len(unique_pairs))
            
            # assign the correct store_pair_effect to each percentage difference using pair_indices
            store_pair_effects_repeated = store_pair_effects[pair_indices]
            
            # prior for the observational standard deviation (sigma_obs)
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.005)
            
            # Likelihood: Using normal distribution
            observed_perc_diff = pm.Normal(
                'observed_perc_diff',
                mu=store_pair_effects_repeated,
                sigma=sigma_obs,
                observed=posterior_percentage_differences_df['posterior_perc_diff'].values
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


    def evaluate_model(summary_df):
        population_mean = summary_df.loc['mu_pop', 'mean']
        group_estimates = summary_df.loc[summary_df.index.str.contains('store_'), 'mean']
        
        # Calculate the percentage of group estimates within ±0.05 of the population mean
        within_range = np.abs(group_estimates - population_mean) <= 0.3
        percentage_within_range = np.mean(within_range) * 100
        
        # Calculate the percentage of estimates exactly equal to the population mean (underfitting)
        percentage_exact = np.mean(np.abs(group_estimates - population_mean) < 1e-6) * 100
        
        return percentage_within_range, percentage_exact

    
    def optimize_sigma(merged_df, matched_pairs, initial_sigma=0.00088, initial_step=0.0005, max_iterations=80):
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

        return results_df, optimal_trace, optimal_summary, summary_df, unique_pairs


    # Usage example (assuming merged_df and matched_pairs are already prepared)
    results_df, optimal_trace, optimal_summary, unique_pairs = optimize_sigma(merged_df, matched_pairs)
    summary_df.to_csv(f"./{campaign_id}_regularization_output.csv", index=False)

    # If you want to display the final results:
    if optimal_summary is not None:
        print("\nFinal Model Summary:")
        print(optimal_summary[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']])

    
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
        with campaign_period_sum_of_sales AS (
            SELECT 
                Site, 
                test_or_control,
                SUM(sales_amount) AS total_campaign_period_sales_amount 
            FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_campaign_period_transactions
            WHERE test.campaign_id = '{campaign_id}'
            GROUP BY 1,2
        ), 
        historical_performance AS (
            SELECT 
                baseline.Site, 
                baseline.test_or_control,
                baseline.weekly_avg_sales_amount, 
                baseline.stddev_sales_amount,
                campaign_period_sum_of_sales.total_campaign_period_sales_amount, 
                (campaign_period_sum_of_sales.total_campaign_period_sales_amount - baseline.weekly_avg_sales_amount) / baseline.stddev_sales_amount AS campaign_z_score_empirical
            FROM cart-dai-sandbox-nonprod-c3e7.{data_scientist}.regularization_process_baseline_statistics_with_campaign baseline
            LEFT JOIN campaign_period_sum_of_sales 
                ON baseline.Site = campaign_period_sum_of_sales.Site
            WHERE booking_id = '{campaign_id}'
        )
        SELECT 
            '{campaign_id}' AS campaign_id, 
            test.Site AS test_store, 
            control.Site AS control_store, 
            test.total_campaign_period_sales_amount AS test_store_campaign_period_sales,
            control.total_campaign_period_sales_amount AS control_store_campaign_period_sales, 
            test.weekly_avg_sales_amount AS test_store_weekly_avg_sales_amount, 
            control.weekly_avg_sales_amount AS control_store_weekly_avg_sales_amount, 
            test.stddev_sales_amount AS test_store_stddev_sales_amount, 
            control.stddev_sales_amount AS control_store_stddev_sales_amount, 
            test.campaign_z_score_empirical AS test_store_empirical_sales_amount_z_score,
            control.campaign_z_score_empirical AS control_store_empirical_sales_amount_z_score
        FROM historical_performance test 
        LEFT JOIN historical_performance control
            ON test.Site <> control.Site 
        WHERE test.campaign_id = '{campaign_id}'
        AND control.campaign_id = '{campaign_id}'
        AND test.test_or_control = "Test" 
        AND control.test_or_control = "Control" 
        AND test.weeks_count = 12
        AND control.weeks_count = 12
        ) 
        SELECT 
            *, 
            (test_store_empirical_sales_amount_z_score - control_store_empirical_sales_amount_z_score) / ABS(control_store_empirical_sales_amount_z_score) AS z_score_perc_diff_empirical
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
    result_df['z_score_perc_diff_empirical'] = result_df['z_score_perc_diff_empirical'] * 100
    pd.set_option('display.float_format', '{:,.2f}'.format)
    # Join the processed summary dataframe with the merged_df on test_store and control_store
    comparison_df = pd.merge(
        summary_df_processed[['test_store', 'control_store', 'mean']],
        result_df[['test_store', 'control_store', 'z_score_perc_diff_empirical']],  # Keep the relevant columns from merged_df
        on=['test_store', 'control_store'],
        how='left'
    )

    # Rename the columns for clarity
    comparison_df = comparison_df.rename(columns={
        'mean': 'pymc_percentage_diff',
        'z_score_perc_diff_empirical': 'z_score_perc_diff_empirical'
    })

    # Display the merged dataframe for comparison
    comparison_df['pymc_percentage_diff'] = comparison_df['pymc_percentage_diff'].round(2)
    # Adding the campaign_id column to the DataFrame
    comparison_df.insert(0, 'campaign_id', campaign_id)

    # Display the updated DataFrame
    print(comparison_df)
    comparison_df.to_csv(f"./{campaign_id}_regularization_output.csv", index=False)
    chime.success() 
    chime.success() 
    chime.success() 

