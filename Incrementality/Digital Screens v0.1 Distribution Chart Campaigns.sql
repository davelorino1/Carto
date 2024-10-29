
## Final Digital Screens Campaigns in Distribution Chart from v0.1

with step_one AS (
    SELECT 
        *, 
        ROW_NUMBER() OVER(PARTITION BY campaign_id, control_store ORDER BY control_store) AS n_control_store_occurrence,
        ROW_NUMBER() OVER(PARTITION BY campaign_id, test_store ORDER BY control_store) AS n_test_store_occurrence
    FROM gcp-wow-cart-data-dev-d4d7.davide.digital_screens_store_similarity
    WHERE test_stddev_sales_amount IS NOT NULL
    ORDER BY media_start_date DESC
),

step_two AS (
    SELECT 
        *
    FROM step_one 
    WHERE n_control_store_occurrence = 1
    AND n_test_store_occurrence = 1
),

n_stores AS (
    SELECT 
        campaign_id, 
        COUNT(DISTINCT test_store) AS n_tests, 
        COUNT(DISTINCT control_store) AS n_controls 
    FROM step_two 
    GROUP BY 1
),

step_three AS (
    SELECT 
        step_two.campaign_id,
        test_store,
        control_store,
        test_store_raw_uplift,
        control_store_raw_uplift,
        test_stddev_sales_amount,
        control_stddev_sales_amount,
        test_mean_historical_sales,
        control_mean_historical_sales,
        test_store_campaign_sales,
        control_store_campaign_sales,
        SAFE_DIVIDE(test_store_campaign_sales - test_mean_historical_sales, test_stddev_sales_amount) AS test_sales_z_score,
        SAFE_DIVIDE(control_store_campaign_sales - control_mean_historical_sales, control_stddev_sales_amount) AS control_sales_z_score,
        SAFE_DIVIDE(test_store_campaign_sales - test_mean_historical_sales, test_stddev_sales_amount) - SAFE_DIVIDE(control_store_campaign_sales - control_mean_historical_sales, control_stddev_sales_amount) AS z_score_diff,
        SAFE_DIVIDE(test_store_campaign_sales - test_mean_historical_sales, test_stddev_sales_amount) * test_mean_historical_sales AS test_dollars_vs_expectation,
        SAFE_DIVIDE(control_store_campaign_sales - control_mean_historical_sales, control_stddev_sales_amount) * control_mean_historical_sales AS control_dollars_vs_expectation
    FROM step_two
    INNER JOIN n_stores 
        ON step_two.campaign_id = n_stores.campaign_id 
    WHERE n_stores.n_tests >= 50 AND n_stores.n_controls >= 50
), 

step_four AS (
    SELECT 
        campaign_id,
        media_type,
        media_start_date,
        media_end_date,
        DATE_DIFF(media_end_date, media_start_date, DAY) + 1 AS n_days,
        SUM(test_store_raw_uplift) AS sum_test_store_raw_dollar_uplift, 
        (SUM(control_store_raw_uplift) * 1.0) AS sum_control_store_raw_dollar_uplift,
        (SUM(test_dollars_vs_expectation) * 1.0) AS test_dollars_vs_expectation,
        SUM(control_dollars_vs_expectation) * 1.0 AS control_dollars_vs_expectation,
        (SUM(test_dollars_vs_expectation) * 1.0) - (SUM(control_dollars_vs_expectation) * 1.0) AS test_vs_control_dollars_vs_expectation,
        (SUM(test_dollars_vs_expectation) * 1.0) / (SUM(control_dollars_vs_expectation) * 1.0) -1 AS perc_test_vs_control_dollars_vs_expectation,
        CASE WHEN SUM(test_dollars_vs_expectation) < 0 OR SUM(control_dollars_vs_expectation) < 0 THEN (SAFE_DIVIDE(SUM(test_dollars_vs_expectation) , SUM(control_dollars_vs_expectation)) - 1) * -1 ELSE (SAFE_DIVIDE(SUM(test_dollars_vs_expectation) , SUM(control_dollars_vs_expectation)) -1) END AS percentage_diff_in_dollars_vs_expectation,
        AVG(test_stddev_sales_amount) AS avg_test_store_stddev_dollar_amount,
        AVG(control_stddev_sales_amount) AS avg_control_store_stddev_dollar_amount,
        AVG(test_sales_z_score) AS avg_test_sales_z_score, 
        AVG(control_sales_z_score) AS avg_control_sales_z_score,
        AVG(z_score_diff) AS avg_z_score_diff,
        AVG(test_sales_z_score) - AVG(control_sales_z_score) AS avg_diff_sales_z_score,
        CASE WHEN AVG(test_sales_z_score) < 0 OR AVG(control_sales_z_score) < 0 THEN (SAFE_DIVIDE(AVG(test_sales_z_score) , AVG(control_sales_z_score)) - 1) * -1 ELSE (SAFE_DIVIDE(AVG(test_sales_z_score) , AVG(control_sales_z_score)) -1) END AS percentage_diff_in_normalized_sales
    FROM step_three 
    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2 m_type 
        ON step_three.campaign_id = m_type.booking_and_asset_number
    GROUP BY 1,2,3,4,5
    --ORDER BY avg_diff_uplift_z_score DESC
) 
SELECT 
    *
FROM step_four WHERE ABS(percentage_diff_in_dollars_vs_expectation) <= .2

ORDER BY percentage_diff_in_normalized_sales DESC
