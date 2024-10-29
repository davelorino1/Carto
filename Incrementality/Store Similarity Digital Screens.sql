CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.digital_screens_store_comparisons_plus_baseline_4 AS
    WITH 

    n_days AS (
        SELECT 
            campaign_id, 
            MAX(n_days_campaign_period) AS max_days_campaign_period 
        FROM  gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_pre_vs_during_period_plus_baseline_4
        --WHERE campaign_id = current_campaign_global_var
        GROUP BY 1
    ),

    step_two AS (
        SELECT 
            res.*, 
            SAFE_DIVIDE(total_sales_campaign_period , total_sales_pre_period) - 1 AS perc_sales_uplift,
            total_sales_campaign_period - total_sales_pre_period AS raw_sales_uplift
        FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_pre_vs_during_period_plus_baseline_4 res
        LEFT JOIN n_days
            ON res.campaign_id = n_days.campaign_id 
            AND res.n_days_campaign_period = n_days.max_days_campaign_period 
        WHERE max_days_campaign_period IS NOT NULL
    ),

    step_three AS (
        SELECT
            test.campaign_id, 
            test.media_start_date,
            test.media_end_date,
            test.Site AS test_store, 
            control.Site AS control_store,
            test.perc_sales_uplift AS test_store_perc_uplift, 
            control.perc_sales_uplift AS control_store_perc_uplift, 
            test.raw_sales_uplift AS test_store_raw_uplift,
            control.raw_sales_uplift AS control_store_raw_uplift,
            test.total_sales_pre_period AS test_store_pre_sales,
            control.total_sales_pre_period AS control_store_pre_sales,
            test.total_sales_campaign_period AS test_store_campaign_sales, 
            control.total_sales_campaign_period AS control_store_campaign_sales
        FROM step_two test
        LEFT JOIN step_two control 
            ON test.campaign_id = control.campaign_id 
            AND test.Site <> control.Site 
        WHERE test.test_or_control = "Test" 
        AND control.test_or_control = "Control" 
        --AND test.campaign_id = current_campaign_global_var
        --AND control.campaign_id = current_campaign_global_var
    ), 

    step_four AS (
        SELECT 
            step_three.*,
            SAFE_DIVIDE(baseline_test.sales_amount , 12) AS test_mean_historical_sales,
            SAFE_DIVIDE(baseline_control.sales_amount , 12) AS control_mean_historical_Sales, 
            SAFE_DIVIDE(SAFE_DIVIDE(baseline_test.sales_amount , 12) , SAFE_DIVIDE(baseline_control.sales_amount , 12)) AS test_vs_control_mean_historical_sales_perc_diff, 
            baseline_test.stddev_sales_amount AS test_stddev_sales_amount, 
            baseline_control.stddev_sales_amount AS control_stddev_sales_amount,
            SAFE_DIVIDE(baseline_test.stddev_sales_amount , baseline_control.stddev_sales_amount) AS test_vs_control_stddev_sales_amount_perc_diff, 
            ABS(SAFE_DIVIDE(SAFE_DIVIDE(baseline_test.sales_amount , 12) , SAFE_DIVIDE(baseline_control.sales_amount , 12))) + ABS(SAFE_DIVIDE(baseline_test.stddev_sales_amount , baseline_control.stddev_sales_amount)) AS sum_of_abs_perc_diffs
        FROM step_three 
        
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_3 baseline_test 
            ON step_three.test_store = baseline_test.Site AND step_three.campaign_id = baseline_test.campaign_id

        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_3 baseline_control 
            ON step_three.control_store = baseline_control.Site AND step_three.campaign_id = baseline_control.campaign_id
        
        /*LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_store_comparison_of_prior_campaign_impact_medians sim 
            ON CAST(sim.store_id AS STRING) = CAST(step_three.test_store AS STRING)
            AND CAST(sim.comparison_store_id AS STRING) = CAST(step_three.control_store AS STRING)
        */
    ), 

    step_five AS (
        SELECT 
            *, 
            ROW_NUMBER() OVER(PARTITION BY campaign_id, test_store ORDER BY campaign_id, test_store, sum_of_abs_perc_diffs) AS sim_rank 
        FROM step_four 
    ),

    step_six AS (

        SELECT 
            campaign_id, 
            media_start_date, 
            test_store, 
            MIN(sim_rank) AS min_sim_rank 
        FROM step_five 
        GROUP BY 1,2,3 
    ),

    step_seven AS (
        SELECT DISTINCT step_five.* 
        FROM step_five 
        LEFT JOIN step_six 
            ON step_five.campaign_id = step_six.campaign_id 
            AND step_five.test_store = step_six.test_store 
            AND step_five.sim_rank = step_six.min_sim_rank 
        WHERE step_six.min_sim_rank IS NOT NULL 
    )

    SELECT 
        step_seven.*,
        baseline_test.mean_transactions AS test_store_mean_transactions,
        baseline_test.stddev_transactions AS test_store_stddev_transactions,
        baseline_test.stddev_sales_amount AS test_store_stddev_sales_amount,
        baseline_test.variance_transactions AS test_store_variance_transactions,
        baseline_control.mean_transactions AS control_store_mean_transactions,
        baseline_control.stddev_transactions AS control_store_stddev_transactions,
        baseline_control.stddev_sales_amount AS control_store_stddev_sales_amount,
        baseline_control.variance_transactions AS control_store_variance_transactions,
        test_store_perc_uplift - control_store_perc_uplift AS perc_uplift_effect,
        test_store_raw_uplift - control_store_raw_uplift AS raw_uplift_effect,
        CASE 
            WHEN ABS(test_store_raw_uplift - control_store_raw_uplift) > baseline_test.stddev_sales_amount OR 
                ABS(test_store_raw_uplift - control_store_raw_uplift) > baseline_control.stddev_sales_amount
            THEN "Significant"
            ELSE "Not Significant"
        END AS significance,
        baseline_test.sales_amount AS test_store_sales_amount,
        baseline_control.sales_amount AS control_store_sales_amount

    FROM step_seven
    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_3 baseline_test 
        ON step_seven.test_store = baseline_test.Site AND step_seven.campaign_id = baseline_test.campaign_id
    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_3 baseline_control 
        ON step_seven.control_store = baseline_control.Site AND step_seven.campaign_id = baseline_control.campaign_id;