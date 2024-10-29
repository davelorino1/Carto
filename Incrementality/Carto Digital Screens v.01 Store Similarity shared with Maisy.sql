CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.digital_screens_store_similarity AS
    WITH 

    n_days AS (
        SELECT 
            campaign_id, 
            MAX(n_days_campaign_period) AS max_days_campaign_period 
        FROM  gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_pre_vs_during_period_plus_baseline_4
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
        AND total_baskets_campaign_period / (n_days_campaign_period / 7) >= 100 
        AND total_baskets_pre_period / (n_days_campaign_period / 7) >= 100 
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
    )
    , 

    step_four AS (
        SELECT 
            step_three.*,
            SAFE_DIVIDE(baseline_test.sales_amount , 12) AS test_mean_historical_sales,
            SAFE_DIVIDE(baseline_control.sales_amount , 12) AS control_mean_historical_Sales, 
            SAFE_DIVIDE(SAFE_DIVIDE(baseline_test.sales_amount , 12) , SAFE_DIVIDE(baseline_control.sales_amount , 12)) -1 AS test_vs_control_mean_historical_sales_perc_diff, 
            baseline_test.stddev_sales_amount AS test_stddev_sales_amount, 
            baseline_control.stddev_sales_amount AS control_stddev_sales_amount,
            SAFE_DIVIDE(baseline_test.stddev_sales_amount , baseline_control.stddev_sales_amount) -1 AS test_vs_control_stddev_sales_amount_perc_diff, 
            ABS(SAFE_DIVIDE(SAFE_DIVIDE(baseline_test.sales_amount , 12) , SAFE_DIVIDE(baseline_control.sales_amount , 12)) -1) + ABS(SAFE_DIVIDE(baseline_test.stddev_sales_amount , baseline_control.stddev_sales_amount) -1) AS sum_of_abs_perc_diffs
        FROM step_three 
        
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_3 baseline_test 
            ON step_three.test_store = baseline_test.Site AND step_three.campaign_id = baseline_test.campaign_id

        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_3 baseline_control 
            ON step_three.control_store = baseline_control.Site AND step_three.campaign_id = baseline_control.campaign_id
 
    ),
    step_five AS (

        SELECT
            campaign_id, 
            test_store, 
            control_store,
            test_mean_historical_sales,
            control_mean_historical_Sales,
            test_stddev_sales_amount,
            control_stddev_sales_amount,
            test_vs_control_mean_historical_sales_perc_diff,
            test_vs_control_stddev_sales_amount_perc_diff,
            sum_of_abs_perc_diffs,
            ROW_NUMBER() OVER(PARTITION BY campaign_id, test_store ORDER BY sum_of_abs_perc_diffs) AS row_num
        FROM step_four
        WHERE sum_of_abs_perc_diffs IS NOT NULL
        ORDER BY 1,2,sum_of_abs_perc_diffs
    ),
    step_six AS (
        SELECT
            step_five.*,
            ROW_NUMBER() OVER(PARTITION BY campaign_id, control_store ORDER BY sum_of_abs_perc_diffs) AS control_store_occurrence
        FROM step_five
        WHERE row_num = 1
    ),

    store_counts AS (
    
        SELECT 
            llama.campaign_id,
            COUNT(DISTINCT test_store) AS test_stores_matched,
            test_stores_unmatched,
            COUNT(DISTINCT control_store) AS control_stores_matched,
            control_stores_unmatched
        FROM step_six
        LEFT JOIN (
            SELECT 
                campaign_id, 
                COUNT(DISTINCT test_store) AS test_stores_unmatched,
                COUNT(DISTINCT control_store) AS control_stores_unmatched
            FROM step_three
            GROUP BY 1
        ) unmatched ON step_six.campaign_id = unmatched.campaign_id
        WHERE control_store_occurrence = 1
        GROUP BY 1,3,5
    )

    SELECT 
        step_four.*,

    FROM step_four 
    INNER JOIN step_six 
        ON step_six.campaign_id = step_four.campaign_id 
        AND step_six.control_store = step_four.control_store
        AND step_six.test_store = step_four.test_store
    WHERE
        step_four.campaign_id IN (SELECT DISTINCT campaign_id FROM store_counts WHERE test_stores_matched >= 50)
    AND 
        step_six.control_store_occurrence = 1
    ORDER BY
        campaign_id, test_store, sum_of_abs_perc_diffs
;