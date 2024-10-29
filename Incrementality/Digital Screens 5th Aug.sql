-- Global vars 
DECLARE start_index INT64 DEFAULT 0;
DECLARE end_index INT64;

DECLARE campaigns_to_be_analysed_array_global_var ARRAY<STRING>;
DECLARE current_campaign_global_var STRING;

DECLARE campaign_run_start_time DATETIME;
DECLARE query_start_time DATETIME;
DECLARE query_end_time DATETIME;

 -- Check if the table contains more than zero rows
DECLARE row_count INT64;

-- Determines which campaigns to analyze in the loop
SET campaigns_to_be_analysed_array_global_var = (
    SELECT 
        ARRAY_AGG(DISTINCT booking_and_asset_number IGNORE NULLS) AS campaigns 
    FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2  --gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards
    WHERE booking_and_asset_number NOT IN (SELECT DISTINCT campaign_id FROM `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` WHERE query_step = "6" AND query_end_time >= "2024-08-01 22:25:25.302964")
    AND DATE_TRUNC(media_start_date, MONTH) NOT IN ("2023-12-01", "2024-01-01")
);

-- Set Loop Exit Point
SET end_index = ARRAY_LENGTH(campaigns_to_be_analysed_array_global_var);


-- ######################################################################################################################################## 

LOOP
  IF start_index >= end_index THEN 
    LEAVE;
  END IF;

    -- Start time for this run
    SET campaign_run_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');

    -- Campaign to analyze on this run
    SET current_campaign_global_var = campaigns_to_be_analysed_array_global_var[OFFSET(start_index)];
    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');




    -- Skus in this campaign
    CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.unique_skus_2 AS (
        SELECT DISTINCT 
            booking_and_asset_number, 
            sku 
        FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2, 
        UNNEST(SPLIT(quoteline_skus_string, ",")) AS sku 
        WHERE booking_and_asset_number = current_campaign_global_var --"WOW20007136"
        AND sku IS NOT NULL
        AND LOWER(sku) <> "npd"
        AND sku <> ""
    );

      -- Check if the table contains more than zero rows
    SET row_count = (SELECT COUNT(*) FROM gcp-wow-cart-data-dev-d4d7.davide.unique_skus_2);

    IF row_count = 0 THEN
    -- Increment start_index and restart the loop
        SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
        DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` WHERE campaign_id = current_campaign_global_var;
        INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` 
        SELECT  
            current_campaign_global_var AS campaign_id,
            "1" AS query_step,
            "Excluded for having zero promoted skus" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
        ;
        SET start_index = start_index + 1;
        CONTINUE;
    END IF;

    CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.prior_period_skus_2 AS (
        with campaign_dates AS (
            SELECT DISTINCT 
                media_start_date, 
                media_end_date, 
                DATE_DIFF(media_end_date, media_start_date, DAY) AS n_campaign_days 
            FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2
            WHERE booking_and_asset_number = current_campaign_global_var
        )
            SELECT DISTINCT
                prior_period_sku 
            FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2 assets
            INNER JOIN campaign_dates 
                ON assets.campaign_start_date >= DATE_ADD(campaign_dates.media_start_date, INTERVAL -campaign_dates.n_campaign_days DAY)
                AND assets.campaign_end_date <= campaign_dates.media_start_date - 1,
            UNNEST(SPLIT(quoteline_skus_string, ",")) prior_period_sku
            INNER JOIN gcp-wow-cart-data-dev-d4d7.davide.unique_skus_2 skus 
                ON skus.sku = prior_period_sku
    );

    SET row_count = (SELECT COUNT(*) FROM gcp-wow-cart-data-dev-d4d7.davide.prior_period_skus_2);

    IF row_count > 0 THEN
        -- Continue with the loop
        -- Log query execution time
        SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
        DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` WHERE campaign_id = current_campaign_global_var;
        INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` 
        SELECT  
            current_campaign_global_var AS campaign_id,
            "1" AS query_step,
            "Excluded for having promoted skus in the pre period" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
        ;
        -- Increment start_index and restart the loop
        SET start_index = start_index + 1;
        CONTINUE;
    END IF;

    -- Test stores where digital screens were placed 
    CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.test_stores_2 AS (
        SELECT DISTINCT 
            booking_number, 
            booking_and_asset_number,
            test_store 
        FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2, 
        UNNEST(store_ids) AS test_store 
        WHERE booking_and_asset_number = current_campaign_global_var
        AND test_store IS NOT NULL
    );

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');

    -- Log query execution time
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` WHERE campaign_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` 
    SELECT  
        current_campaign_global_var AS campaign_id,
        "1" AS query_step,
        "Skus and Stores" AS query_type,
        query_start_time, 
        query_end_time, 
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');

    -- Determine baseline weekly variance in sales & other baseline stats for the campaign skus
        DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_3 WHERE campaign_id = current_campaign_global_var;
        INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_3
        WITH campaign_start_date AS (
            SELECT DISTINCT media_start_date 
            FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2
            WHERE booking_and_asset_number = current_campaign_global_var
        ),

        baseline_weekly_sales AS (
            SELECT 
                Site,
                DATE_TRUNC(TXNStartDate, WEEK) AS sales_week,
                COUNT(DISTINCT BasketKey) AS n_transactions,
                SUM(TotalAmountIncldTax) AS sales_amount
            FROM `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` ass_campaign_period, campaign_start_date
            INNER JOIN gcp-wow-cart-data-dev-d4d7.davide.unique_skus_2 unique_skus
                ON ass_campaign_period.Article = unique_skus.sku 
            WHERE ass_campaign_period.TXNStartDate BETWEEN DATE_SUB(campaign_start_date.media_start_date, INTERVAL 13 WEEK) AND DATE_SUB(campaign_start_date.media_start_date, INTERVAL 1 WEEK)
            AND ass_campaign_period.SalesOrg = 1005
            AND LOWER(ass_campaign_period.SalesChannelDescription) <> "online"
            GROUP BY Site, sales_week
        ),

        baseline_statistics AS (
            SELECT 
                Site,
                AVG(n_transactions) AS mean_transactions,
                STDDEV(n_transactions) AS stddev_transactions,
                STDDEV(sales_amount) AS stddev_sales_amount,
                SUM(sales_amount) AS sales_amount,
                VARIANCE(n_transactions) AS variance_transactions,
                MIN(n_transactions) AS min_transactions,
                MAX(n_transactions) AS max_transactions,
                COUNT(sales_week) AS weeks_count,
                SUM(n_transactions) AS total_transactions
            FROM baseline_weekly_sales
            GROUP BY Site
        )

        SELECT 
            current_campaign_global_var AS campaign_id,
            Site,
            mean_transactions,
            sales_amount,
            stddev_transactions,
            stddev_sales_amount,
            variance_transactions,
            min_transactions,
            max_transactions,
            weeks_count,
            total_transactions
        FROM baseline_statistics;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');

    -- Log query execution time
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` 
    SELECT  
        current_campaign_global_var AS campaign_id,
        "2.5" AS query_step,
        "Baseline Stats (12 Week Lookback) Re-Run" AS query_type,
        query_start_time, 
        query_end_time, 
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;


    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    -- Calculate sales during campaign period (store level)
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_during_period_2 WHERE campaign_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_during_period_2 
    --CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_during_period_2 AS
    SELECT 
        trading.booking_and_asset_number AS campaign_id,
        trading.media_start_date,
        trading.media_end_date,
        ass_campaign_period.Site, 
        CASE WHEN test_stores.test_store IS NOT NULL THEN "Test" ELSE "Control" END AS test_or_control, 
        ARRAY_AGG(DISTINCT Article IGNORE NULLS ORDER BY Article) AS campaign_skus,
        
        -- Pre Period Sales (total and daily avg)
        COUNT(DISTINCT ass_campaign_period.TXNStartDate) AS n_days_campaign_period,
        SUM(ass_campaign_period.TotalAmountIncldTax) AS total_sales_campaign_period,
        COUNT(DISTINCT ass_campaign_period.BasketKey) AS total_baskets_campaign_period

    FROM `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` ass_campaign_period
    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.instore_screens_campaigns_june_2023_onwards_2 trading
        ON ass_campaign_period.TXNStartDate >= trading.media_start_date 
        AND ass_campaign_period.TXNStartDate <= trading.media_end_date 
    INNER JOIN gcp-wow-cart-data-dev-d4d7.davide.unique_skus_2 skus 
        ON skus.sku = ass_campaign_period.Article 
    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.test_stores_2 test_stores 
        ON CAST(test_stores.test_store AS INT64) = CAST(ass_campaign_period.Site AS INT64)
    WHERE trading.booking_and_asset_number = current_campaign_global_var
    AND LOWER(ass_campaign_period.SalesChannelDescription) <> "online"
    AND ass_campaign_period.TXNStartDate >= trading.media_start_date 
    AND ass_campaign_period.TXNStartDate <= trading.media_end_date
    AND ass_campaign_period.SalesOrg = 1005
    GROUP BY ALL;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');

    -- Log query execution time
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` 
    SELECT  
        current_campaign_global_var AS campaign_id,
        "3" AS query_step,
        "Campaign Period Sales" AS query_type,
        query_start_time, 
        query_end_time, 
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;
    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');

    -- Add pre-period sales and baseline stats to the during-campaign period sales table from the previous step
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_pre_vs_during_period_plus_baseline_4 WHERE campaign_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_pre_vs_during_period_plus_baseline_4 
    --CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_pre_vs_during_period_plus_baseline_4 AS 
    SELECT 
        during_period.* EXCEPT(campaign_skus), 
        SUM(ass_pre_period.TotalAmountIncldTax) AS total_sales_pre_period,
        COUNT(DISTINCT ass_pre_period.TXNStartDate) AS n_days_pre_period,
        COUNT(DISTINCT ass_pre_period.BasketKey) AS total_baskets_pre_period,
        bs.mean_transactions,
        bs.stddev_transactions,
        bs.stddev_sales_amount,
        bs.variance_transactions,
        bs.min_transactions,
        bs.max_transactions,
        bs.weeks_count,
        bs.total_transactions,
        bs.sales_amount
    FROM gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_during_period_2 during_period 
    LEFT JOIN `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` ass_pre_period
        ON ass_pre_period.TXNStartDate >= (during_period.media_start_date - INTERVAL during_period.n_days_campaign_period DAY) 
    AND ass_pre_period.TXNStartDate < during_period.media_start_date 
    AND CAST(during_period.Site AS INT64) = CAST(ass_pre_period.Site AS INT64)
    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.baseline_statistics_with_campaign_3 bs
        ON bs.campaign_id = during_period.campaign_id AND bs.Site = during_period.Site
    INNER JOIN gcp-wow-cart-data-dev-d4d7.davide.unique_skus_2 skus 
        ON skus.sku = ass_pre_period.Article 

    WHERE LOWER(ass_pre_period.SalesChannelDescription) <> "online"
    AND ass_pre_period.SalesOrg = 1005
    AND during_period.campaign_id = current_campaign_global_var
    AND ass_pre_period.TXNStartDate >= (during_period.media_start_date - (during_period.n_days_campaign_period + 1)) AND ass_pre_period.TXNStartDate < during_period.media_start_date 
    GROUP BY ALL;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');

    -- Log query execution time
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` 
    SELECT  
        current_campaign_global_var AS campaign_id,
        "4" AS query_step,
        "Pre Period Sales" AS query_type,
        query_start_time, 
        query_end_time, 
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;
    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');


    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.digital_screens_store_comparisons_plus_baseline_4 WHERE campaign_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.digital_screens_store_comparisons_plus_baseline_4 
    WITH 

        n_days AS (
            SELECT 
                campaign_id, 
                MAX(n_days_campaign_period) AS max_days_campaign_period 
            FROM  gcp-wow-cart-data-dev-d4d7.davide.instore_screens_sales_pre_vs_during_period_plus_baseline_4
            WHERE campaign_id = current_campaign_global_var
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
            AND res.campaign_id = current_campaign_global_var
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
            AND test.campaign_id = current_campaign_global_var
            AND control.campaign_id = current_campaign_global_var
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
            ON step_seven.control_store = baseline_control.Site AND step_seven.campaign_id = baseline_control.campaign_id
    ; -- Store Comparisons with new matching (std. dev and mean based matching)

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');

    -- Log query execution time
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` 
    SELECT  
        current_campaign_global_var AS campaign_id,
        "5" AS query_step,
        "Store Comparisons" AS query_type,
        query_start_time, 
        query_end_time, 
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;


    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.instore_screens_run_logs_2` 
    SELECT  
        current_campaign_global_var AS campaign_id,
        "6" AS query_step,
        "Total Runtime" AS query_type,
        query_start_time, 
        query_end_time, 
        DATETIME_DIFF(query_end_time, campaign_run_start_time, SECOND) AS query_duration_in_seconds,
        DATETIME_DIFF(query_end_time, campaign_run_start_time, SECOND) / 60 AS query_duration_in_minutes;
  
  -- Increment the iterator and repeat from the top of the loop
  SET start_index = start_index + 1;
END LOOP;

