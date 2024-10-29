



/*
 - Cartology Campaign Incrementality Retrospective Analysis

Davide Lorino - June 6th 2024
*/


-- ######################################################################################################################################## 
-- Loop and Date Variables
DECLARE start_index INT64 DEFAULT 0;
DECLARE end_index INT64;

DECLARE campaigns_to_be_analysed_array_global_var ARRAY<STRING>;
DECLARE current_campaign_global_var STRING;

-- Query Execution Time Logging Variables
DECLARE campaign_run_start_time DATETIME;
DECLARE query_start_time DATETIME;
DECLARE query_end_time DATETIME;


SET campaigns_to_be_analysed_array_global_var = (
    SELECT 
        ARRAY_AGG(DISTINCT campaign_id IGNORE NULLS) AS campaigns 
    FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_compliant_cohorts_and_stockouts_since_2023_by_trading_hour
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
    SET current_campaign_global_var = campaigns_to_be_analysed_array_global_var[OFFSET(start_index)];

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_one_stock_levels_by_trading_hour WHERE campaign_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_one_stock_levels_by_trading_hour 
            SELECT 
                campaign_id,
                store_id,
                --COUNT(DISTINCT campaign_id) AS all_campaigns,
                --COUNT(DISTINCT CASE WHEN total_trading_hours IS NULL THEN campaign_id ELSE NULL END) AS campaigns_with_missing_sku_counts
                COUNT(DISTINCT sku) AS total_store_skus,
                COUNT(DISTINCT CASE WHEN total_trading_hours IS NULL THEN sku ELSE NULL END) AS unchecked_stock_level_skus,
                COUNT(DISTINCT CASE WHEN total_trading_hours IS NULL THEN sku ELSE NULL END) / COUNT(DISTINCT sku) unchecked_stock_level_skus_perc
            FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_compliant_cohorts_and_stockouts_since_2023_by_trading_hour, 
            UNNEST(campaign_skus) sku
            WHERE campaign_id = current_campaign_global_var
            GROUP BY 1,2
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_run_logs` WHERE campaign_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_run_logs` 
        SELECT  
            current_campaign_global_var AS campaign_id,
            "1" AS query_step,
            "Stock Levels" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_two_compliance_by_media_type WHERE campaign_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_two_compliance_by_media_type 
            SELECT 
                trading.campaign_start_date,
                trading.campaign_id, 
                CONCAT("GROUP ", trading.media_split_cohort) AS test_group,
                trading.store_id,
                ARRAY_AGG(CASE WHEN is_compliant = 1 THEN media_type ELSE NULL END IGNORE NULLS) AS compliant_media_types,
                ARRAY_AGG(CASE WHEN is_compliant = 0 THEN media_type ELSE NULL END IGNORE NULLS) AS non_compliant_media_types
            FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_compliant_cohorts_and_stockouts_since_2023_by_trading_hour trading
            WHERE campaign_id = current_campaign_global_var
            GROUP BY 1,2,3,4 
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_run_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "2" AS query_step,
            "Compliance by Media Type" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;


    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
        -- unnest and roll up the distinct skus from all "week 1" media types
        -- this allows us to compare performance across all campaign skus regardless of which asset ran or was compliant
        -- this gives us an objective measure of whether an asset had an influence or whether sales were the same or higher without the asset
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_three_unique_skus_by_campaign_id  WHERE campaign_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_three_unique_skus_by_campaign_id
        SELECT 
            trading.campaign_start_date,
            trading.campaign_id,
            -- unnest and roll up the distinct skus from all "week 1" media types
            ARRAY_AGG(DISTINCT campaign_sku IGNORE NULLS) AS all_campaign_skus 
        FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_compliant_cohorts_and_stockouts_since_2023_by_trading_hour trading
        LEFT JOIN `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns` ci 
            ON trading.campaign_start_date = ci.media_start_date -- joining only the media and skus that ran on the campaign_start_date to get week 1 media only
            AND trading.campaign_id = ci.booking_number, 
        UNNEST(SPLIT(quoteline_sku, ",")) campaign_sku
        WHERE trading.campaign_id = current_campaign_global_var
        GROUP BY 1,2
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_run_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "3" AS query_step,
            "Unique Skus by Campaign ID" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_four_pre_campaign_sales WHERE campaign_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_four_pre_campaign_sales   
        SELECT 
            trading.campaign_start_date,
            trading.campaign_id,
            CONCAT("GROUP ", trading.media_split_cohort) AS test_group,
            trading.store_id,
            trading.week_number,
            trading.subcats,
            trading.media_types_string,
            ARRAY_TO_STRING(step_two.compliant_media_types, ", ") AS compliant_media_types,
            ARRAY_TO_STRING(step_two.non_compliant_media_types, ", ") AS non_compliant_media_types,
            all_baskets_with_or_without_promoted_skus_pre_campaign,
            ARRAY_AGG(DISTINCT campaign_sku IGNORE NULLS) AS campaign_skus,
            
            -- Pre Period Sales (total and daily avg)
            COUNT(DISTINCT ass_pre_period.TXNStartDate) AS n_days_pre_period,
            SUM(ass_pre_period.TotalAmountIncldTax) AS total_sales_pre_period,
            COUNT(DISTINCT ass_pre_period.BasketKey) AS total_baskets_pre_period,
            --SUM(ass_pre_period.tot_amt_incld_gst) / COUNT(DISTINCT DATE(ass_pre_period.start_txn_time)) AS daily_avg_sales_pre_period,

        FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_compliant_cohorts_and_stockouts_since_2023_by_trading_hour trading
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_one_stock_levels_by_trading_hour step_one 
            ON step_one.campaign_id = trading.campaign_id 
            AND step_one.store_id = trading.store_id
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_two_compliance_by_media_type step_two 
            ON step_two.campaign_id = trading.campaign_id 
            AND step_two.campaign_start_date = trading.campaign_start_date  
            AND step_two.store_id = trading.store_id
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_three_unique_skus_by_campaign_id step_three 
            ON step_three.campaign_id = trading.campaign_id 
            AND step_three.campaign_start_date = trading.campaign_start_date, 
        UNNEST(all_campaign_skus) AS campaign_sku
        
        -- Pre Period
        LEFT JOIN `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` ass_pre_period
            ON ass_pre_period.Site = CAST(trading.store_id AS INT64)
            AND ass_pre_period.TXNStartDate BETWEEN DATE_ADD(trading.campaign_start_date, INTERVAL -7 DAY) AND DATE_ADD(trading.campaign_start_date, INTERVAL -1 DAY)
            AND ass_pre_period.Article = campaign_sku 

        LEFT JOIN (
            SELECT 
                Site, 
                COUNT(DISTINCT BasketKey) AS all_baskets_with_or_without_promoted_skus_pre_campaign
            FROM `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` all_baskets
            INNER JOIN (
                SELECT DISTINCT  
                    booking_number, 
                    campaign_start_date,
                    campaign_end_date 
                FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns` campaigns 
                WHERE booking_number = current_campaign_global_var
            ) campaigns ON all_baskets.TXNStartDate BETWEEN (campaigns.campaign_start_date -7) AND (campaigns.campaign_start_date -1)
            GROUP BY 1
        ) all_baskets    
            ON all_baskets.Site = CAST(trading.store_id AS INT64)

        WHERE trading.campaign_id = current_campaign_global_var
        AND step_one.unchecked_stock_level_skus_perc <= 0.025
        AND ass_pre_period.TXNStartDate BETWEEN DATE_ADD(trading.campaign_start_date, INTERVAL -7 DAY) AND DATE_ADD(trading.campaign_start_date, INTERVAL -1 DAY)
        GROUP BY 1,2,3,4,5,6,7,8,9,10
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_run_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "4" AS query_step,
            "Pre Campaign Sales" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_five_during_campaign_sales WHERE campaign_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_five_during_campaign_sales   
        SELECT 
            trading.campaign_start_date,
            trading.campaign_id,
            CONCAT("GROUP ", trading.media_split_cohort) AS test_group,
            trading.store_id,
            trading.week_number,
            trading.subcats,
            trading.media_types_string,
            ARRAY_TO_STRING(step_two.compliant_media_types, ", ") AS compliant_media_types,
            ARRAY_TO_STRING(step_two.non_compliant_media_types, ", ") AS non_compliant_media_types,
            all_baskets_with_or_without_promoted_skus_during_campaign,
            ARRAY_AGG(DISTINCT campaign_sku IGNORE NULLS) AS campaign_skus,

            -- Campaign Period Sales (total and daily avg)
            COUNT(DISTINCT ass_campaign_period.TXNStartDate) AS n_days_campaign_period,
            SUM(ass_campaign_period.TotalAmountIncldTax) AS total_sales_campaign_period,
            COUNT(DISTINCT ass_campaign_period.BasketKey) AS total_baskets_campaign_period,
            --SUM(ass_campaign_period.tot_amt_incld_gst) / COUNT(DISTINCT DATE(ass_campaign_period.start_txn_time)) AS daily_avg_sales_campaign_period,
        FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_compliant_cohorts_and_stockouts_since_2023_by_trading_hour trading
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_one_stock_levels_by_trading_hour step_one 
            ON step_one.campaign_id = trading.campaign_id 
            AND step_one.store_id = trading.store_id
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_two_compliance_by_media_type step_two 
            ON step_two.campaign_id = trading.campaign_id 
            AND step_two.campaign_start_date = trading.campaign_start_date  
            AND step_two.store_id = trading.store_id
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_three_unique_skus_by_campaign_id step_three 
            ON step_three.campaign_id = trading.campaign_id 
            AND step_three.campaign_start_date = trading.campaign_start_date, 
        UNNEST(all_campaign_skus) AS campaign_sku
        
        -- Campaign Period
        LEFT JOIN `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` ass_campaign_period
            ON ass_campaign_period.Site = CAST(trading.store_id AS INT64)
            AND ass_campaign_period.TXNStartDate BETWEEN trading.campaign_start_date AND DATE_ADD(trading.campaign_start_date, INTERVAL 6 DAY)
            AND ass_campaign_period.Article = campaign_sku 

        LEFT JOIN (
            SELECT 
                Site, 
                COUNT(DISTINCT BasketKey) AS all_baskets_with_or_without_promoted_skus_during_campaign
            FROM `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` all_baskets
            INNER JOIN (
                SELECT DISTINCT  
                    booking_number, 
                    campaign_start_date,
                    campaign_end_date 
                FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns` campaigns 
                WHERE booking_number = current_campaign_global_var
            ) campaigns ON all_baskets.TXNStartDate BETWEEN campaigns.campaign_start_date AND (campaigns.campaign_start_date + 6)
            GROUP BY 1
        ) all_baskets    
            ON all_baskets.Site = CAST(trading.store_id AS INT64)

        WHERE trading.campaign_id = current_campaign_global_var
        AND step_one.unchecked_stock_level_skus_perc <= 0.025
        AND ass_campaign_period.TXNStartDate BETWEEN trading.campaign_start_date AND DATE_ADD(trading.campaign_start_date, INTERVAL 6 DAY)
        GROUP BY 1,2,3,4,5,6,7,8,9,10
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_run_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "5" AS query_step,
            "During Campaign Sales" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_six_post_campaign_sales WHERE campaign_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_six_post_campaign_sales   
        SELECT 
            trading.campaign_start_date,
            trading.campaign_id,
            CONCAT("GROUP ", trading.media_split_cohort) AS test_group,
            trading.store_id,
            trading.week_number,
            trading.subcats,
            trading.media_types_string,
            ARRAY_TO_STRING(step_two.compliant_media_types, ", ") AS compliant_media_types,
            ARRAY_TO_STRING(step_two.non_compliant_media_types, ", ") AS non_compliant_media_types,
            all_baskets_with_or_without_promoted_skus_post_campaign,
            ARRAY_AGG(DISTINCT campaign_sku IGNORE NULLS) AS campaign_skus,

            -- Post Period Sales (total and daily avg)
            COUNT(DISTINCT ass_post_period.TXNStartDate) AS n_days_post_period,
            SUM(ass_post_period.TotalAmountIncldTax) AS total_sales_post_period,
            COUNT(DISTINCT ass_post_period.BasketKey) AS total_baskets_post_period
            --SUM(ass_post_period.tot_amt_incld_gst) / COUNT(DISTINCT DATE(ass_post_period.start_txn_time)) AS daily_avg_sales_post_period

        FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_compliant_cohorts_and_stockouts_since_2023_by_trading_hour trading
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_one_stock_levels_by_trading_hour step_one 
            ON step_one.campaign_id = trading.campaign_id 
            AND step_one.store_id = trading.store_id
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_two_compliance_by_media_type step_two 
            ON step_two.campaign_id = trading.campaign_id 
            AND step_two.campaign_start_date = trading.campaign_start_date  
            AND step_two.store_id = trading.store_id
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_three_unique_skus_by_campaign_id step_three 
            ON step_three.campaign_id = trading.campaign_id 
            AND step_three.campaign_start_date = trading.campaign_start_date, 
        UNNEST(all_campaign_skus) AS campaign_sku

        -- Campaign End Date
        LEFT JOIN (SELECT DISTINCT booking_number, campaign_start_date, campaign_end_date FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns`) campaigns 
            ON trading.campaign_id = campaigns.booking_number 
            AND trading.campaign_start_date = campaigns.campaign_start_date
        
        -- Post Period
        LEFT JOIN `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v`  ass_post_period
            ON ass_post_period.Site = CAST(trading.store_id AS INT64)
            AND ass_post_period.TXNStartDate BETWEEN DATE_ADD(campaigns.campaign_end_date, INTERVAL 1 DAY) AND DATE_ADD(campaigns.campaign_end_date, INTERVAL 7 DAY)
            AND ass_post_period.Article = campaign_sku 
        
        LEFT JOIN (
            SELECT 
                Site, 
                COUNT(DISTINCT BasketKey) AS all_baskets_with_or_without_promoted_skus_post_campaign
            FROM `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_integrated_sales_view.article_sales_summary_v` all_baskets
            INNER JOIN (
                SELECT DISTINCT  
                    booking_number, 
                    campaign_start_date,
                    campaign_end_date 
                FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns` campaigns 
                WHERE booking_number = current_campaign_global_var
            ) campaigns ON all_baskets.TXNStartDate BETWEEN (campaigns.campaign_end_date +1) AND (campaigns.campaign_end_date +7)
            GROUP BY 1
        ) all_baskets    
            ON all_baskets.Site = CAST(trading.store_id AS INT64)

        WHERE trading.campaign_id = current_campaign_global_var
        AND step_one.unchecked_stock_level_skus_perc <= 0.025
        AND ass_post_period.TXNStartDate BETWEEN DATE_ADD(campaigns.campaign_end_date, INTERVAL 1 DAY) AND DATE_ADD(campaigns.campaign_end_date, INTERVAL 7 DAY)
        GROUP BY 1,2,3,4,5,6,7,8,9,10
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_run_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "6" AS query_step,
            "Post Campaign Sales" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_run_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "7" AS query_step,
            "Entire Duration" AS query_type,
            campaign_run_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, campaign_run_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, campaign_run_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;



  SET start_index = start_index + 1;
END LOOP;