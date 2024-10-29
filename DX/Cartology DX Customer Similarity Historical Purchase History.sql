

-- gcp-wow-food-wlx-digaspt-dev.playground.adobe_events_and_cartology_campaign_info_v4_test_and_control
-- `gcp-wow-food-wlx-digaspt-dev.playground.test_transactions` 


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
        ARRAY_AGG(DISTINCT booking_id IGNORE NULLS) AS campaigns 
    FROM gcp-wow-food-wlx-digaspt-dev.playground.adobe_events_and_cartology_campaign_info_v4_test_and_control
    WHERE campaign_start_date >= DATE("2024-01-01") 
    AND campaign_end_date <= DATE("2024-02-07")
);


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
    DELETE FROM gcp-wow-food-wlx-digaspt-dev.playground.cartology_dx_shopper_frequency WHERE booking_id = current_campaign_global_var;
    INSERT INTO gcp-wow-food-wlx-digaspt-dev.playground.cartology_dx_shopper_frequency 
        with spend_and_baskets AS (
        SELECT
            booking_id,
            campaign_start_date,
            campaign_end_date, 
            cohort,
            CASE WHEN shopper_identification_number IS NULL THEN "Never Buyer" ELSE "Buyer" END AS product_customer_relationship,
            shopper_identification_method, 
            exposure.crn AS exposure_identification_number,
            shopper_identification_number,      
            start_txn_date, 
            basket_key, 
            order_context,
            --individual_campaign_product_string,
            COUNT(DISTINCT article) AS n_articles, 
            SUM(tot_net_incld_gst) AS sku_level_spend
        FROM `gcp-wow-food-wlx-digaspt-dev.playground.adobe_events_and_cartology_campaign_info_v4_test_and_control` exposure,
        UNNEST(SPLIT(skus, ",")) AS individual_campaign_product_string

        LEFT JOIN `gcp-wow-food-wlx-digaspt-dev.playground.test_transactions` transactions
            ON exposure.crn = transactions.shopper_identification_number
            --AND TO_HEX(SHA256(transactions.shopper_identification_number)) IS NOT NULL
            AND individual_campaign_product_string = transactions.article 

        WHERE booking_id = current_campaign_global_var
        AND (start_txn_date IS NULL OR start_txn_date <= campaign_end_date)
        --AND COALESCE(TO_HEX(SHA256(exposure.crn)), exposure.hashed_crn) IS NOT NULL
        --AND COALESCE(TO_HEX(SHA256(exposure.crn)), exposure.hashed_crn) <> ""
        AND individual_campaign_product_string IS NOT NULL 
        AND individual_campaign_product_string <> ""
        --AND start_txn_date IS NOT NULL
        GROUP BY 1,2,3,4,5,6,7,8,9,10,11
        ORDER BY shopper_identification_number, start_txn_date
        ), 
        previous_order AS (
        SELECT 
            *, 
            LAG(start_txn_date) OVER(PARTITION BY shopper_identification_number ORDER BY start_txn_date) AS previous_txn_date
        FROM spend_and_baskets 
        ORDER BY shopper_identification_number, start_txn_date
        ), 
        n_days_since_last_order AS (
            SELECT 
                *, 
                DATE_DIFF(start_txn_date, previous_txn_date, DAY) AS days_since_last_order
            FROM previous_order 
        )
        SELECT 
            *, 
            AVG(days_since_last_order) OVER(PARTITION BY shopper_identification_number ORDER BY start_txn_date RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rolling_avg_days_between_orders,
            AVG(sku_level_spend) OVER(PARTITION BY shopper_identification_number ORDER BY start_txn_date RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rolling_avg_sku_level_spend_per_order
        FROM n_days_since_last_order
        ORDER BY shopper_identification_number, start_txn_date
    ;
    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');

    
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_run_logs` WHERE campaign_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_run_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "1" AS query_step,
            "Sku Level Sales" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM gcp-wow-food-wlx-digaspt-dev.playground.cartology_dx_shopper_frequency_brand_level WHERE booking_id = current_campaign_global_var;
    INSERT INTO gcp-wow-food-wlx-digaspt-dev.playground.cartology_dx_shopper_frequency_brand_level 
        with spend_and_baskets AS (
            SELECT DISTINCT
                booking_id,
                campaign_start_date,
                campaign_end_date, 
                cohort,
                CASE WHEN shopper_identification_number IS NULL THEN "Never Buyer" ELSE "Buyer" END AS product_customer_relationship,
                shopper_identification_method, 
                exposure.crn AS exposure_identification_number,
                shopper_identification_number,      
                start_txn_date, 
                basket_key, 
                order_context,
                --individual_campaign_product_string,
                ARRAY_TO_STRING(brands, ", ") AS brands,
                n_articles,
                brand_level_spend
            FROM `gcp-wow-food-wlx-digaspt-dev.playground.adobe_events_and_cartology_campaign_info_v4_test_and_control` exposure,
            UNNEST(SPLIT(skus, ",")) AS individual_campaign_product_string

            -- brand info
            LEFT JOIN (
                SELECT 
                    SUBSTR(ProductNumber, 1, STRPOS(ProductNumber, '-') - 1) AS Article, 
                    MAX(SubCategoryShortDescription) AS SubcatDescription,
                    MAX(Brand) AS BrandDescription
                FROM `gcp-wow-ent-im-tbl-prod.adp_quantium_wow_commercials_view.qtm_product_attributes_v`
                WHERE SalesOrganisation = "1005"
                AND UPPER(Brand) NOT IN ("UNBRANDED", "OTHER", "")
                AND Brand IS NOT NULL
                AND SubCategoryShortDescription IS NOT NULL
                GROUP BY 1
            ) am ON am.Article = individual_campaign_product_string

            -- brand level spend
            LEFT JOIN (
                SELECT 
                    shopper_identification_method,
                    shopper_identification_number, 
                    start_txn_date,
                    basket_key, 
                    order_context, 
                    ARRAY_AGG(DISTINCT brand IGNORE NULLS) AS brands,
                    COUNT(DISTINCT article) AS n_articles, 
                    SUM(tot_net_incld_gst) AS brand_level_spend
                FROM `gcp-wow-food-wlx-digaspt-dev.playground.test_transactions` 
                WHERE brand IN (
                    SELECT DISTINCT 
                        BrandDescription 
                    FROM `gcp-wow-food-wlx-digaspt-dev.playground.adobe_events_and_cartology_campaign_info_v4_test_and_control` exposure2,
                    UNNEST(SPLIT(skus, ",")) AS individual_campaign_product_string2
                    LEFT JOIN (
                        SELECT 
                            SUBSTR(ProductNumber, 1, STRPOS(ProductNumber, '-') - 1) AS Article, 
                            MAX(SubCategoryShortDescription) AS SubcatDescription,
                            MAX(Brand) AS BrandDescription
                        FROM `gcp-wow-ent-im-tbl-prod.adp_quantium_wow_commercials_view.qtm_product_attributes_v`
                        WHERE SalesOrganisation = "1005"
                        AND UPPER(Brand) NOT IN ("UNBRANDED", "OTHER", "")
                        AND Brand IS NOT NULL
                        AND SubCategoryShortDescription IS NOT NULL
                        GROUP BY 1
                    ) am2 ON am2.Article = individual_campaign_product_string2
                    WHERE  exposure2.booking_id = current_campaign_global_var
                )
                GROUP BY 1,2,3,4,5
            ) transactions
                ON exposure.crn = transactions.shopper_identification_number

            WHERE booking_id = current_campaign_global_var
            AND (start_txn_date IS NULL OR start_txn_date < campaign_start_date)
            --AND COALESCE(TO_HEX(SHA256(exposure.crn)), exposure.hashed_crn) IS NOT NULL
            --AND COALESCE(TO_HEX(SHA256(exposure.crn)), exposure.hashed_crn) <> ""
            AND individual_campaign_product_string IS NOT NULL 
            AND individual_campaign_product_string <> ""
            --AND start_txn_date IS NOT NULL
            ORDER BY shopper_identification_number, start_txn_date
            ), 
            previous_order AS (
            SELECT 
                *, 
                CASE WHEN shopper_identification_number IS NOT NULL THEN LAG(start_txn_date) OVER(PARTITION BY shopper_identification_number ORDER BY start_txn_date) ELSE NULL END AS previous_txn_date
            FROM spend_and_baskets 
            ORDER BY shopper_identification_number, start_txn_date
            ), 
            n_days_since_last_order AS (
                SELECT 
                    *, 
                    DATE_DIFF(start_txn_date, previous_txn_date, DAY) AS days_since_last_order
                FROM previous_order 
            )
            SELECT 
                *, 
                CASE WHEN shopper_identification_number IS NOT NULL THEN AVG(days_since_last_order) OVER(PARTITION BY shopper_identification_number ORDER BY start_txn_date RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) ELSE NULL END AS rolling_avg_days_between_orders,
                CASE WHEN shopper_identification_number IS NOT NULL THEN AVG(brand_level_spend) OVER(PARTITION BY shopper_identification_number ORDER BY start_txn_date RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) ELSE NULL END AS rolling_avg_brand_level_spend_per_order
            FROM n_days_since_last_order
            ORDER BY shopper_identification_number, start_txn_date
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_run_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "2" AS query_step,
            "Brand Level Sales" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_run_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "3" AS query_step,
            "Total Runtime" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, campaign_run_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, campaign_run_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

  SET start_index = start_index + 1;
END LOOP;