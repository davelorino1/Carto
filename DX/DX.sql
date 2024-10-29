

-- ######################################################################################################################################## 
-- Loop and Date Variables
DECLARE start_index INT64 DEFAULT 0;
DECLARE end_index INT64;
DECLARE row_count INT64;

DECLARE campaigns_to_be_analysed_array_global_var ARRAY<STRING>;
DECLARE current_campaign_global_var STRING;

-- Query Execution Time Logging Variables
DECLARE campaign_run_start_time DATETIME;
DECLARE query_start_time DATETIME;
DECLARE query_end_time DATETIME;


-- Campaigns to run the loop across
SET campaigns_to_be_analysed_array_global_var = (
    SELECT 
        ARRAY_AGG(DISTINCT booking_number IGNORE NULLS) AS campaigns 
    FROM gcp-wow-cart-data-dev-d4d7.davide.carto_campaigns
    WHERE campaign_start_date >= DATE("2023-10-01") 
    AND campaign_end_date <= DATE("2024-06-01")
    AND booking_id NOT IN (
        SELECT DISTINCT 
            campaign_id 
        FROM `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs` 
        WHERE query_end_time >= "2024-06-30 00:26:11.539971"
        AND query_step = "14"
    )
);
-- Loop exit point:
SET end_index = ARRAY_LENGTH(campaigns_to_be_analysed_array_global_var);

/*
CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v3 AS (

-- CIID Rows
    SELECT 
        "CI ID Rows" AS join_method,
        adobe.date_time,
        adobe.shopper_id,
        adobe.crn,
        adobe.booking_id, 
        adobe.ci_id,
        ciid_rows.campaign_start_date,
        ciid_rows.campaign_end_date,
        ciid_rows.media_type AS media_type, 
        ciid_rows.media_start_date AS media_start_date,
        ciid_rows.media_end_date AS media_end_date,
        ciid_rows.campaign_skus AS campaign_skus

    FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_carto_catalogue_homepage_ciid`  adobe 

    -- Join Cartology Campaign Info for Adobe Rows that have CIID
    LEFT JOIN (
        SELECT 
            booking_number, 
            campaign_start_date,
            campaign_end_date,
            media_type,
            line_name, 
            media_start_date,
            media_end_date, 
            ARRAY_AGG(DISTINCT individual_product_string ORDER BY individual_product_string) AS campaign_skus
        FROM `gcp-wow-cart-data-dev-d4d7.davide.carto_campaigns`
        GROUP BY 1,2,3,4,5,6,7
    ) ciid_rows 
        ON ciid_rows.line_name = adobe.ci_id 
        AND adobe.booking_id = ciid_rows.booking_number
        AND adobe.date_time BETWEEN TIMESTAMP(ciid_rows.media_start_date) AND TIMESTAMP(ciid_rows.media_end_date + 1)
        AND adobe.ci_id IS NOT NULL 
        AND adobe.booking_id IS NOT NULL

    WHERE 
        -- Line Name exists
        (ciid_rows.line_name IS NOT NULL AND ciid_rows.line_name <> "") 
        AND adobe.date_time BETWEEN TIMESTAMP(ciid_rows.media_start_date) AND TIMESTAMP(ciid_rows.media_end_date + 1)
        AND adobe.ci_id IS NOT NULL 
        AND adobe.booking_id IS NOT NULL

    UNION ALL

-- Catalogue Rows 
    SELECT 
        "Catalogue" AS join_method,
        adobe2.date_time,
        adobe2.shopper_id,
        adobe2.crn,
        catalogue_rows.booking_number AS booking_id,
        adobe2.ci_id,
        catalogue_rows.campaign_start_date, 
        catalogue_rows.campaign_end_date, 
        catalogue_rows.media_type AS media_type, 
        catalogue_rows.media_start_date AS media_start_date,
        catalogue_rows.media_end_date AS media_end_date,
        catalogue_rows.campaign_skus AS campaign_skus

    FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_carto_catalogue_homepage_ciid`  adobe2
    LEFT JOIN (
        SELECT 
            booking_number, 
            campaign_start_date,
            campaign_end_date,
            media_type, 
            line_name, 
            media_start_date, 
            media_end_date, 
            ARRAY_AGG(DISTINCT individual_product_string ORDER BY individual_product_string) AS campaign_skus
        FROM `gcp-wow-cart-data-dev-d4d7.davide.carto_campaigns`
        WHERE LOWER(media_type) = "catalogue"
        GROUP BY 1,2,3,4,5,6,7
    ) catalogue_rows
        ON adobe2.catalogue_flag = 1
        AND adobe2.booking_id = "Not Found"
    
    WHERE adobe2.catalogue_flag = 1
    AND adobe2.booking_id = "Not Found" 
    AND adobe2.date_time BETWEEN TIMESTAMP(catalogue_rows.media_start_date) AND TIMESTAMP(catalogue_rows.media_end_date + 1)

-- Home Page rows that have ci_id and booking_id in the adobe tracking

    UNION ALL  

    SELECT 
        "Homepage Rows" AS join_method,
        adobe3.date_time,
        adobe3.shopper_id,
        adobe3.crn,
        homepage_rows.booking_number AS booking_id,
        adobe3.ci_id,
        homepage_rows.campaign_start_date, 
        homepage_rows.campaign_end_date,
        homepage_rows.media_type AS media_type, 
        homepage_rows.media_start_date AS media_start_date,
        homepage_rows.media_end_date AS media_end_date,
        homepage_rows.campaign_skus AS campaign_skus

    FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_carto_catalogue_homepage_ciid`  adobe3
    LEFT JOIN (
        SELECT 
            booking_number, 
            campaign_start_date,
            campaign_end_date,
            media_type, 
            "" AS line_name, 
            media_start_date, 
            media_end_date, 
            ARRAY_AGG(DISTINCT individual_product_string ORDER BY individual_product_string) AS campaign_skus
        FROM `gcp-wow-cart-data-dev-d4d7.davide.carto_campaigns`
        WHERE LOWER(media_type) = "home page"
        GROUP BY 1,2,3,4,5,6,7
    ) homepage_rows
        ON adobe3.homepage_flag = 1
        AND adobe3.booking_id = "Not Found"
        AND adobe3.date_time BETWEEN TIMESTAMP(homepage_rows.media_start_date) AND TIMESTAMP(homepage_rows.media_end_date + 1)
    WHERE adobe3.homepage_flag = 1
    AND adobe3.booking_id = "Not Found"
    AND adobe3.date_time BETWEEN TIMESTAMP(homepage_rows.media_start_date) AND TIMESTAMP(homepage_rows.media_end_date + 1)

);


-- 4. Reduce to only the distinct shoppers events + media types / bookings numbers
CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4 AS 
    SELECT DISTINCT
        DATE(date_time) date,
        booking_id, 
        campaign_start_date,
        campaign_end_date,
        ci_id,
        media_type, 
        shopper_id, 
        crn, 
        ARRAY_TO_STRING(campaign_skus, ",") skus
FROM gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v3
WHERE crn IS NOT NULL;
*/

LOOP
    IF start_index >= end_index THEN 
        LEAVE;
    END IF;

    -- Start time for this run
    SET campaign_run_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    SET current_campaign_global_var = campaigns_to_be_analysed_array_global_var[OFFSET(start_index)];

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_control WHERE booking_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_control 
        WITH campaign_dates AS (
            SELECT DISTINCT
                booking_id,
                campaign_start_date,
                campaign_end_date,
                skus
            FROM
                gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4 test
            WHERE booking_id = current_campaign_global_var
        ),
        exposed_shoppers AS (
            SELECT DISTINCT
                booking_id,
                campaign_start_date,
                shopper_id,
                crn
            FROM
                gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4 test
            WHERE booking_id = current_campaign_global_var
            AND crn IS NOT NULL
        ),
        activity_during_campaign AS (
            SELECT DISTINCT
                c.booking_id,
                c.campaign_start_date,
                w.shopper_id,
                w.crn
            FROM
                `gcp-wow-cart-data-dev-d4d7.davide.adobe_carto_catalogue_homepage_ciid_control`w
            INNER JOIN
                campaign_dates c
            ON
                w.date BETWEEN c.campaign_start_date AND c.campaign_end_date
            AND w.crn IS NOT NULL
        )
        SELECT
            adc.booking_id,
            campaign_dates.campaign_start_date,
            campaign_dates.campaign_end_date,
            'Not Exposed' AS cohort,
            NULL AS date,  -- Since non-exposure is not tied to a specific date
            adc.shopper_id,
            adc.crn,
            campaign_dates.skus
        FROM activity_during_campaign adc
        
        LEFT JOIN campaign_dates 
            ON adc.booking_id = campaign_dates.booking_id
        
        -- remove exposed shopper_ids
        LEFT JOIN (
            SELECT DISTINCT 
                booking_id,
                campaign_start_date, 
                shopper_id
            FROM exposed_shoppers 
        ) exposed_shopper_id 
            ON adc.booking_id = exposed_shopper_id.booking_id 
            AND adc.campaign_start_date = exposed_shopper_id.campaign_start_date 
            AND adc.shopper_id = exposed_shopper_id.shopper_id

        -- remove exposed crns
        LEFT JOIN (
            SELECT DISTINCT 
                booking_id,
                campaign_start_date, 
                crn
            FROM exposed_shoppers 
        ) exposed_crn
            ON adc.booking_id = exposed_crn.booking_id 
            AND adc.campaign_start_date = exposed_crn.campaign_start_date 
            AND adc.crn = exposed_crn.crn

        WHERE exposed_shopper_id.shopper_id IS NULL 
        AND exposed_crn.crn IS NULL 
        AND adc.crn IS NOT NULL
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "1" AS query_step,
            "All Control Shoppers" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_test_and_control WHERE booking_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_test_and_control  
        SELECT       
            date,
            booking_id, 
            campaign_start_date,
            campaign_end_date,
            "Exposed" AS cohort,
            ci_id,
            media_type, 
            shopper_id, 
            crn, 
            skus
        FROM gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4
        WHERE booking_id = current_campaign_global_var

        UNION ALL 

        SELECT 
            CAST(NULL AS DATE) AS date, 
            booking_id, 
            campaign_start_date,
            campaign_end_date,
            cohort, 
            "" AS ci_id, 
            "" AS media_type, 
            shopper_id, 
            crn, 
            skus
        FROM gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_control 
        WHERE booking_id = current_campaign_global_var
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "2" AS query_step,
            "Test Control Union" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_conversions_sku_level_v2` WHERE booking_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_conversions_sku_level_v2`
        with step_one AS (
            SELECT * FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_test_and_control` 
            WHERE campaign_start_date >= DATE("2023-10-01") 
            AND booking_id = current_campaign_global_var
        ),
        step_two AS (
            SELECT transactions.* 
            FROM `gcp-wow-cart-data-dev-d4d7.davide.test_transactions` transactions
            CROSS JOIN (SELECT DISTINCT campaign_start_date, campaign_end_date FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_test_and_control` WHERE booking_id = current_campaign_global_var LIMIT 1)
            WHERE DATE(start_txn_time) >= campaign_start_date 
            AND DATE(start_txn_time) <= campaign_end_date

        )
        -- Exposed Shopper Transactions
        SELECT 
            DATE(exposure.date) AS impression_date, 
            exposure.booking_id, 
            exposure.cohort AS exposure_context,
            exposure.crn AS exposure_crn,
            transactions.order_context,
            transactions.shopper_identification_number AS converted_shopper_identification_number,
            transactions.start_txn_time AS start_txn_time,
            transactions.basket_key AS basket_key,
            transactions.article AS article,
            transactions.tot_net_incld_gst AS tot_net_incld_gst,
            CASE WHEN DATE(transactions.start_txn_time) < DATETIME_ADD(exposure.date, INTERVAL 3 DAY) 
                THEN "3 Days" 
                ELSE "7 Days" 
            END AS exposure_to_purchase_window
        
        FROM step_one  exposure,
        UNNEST(SPLIT(skus, ",")) AS individual_campaign_product_string

        -- WHere crn is not null
        LEFT JOIN step_two transactions 
            ON exposure.crn = transactions.shopper_identification_number
            AND individual_campaign_product_string = transactions.article 
            AND 
                (
                    (
                        DATE(transactions.start_txn_time) >= campaign_start_date
                        AND DATE(transactions.start_txn_time) <= campaign_end_date
                        AND cohort = "Exposed"
                    )
                    OR 
                    ( 
                        DATE(transactions.start_txn_time) <= campaign_end_date
                        AND DATE(transactions.start_txn_time) >= campaign_start_date
                        AND exposure.cohort = "Not Exposed"
                    )
                )
            AND transactions.shopper_identification_number IS NOT NULL
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "3" AS query_step,
            "Sku Level Sales - Test & Control" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;
    
    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_conversions_brand_level_v2` WHERE booking_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_conversions_brand_level_v2`
                with step_one AS (
            SELECT * FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_test_and_control` 
            WHERE campaign_start_date >= DATE("2023-10-01") 
            AND booking_id = current_campaign_global_var
        ),
        step_two AS (
            SELECT transactions.* 
            FROM `gcp-wow-cart-data-dev-d4d7.davide.test_transactions` transactions
            CROSS JOIN (SELECT DISTINCT campaign_start_date, campaign_end_date FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_test_and_control` WHERE booking_id = current_campaign_global_var LIMIT 1)
            WHERE DATE(start_txn_time) >= campaign_start_date 
            AND DATE(start_txn_time) <= campaign_end_date

        )
        SELECT 
            DATE(exposure.date) AS impression_date, 
            exposure.booking_id, 
            exposure.cohort AS exposure_context,
            exposure.crn AS exposure_crn,
            transactions.order_context,
            transactions.shopper_identification_number AS converted_shopper_identification_number,
            transactions.start_txn_time AS start_txn_time,
            transactions.basket_key AS basket_key,
            brand_map.brand,
            transactions.article AS article,
            transactions.tot_net_incld_gst AS tot_net_incld_gst,
            CASE WHEN DATE(transactions.start_txn_time) < DATETIME_ADD(exposure.date, INTERVAL 3 DAY) 
                THEN "3 Days" 
                ELSE "7 Days" 
            END AS exposure_to_purchase_window
        
        FROM step_one exposure,
        UNNEST(SPLIT(skus, ",")) AS individual_campaign_product_string

        LEFT JOIN (
            SELECT DISTINCT
                SUBSTR(ProductNumber, 1, STRPOS(ProductNumber, '-') - 1) AS Article, 
                Brand AS brand
            FROM `gcp-wow-ent-im-tbl-prod.adp_quantium_wow_commercials_view.qtm_product_attributes_v`
            WHERE SalesOrganisation = "1005"
        ) brand_map
            ON brand_map.Article = individual_campaign_product_string

        -- WHere crn is not null
        LEFT JOIN step_two transactions 
            ON exposure.crn = transactions.shopper_identification_number
            AND brand_map.brand = transactions.brand 
            AND 
            (
                (
                    --DATE(transactions.start_txn_time) >= DATE(exposure.date) 
                    --AND DATE(transactions.start_txn_time) <= DATE(DATE_ADD(exposure.date, INTERVAL 7 DAY))
                    DATE(transactions.start_txn_time) >= campaign_start_date
                    AND DATE(transactions.start_txn_time) <= campaign_end_date
                    AND cohort = "Exposed"
                )
                OR 
                ( 
                    exposure.cohort = "Not Exposed"
                    AND 
                    DATE(transactions.start_txn_time) <= campaign_end_date
                    AND 
                    DATE(transactions.start_txn_time) >= campaign_start_date
                )
            )
            AND (exposure.crn IS NOT NULL)
            AND transactions.shopper_identification_number IS NOT NULL  
        WHERE brand_map.brand IS NOT NULL
        AND brand_map.brand <> "" 
        AND brand_map.brand <> "UNBRANDED" 
        AND brand_map.brand <> "OTHER"
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "4" AS query_step,
            "Brand Level Sales - Test & Control" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_sales_7_days_sku_level_v2` WHERE booking_id = current_campaign_global_var; 
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_sales_7_days_sku_level_v2`
        SELECT DISTINCT
            booking_id, 
            order_context,
            exposure_crn AS exposed_shopper_identification_number,
            converted_shopper_identification_number,
            start_txn_time,
            basket_key, 
            article,
            tot_net_incld_gst,
            "7 Days" AS exposure_to_purchase_window
        FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_conversions_sku_level_v2` converted_products
        WHERE booking_id = current_campaign_global_var
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "5" AS query_step,
            "Distinct Sku Level Sales - Test & Control" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_sales_7_days_brand_level_v2` WHERE booking_id = current_campaign_global_var; 
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_sales_7_days_brand_level_v2`
        SELECT DISTINCT
            booking_id, 
            brand,
            order_context, 
            exposure_crn AS exposed_shopper_identification_number,
            converted_shopper_identification_number,
            start_txn_time,
            basket_key, 
            article,
            tot_net_incld_gst,
            "7 Days" AS exposure_to_purchase_window
        FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_conversions_brand_level_v2` converted_products
        WHERE booking_id = current_campaign_global_var
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "6" AS query_step,
            "Distinct Brand Level Sales - Test & Control" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    ######### SHOPPER SIMILARITY FEATURES - PURCHASE HISTORY
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_shopper_frequency WHERE booking_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_shopper_frequency 
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
            FROM (
                SELECT DISTINCT 
                    booking_id, 
                    campaign_start_date,
                    campaign_end_date,
                    crn, 
                    cohort, 
                    skus 
                FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_test_and_control` 
                WHERE booking_id = current_campaign_global_var
            ) exposure,
            UNNEST(SPLIT(skus, ",")) AS individual_campaign_product_string

            LEFT JOIN `gcp-wow-cart-data-dev-d4d7.davide.test_transactions` transactions
                ON exposure.crn = transactions.shopper_identification_number
                AND individual_campaign_product_string = transactions.article 

            WHERE booking_id = current_campaign_global_var
            AND (start_txn_date IS NULL OR start_txn_date <= campaign_end_date)
            AND individual_campaign_product_string IS NOT NULL 
            AND individual_campaign_product_string <> ""
            
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
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "7" AS query_step,
            "K-Means Feature Gen - Sku Level Historical Frequency" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_shopper_frequency_brand_level WHERE booking_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_shopper_frequency_brand_level 
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
            FROM (
                SELECT DISTINCT 
                    booking_id, 
                    campaign_start_date,
                    campaign_end_date,
                    crn, 
                    cohort, 
                    skus 
                FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_test_and_control` 
                WHERE booking_id = current_campaign_global_var
            ) exposure,
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
                FROM `gcp-wow-cart-data-dev-d4d7.davide.test_transactions` 
                WHERE brand IN (
                    SELECT DISTINCT 
                        BrandDescription 
                    FROM `gcp-wow-cart-data-dev-d4d7.davide.adobe_events_and_cartology_campaign_info_v4_test_and_control` exposure2,
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
            AND individual_campaign_product_string IS NOT NULL 
            AND individual_campaign_product_string <> ""
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
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "8" AS query_step,
            "K-Means Feature Gen - Brand Level Historical Frequency" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    DELETE FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_summary_stats WHERE booking_id = current_campaign_global_var;
    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_summary_stats 
        with transaction_number_sku_level AS ( 
            SELECT 
                booking_id, 
                campaign_start_date, 
                campaign_end_date, 
                cohort,
                product_customer_relationship,
                exposure_identification_number,
                shopper_identification_number,
                start_txn_date, 
                basket_key,
                order_context, 
                n_articles,
                sku_level_spend,
                rolling_avg_days_between_orders,
                rolling_avg_sku_level_spend_per_order,
                ROW_NUMBER() OVER(PARTITION BY shopper_identification_number ORDER BY start_txn_date) AS txn_number
            FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_shopper_frequency a 
            WHERE shopper_identification_number IS NOT NULL 
            AND start_txn_date < campaign_start_date
            AND booking_id = current_campaign_global_var
            ), 
            latest_entry_sku_level AS (
                SELECT 
                    booking_id,
                    campaign_start_date,
                    shopper_identification_number, 
                    MAX(txn_number) AS latest_txn
                FROM transaction_number_sku_level 
                WHERE shopper_identification_number IS NOT NULL 
                AND booking_id = current_campaign_global_var
                GROUP BY 1,2,3
            ),
            aggregations_sku_level AS (
                SELECT 
                    booking_id,
                    campaign_start_date,
                    campaign_end_date, 
                    cohort, 
                    product_customer_relationship, 
                    exposure_identification_number,
                    shopper_identification_number, 
                    MIN(start_txn_date) AS earliest_order_sku_level, 
                    MAX(start_txn_date) AS latest_order_sku_level, 
                    COUNT(DISTINCT basket_key) AS total_orders_sku_level, 
                    SUM(sku_level_spend) AS sku_level_spend    
                FROM transaction_number_sku_level 
                WHERE shopper_identification_number IS NOT NULL 
                AND booking_id = current_campaign_global_var
                GROUP BY ALL
            ),
            final_results_sku_level AS (
                SELECT 
                    aggregations_sku_level.*,
                    latest_moving_averages_sku_level.rolling_avg_days_between_orders AS rolling_avg_days_between_orders_sku_level,
                    latest_moving_averages_sku_level.rolling_avg_sku_level_spend_per_order AS rolling_avg_sku_level_spend_per_order_sku_level
                FROM aggregations_sku_level
                LEFT JOIN (
                    SELECT 
                        latest_entry_sku_level.*,
                        transaction_number_sku_level.rolling_avg_days_between_orders,
                        transaction_number_sku_level.rolling_avg_sku_level_spend_per_order
                    FROM latest_entry_sku_level
                    INNER JOIN transaction_number_sku_level
                        ON transaction_number_sku_level.booking_id = latest_entry_sku_level.booking_id 
                        AND transaction_number_sku_level.campaign_start_date = latest_entry_sku_level.campaign_start_date 
                        AND transaction_number_sku_level.shopper_identification_number = latest_entry_sku_level.shopper_identification_number
                        AND transaction_number_sku_level.txn_number = latest_entry_sku_level.latest_txn
                        AND transaction_number_sku_level.shopper_identification_number IS NOT NULL
                    WHERE latest_entry_sku_level.shopper_identification_number IS NOT NULL 
                    AND latest_entry_sku_level.booking_id = current_campaign_global_var
                ) latest_moving_averages_sku_level
                    ON latest_moving_averages_sku_level.booking_id = aggregations_sku_level.booking_id 
                    AND latest_moving_averages_sku_level.campaign_start_date = aggregations_sku_level.campaign_start_date 
                    AND latest_moving_averages_sku_level.shopper_identification_number = aggregations_sku_level.shopper_identification_number
                    AND aggregations_sku_level.shopper_identification_number IS NOT NULL
                ORDER BY shopper_identification_number
            ),

            transaction_number_brand_level AS ( 
                SELECT 
                    booking_id, 
                    campaign_start_date, 
                    campaign_end_date, 
                    cohort,
                    product_customer_relationship,
                    exposure_identification_number,
                    shopper_identification_number,
                    start_txn_date, 
                    basket_key,
                    order_context, 
                    n_articles,
                    brand_level_spend,
                    rolling_avg_days_between_orders,
                    rolling_avg_brand_level_spend_per_order,
                    ROW_NUMBER() OVER(PARTITION BY shopper_identification_number ORDER BY start_txn_date) AS txn_number
                FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_shopper_frequency_brand_level
                WHERE shopper_identification_number IS NOT NULL 
                AND start_txn_date < campaign_start_date
                AND booking_id = current_campaign_global_var
            ), 
            latest_entry_brand_level AS (
                SELECT 
                    booking_id,
                    campaign_start_date,
                    shopper_identification_number, 
                    MAX(txn_number) AS latest_txn
                FROM transaction_number_brand_level 
                WHERE shopper_identification_number IS NOT NULL 
                AND booking_id = current_campaign_global_var
                GROUP BY 1,2,3
            ),
            aggregations_brand_level AS (
                SELECT 
                    booking_id,
                    campaign_start_date,
                    campaign_end_date, 
                    cohort, 
                    product_customer_relationship, 
                    exposure_identification_number,
                    shopper_identification_number, 
                    MIN(start_txn_date) AS earliest_order_brand_level, 
                    MAX(start_txn_date) AS latest_order_brand_level, 
                    COUNT(DISTINCT basket_key) AS total_orders_brand_level, 
                    SUM(brand_level_spend) AS brand_level_spend    
                FROM transaction_number_brand_level 
                WHERE shopper_identification_number IS NOT NULL 
                AND booking_id = current_campaign_global_var
                GROUP BY ALL
            ),
            final_results_brand_level AS (
                SELECT 
                    aggregations_brand_level.*,
                    latest_moving_averages_brand_level.rolling_avg_days_between_orders AS rolling_avg_days_between_orders_brand_level,
                    latest_moving_averages_brand_level.rolling_avg_brand_level_spend_per_order AS rolling_avg_brand_level_spend_per_order_brand_level
                FROM aggregations_brand_level
                LEFT JOIN (
                    SELECT 
                        latest_entry_brand_level.*,
                        transaction_number_brand_level.rolling_avg_days_between_orders,
                        transaction_number_brand_level.rolling_avg_brand_level_spend_per_order
                    FROM latest_entry_brand_level
                    INNER JOIN transaction_number_brand_level
                        ON transaction_number_brand_level.booking_id = latest_entry_brand_level.booking_id 
                        AND transaction_number_brand_level.campaign_start_date = latest_entry_brand_level.campaign_start_date 
                        AND transaction_number_brand_level.shopper_identification_number = latest_entry_brand_level.shopper_identification_number
                        AND transaction_number_brand_level.txn_number = latest_entry_brand_level.latest_txn
                        AND transaction_number_brand_level.shopper_identification_number IS NOT NULL
                    WHERE latest_entry_brand_level.shopper_identification_number IS NOT NULL 
                    AND latest_entry_brand_level.booking_id = current_campaign_global_var
                ) latest_moving_averages_brand_level
                    ON latest_moving_averages_brand_level.booking_id = aggregations_brand_level.booking_id 
                    AND latest_moving_averages_brand_level.campaign_start_date = aggregations_brand_level.campaign_start_date 
                    AND latest_moving_averages_brand_level.shopper_identification_number = aggregations_brand_level.shopper_identification_number
                    AND aggregations_brand_level.shopper_identification_number IS NOT NULL
                ORDER BY shopper_identification_number
            )
            SELECT 
                final_results_brand_level.*,
                final_results_sku_level.earliest_order_sku_level, 
                final_results_sku_level.latest_order_sku_level, 
                final_results_sku_level.total_orders_sku_level, 
                final_results_sku_level.sku_level_spend,
                final_results_sku_level.rolling_avg_days_between_orders_sku_level,
                final_results_sku_level.rolling_avg_sku_level_spend_per_order_sku_level
            FROM final_results_brand_level 
            LEFT JOIN final_results_sku_level 
                ON final_results_brand_level.booking_id = final_results_sku_level.booking_id 
                AND final_results_brand_level.exposure_identification_number = final_results_sku_level.exposure_identification_number
                AND final_results_brand_level.shopper_identification_number = final_results_sku_level.shopper_identification_number
                AND final_results_brand_level.campaign_start_date = final_results_sku_level.campaign_start_date
            
    ;

    INSERT INTO gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_summary_stats 
        SELECT DISTINCT 
            brand.booking_id, 
            brand.campaign_start_date, 
            brand.campaign_end_date, 
            brand.cohort, 
            brand.product_customer_relationship, 
            brand.exposure_identification_number,
            brand.shopper_identification_number, 
            CAST(NULL AS DATE) AS earliest_order_brand_level,
            CAST(NULL AS DATE) AS latest_order_brand_level,
            CAST(NULL AS INT64) AS total_orders_brand_level,
            CAST(NULL AS NUMERIC) AS brand_level_spend,
            CAST(NULL AS NUMERIC) AS rolling_avg_days_between_orders_brand_level,
            CAST(NULL AS NUMERIC) AS rolling_avg_brand_level_spend_per_order_brand_level,
            CAST(NULL AS DATE) AS earliest_order_sku_level,
            CAST(NULL AS DATE) AS latest_order_sku_level,
            CAST(NULL AS INT64) AS total_orders_sku_level,
            CAST(NULL AS NUMERIC) AS sku_level_spend, 
            CAST(NULL AS NUMERIC) AS rolling_avg_days_between_orders_sku_level,
            CAST(NULL AS NUMERIC) AS rolling_avg_sku_level_spend_per_order_sku_level
        FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_shopper_frequency_brand_level brand
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_shopper_frequency sku
            ON brand.booking_id = sku.booking_id 
            AND brand.campaign_start_date = sku.campaign_start_date 
            AND brand.exposure_identification_number = sku.exposure_identification_number 
        WHERE brand.product_customer_relationship = "Never Buyer"
        AND brand.booking_id = current_campaign_global_var
        ;


    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history` WHERE booking_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history`
        SELECT 
            converts.exposure_crn,
            CASE WHEN converted_shopper_identification_number IS NULL THEN 0 ELSE 1 END AS converted, 
            COUNT(DISTINCT basket_key) AS n_converted_baskets, 
            SUM(tot_net_incld_gst) AS total_converted_spend,
            summary.* 
        FROM (
            SELECT DISTINCT 
                booking_id, 
                exposure_crn, 
                converted_shopper_identification_number,
                basket_key, 
                article,
                tot_net_incld_gst 
            FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_conversions_brand_level_v2` 
            WHERE booking_id = current_campaign_global_var
        ) converts 
        LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_summary_stats summary
            ON summary.booking_id = converts.booking_id 
            AND summary.exposure_identification_number = converts.exposure_crn
        WHERE summary.booking_id = current_campaign_global_var
        GROUP BY ALL
    ;


    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    ############ K-Means Model Training and Application
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.shopper_features` WHERE booking_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.shopper_features` 
        SELECT 
            current_campaign_global_var AS booking_id,
            exposure_crn,
            IFNULL(total_orders_brand_level, 0) AS total_orders_brand_level,
            IFNULL(brand_level_spend, 0) AS brand_level_spend,
            IFNULL(rolling_avg_days_between_orders_brand_level, 0) AS rolling_avg_days_between_orders_brand_level,
            IFNULL(rolling_avg_brand_level_spend_per_order_brand_level, 0) AS rolling_avg_brand_level_spend_per_order_brand_level,
            IFNULL(total_orders_sku_level, 0) AS total_orders_sku_level,
            IFNULL(sku_level_spend, 0) AS sku_level_spend,
            IFNULL(rolling_avg_days_between_orders_sku_level, 0) AS rolling_avg_days_between_orders_sku_level,
            IFNULL(rolling_avg_sku_level_spend_per_order_sku_level, 0) AS rolling_avg_sku_level_spend_per_order_sku_level,
            cohort
        FROM 
            `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history`
        WHERE 
            booking_id = current_campaign_global_var
            AND (cohort = "Exposed" OR cohort = "Not Exposed");

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "9" AS query_step,
            "K-Means Feature Gen - All Features" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET row_count = (SELECT COUNT(*) FROM `gcp-wow-cart-data-dev-d4d7.davide.shopper_features` WHERE booking_id = current_campaign_global_var);
    
    IF row_count = 0 THEN
        SET start_index = start_index + 1;
        CONTINUE;
    END IF;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    CREATE OR REPLACE MODEL `gcp-wow-cart-data-dev-d4d7.davide.shopper_clusters_kmeans_10_clusters`
        OPTIONS(model_type='kmeans', num_clusters=10) AS
        SELECT 
            total_orders_brand_level,
            brand_level_spend,
            rolling_avg_days_between_orders_brand_level,
            rolling_avg_brand_level_spend_per_order_brand_level,
            total_orders_sku_level,
            sku_level_spend,
            rolling_avg_days_between_orders_sku_level,
            rolling_avg_sku_level_spend_per_order_sku_level
        FROM 
            `gcp-wow-cart-data-dev-d4d7.davide.shopper_features`
        WHERE booking_id = current_campaign_global_var;


    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
        SELECT  
            current_campaign_global_var AS campaign_id,
            "10" AS query_step,
            "K-Means Model Training" AS query_type,
            query_start_time, 
            query_end_time, 
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
            DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.shopper_features_with_clusters` WHERE booking_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.shopper_features_with_clusters`
        SELECT
            *
        FROM 
            ML.PREDICT(MODEL `gcp-wow-cart-data-dev-d4d7.davide.shopper_clusters_kmeans_10_clusters`,
                    (SELECT * FROM `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history` WHERE booking_id = current_campaign_global_var));

    -- DOWNSAMPLING 
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.cluster_cohort_counts` WHERE booking_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cluster_cohort_counts`
        SELECT
            current_campaign_global_var AS booking_id,
            CENTROID_ID,
            cohort,
            COUNT(DISTINCT exposure_crn) AS customer_count
        FROM
            `gcp-wow-cart-data-dev-d4d7.davide.shopper_features_with_clusters`
        WHERE booking_id = current_campaign_global_var
        GROUP BY
            CENTROID_ID, cohort;
    
        DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_with_random_assignment` WHERE booking_id = current_campaign_global_var;
        INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_with_random_assignment`
            with step_one AS (
                SELECT DISTINCT
                    CENTROID_ID,
                    exposure_crn,
                    booking_id,
                    cohort
                FROM
                    `gcp-wow-cart-data-dev-d4d7.davide.shopper_features_with_clusters`
                WHERE booking_id = current_campaign_global_var
            ) SELECT *, RAND() AS random_value FROM step_one;

        
        DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_ranked` WHERE booking_id = current_campaign_global_var;
        INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_ranked`
            SELECT
                a.CENTROID_ID,
                a.exposure_crn,
                a.booking_id,
                a.cohort,
                a.random_value,
                ROW_NUMBER() OVER (PARTITION BY a.CENTROID_ID, a.cohort ORDER BY a.random_value) AS rank
            FROM
                `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_with_random_assignment` a
            WHERE booking_id = current_campaign_global_var;

        
        DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_sampled` WHERE booking_id = current_campaign_global_var;
        INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_sampled`
            -- Calculate the total customer count
            WITH total_customer_count AS (
                SELECT booking_id, COUNT(DISTINCT exposure_crn) AS total_count
                FROM `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_ranked`
                WHERE booking_id = current_campaign_global_var
                GROUP BY 1
            ),
            -- Calculate the desired percentage of each CENTROID_ID-cohort pairing
            desired_percentages AS (
                SELECT 
                    booking_id,
                    CENTROID_ID,
                    cohort,
                    customer_count,
                    customer_count / (SELECT total_count FROM total_customer_count) AS percentage
                FROM 
                    `gcp-wow-cart-data-dev-d4d7.davide.cluster_cohort_counts`
                WHERE booking_id = current_campaign_global_var
            ),
            -- Calculate the sample size for each CENTROID_ID-cohort pairing
            sample_sizes AS (
                SELECT 
                    booking_id,
                    CENTROID_ID,
                    cohort,
                    customer_count,
                    CASE 
                        WHEN customer_count > 6000 THEN 6000
                        ELSE customer_count
                    END AS max_customers,
                    FLOOR(CASE 
                        WHEN customer_count > 6000 THEN 6000 * percentage
                        ELSE customer_count
                    END) AS sample_size
                FROM 
                    desired_percentages
            ),
            -- Perform stratified sampling
            stratified_sample AS (
                SELECT
                    a.CENTROID_ID,
                    a.exposure_crn,
                    a.booking_id,
                    a.cohort,
                    a.rank,
                    b.sample_size
                FROM
                    `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_ranked` a
                JOIN
                    sample_sizes b
                ON
                    a.CENTROID_ID = b.CENTROID_ID
                    AND a.cohort = b.cohort
                    AND a.booking_id = b.booking_id
                WHERE
                    a.booking_id = current_campaign_global_var
                    AND a.rank <= b.sample_size
            )
            SELECT
                CENTROID_ID,
                exposure_crn,
                booking_id,
                cohort
            FROM
                stratified_sample
            WHERE
                booking_id = current_campaign_global_var;

        INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
            SELECT  
                current_campaign_global_var AS campaign_id,
                "12" AS query_step,
                "K-Means Clustering & Stratified Sampling" AS query_type,
                query_start_time, 
                query_end_time, 
                DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
                DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
            ;

        DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history_test_vs_control_step_one` WHERE booking_id = current_campaign_global_var;
        INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history_test_vs_control_step_one`
            WITH exposed_step AS (
                SELECT full_set.*, downsampled.CENTROID_ID
                FROM  `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history` full_set 
                INNER JOIN `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_sampled` downsampled 
                    ON full_set.booking_id = downsampled.booking_id 
                    AND full_set.exposure_crn = downsampled.exposure_crn 
                WHERE 
                full_set.booking_id = current_campaign_global_var
                AND downsampled.exposure_crn IS NOT NULL
                AND full_set.cohort = "Exposed"
            ),
            non_exposed_step AS (
                SELECT full_set.*, downsampled.CENTROID_ID
                FROM  `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history` full_set
                INNER JOIN `gcp-wow-cart-data-dev-d4d7.davide.clustered_data_sampled` downsampled 
                    ON full_set.booking_id = downsampled.booking_id 
                    AND full_set.exposure_crn = downsampled.exposure_crn 
                WHERE full_set.cohort = "Not Exposed"
                AND downsampled.exposure_crn IS NOT NULL
                AND full_set.booking_id = current_campaign_global_var
            )
            SELECT 
                exposed_step.*, 
                non_exposed_step.exposure_crn AS candidate_comparison_shopper,
                non_exposed_step.total_orders_brand_level AS total_orders_brand_level_ccs,
                non_exposed_step.brand_level_spend AS brand_level_spend_ccs,
                non_exposed_step.rolling_avg_days_between_orders_brand_level AS rolling_avg_days_between_orders_brand_level_ccs,
                non_exposed_step.rolling_avg_brand_level_spend_per_order_brand_level AS rolling_avg_brand_level_spend_per_order_brand_level_ccs, 
                non_exposed_step.total_orders_sku_level AS total_orders_sku_level_ccs,
                non_exposed_step.sku_level_spend AS sku_level_spend_ccs, 
                non_exposed_step.rolling_avg_days_between_orders_sku_level AS rolling_avg_days_between_orders_sku_level_ccs,
                non_exposed_step.rolling_avg_sku_level_spend_per_order_sku_level AS rolling_avg_sku_level_spend_per_order_sku_level_ccs,
                ARRAY(SELECT x FROM UNNEST([exposed_step.exposure_crn, non_exposed_step.exposure_crn]) AS x ORDER BY x) AS sorted_crn_array
            FROM exposed_step
            CROSS JOIN non_exposed_step 
                
            WHERE exposed_step.exposure_crn IS NOT NULL 
            AND exposed_step.booking_id = non_exposed_step.booking_id 
            AND exposed_step.CENTROID_ID = non_exposed_step.CENTROID_ID
            AND exposed_step.exposure_crn <> non_exposed_step.exposure_crn
            AND non_exposed_step.exposure_crn IS NOT NULL;

        DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history_test_vs_control_step_two` WHERE booking_id = current_campaign_global_var;
        INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history_test_vs_control_step_two`
        WITH sorted_crn_string AS (
            SELECT * EXCEPT(sorted_crn_array), ARRAY_TO_STRING(sorted_crn_array, ", ") AS sorted_crn_string_array
            FROM `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history_test_vs_control_step_one` 
            WHERE booking_id = current_campaign_global_var
        ),
        unique_sorted_crn_string AS (
            SELECT DISTINCT * FROM sorted_crn_string
        ),
        min_diff AS (
            SELECT 
                *,
                ABS(total_orders_brand_level_ccs - total_orders_brand_level) AS total_orders_brand_level_diff,
                ABS(brand_level_spend_ccs - brand_level_spend) AS brand_level_spend_diff, 
                ABS(rolling_avg_days_between_orders_brand_level_ccs - rolling_avg_days_between_orders_brand_level) AS rolling_avg_days_between_orders_brand_level_diff, 
                ABS(rolling_avg_brand_level_spend_per_order_brand_level_ccs - rolling_avg_brand_level_spend_per_order_brand_level) AS rolling_avg_brand_level_spend_per_order_brand_level_diff, 
                ABS(total_orders_sku_level_ccs - total_orders_sku_level) AS total_orders_sku_level_diff, 
                ABS(sku_level_spend_ccs - sku_level_spend) AS sku_level_spend_diff, 
                ABS(rolling_avg_days_between_orders_sku_level_ccs - rolling_avg_days_between_orders_sku_level) AS rolling_avg_days_between_orders_sku_level_diff, 
                ABS(rolling_avg_sku_level_spend_per_order_sku_level_ccs - rolling_avg_sku_level_spend_per_order_sku_level) AS rolling_avg_sku_level_spend_per_order_sku_level_diff 
            FROM unique_sorted_crn_string
        ),
        ranked_diff AS (
            SELECT 
                *,
                ROW_NUMBER() OVER(PARTITION BY booking_id, exposure_crn ORDER BY total_orders_brand_level_diff) AS total_orders_brand_level_diff_rank,
                ROW_NUMBER() OVER(PARTITION BY booking_id, exposure_crn ORDER BY brand_level_spend_diff) AS brand_level_spend_diff_rank,
                ROW_NUMBER() OVER(PARTITION BY booking_id, exposure_crn ORDER BY rolling_avg_days_between_orders_brand_level_diff) AS rolling_avg_days_between_orders_brand_level_diff_rank,
                ROW_NUMBER() OVER(PARTITION BY booking_id, exposure_crn ORDER BY rolling_avg_brand_level_spend_per_order_brand_level_diff) AS rolling_avg_brand_level_spend_per_order_brand_level_diff_rank,
                ROW_NUMBER() OVER(PARTITION BY booking_id, exposure_crn ORDER BY total_orders_sku_level_diff) AS total_orders_sku_level_diff_rank,
                ROW_NUMBER() OVER(PARTITION BY booking_id, exposure_crn ORDER BY sku_level_spend_diff) AS sku_level_spend_diff_rank,
                ROW_NUMBER() OVER(PARTITION BY booking_id, exposure_crn ORDER BY rolling_avg_days_between_orders_sku_level_diff) AS rolling_avg_days_between_orders_sku_level_diff_rank,
                ROW_NUMBER() OVER(PARTITION BY booking_id, exposure_crn ORDER BY rolling_avg_sku_level_spend_per_order_sku_level_diff) AS rolling_avg_sku_level_spend_per_order_sku_level_diff_rank
            FROM min_diff
        ),
        rank_sum AS (
            SELECT 
                *,
                total_orders_brand_level_diff_rank + brand_level_spend_diff_rank + rolling_avg_days_between_orders_brand_level_diff_rank + rolling_avg_brand_level_spend_per_order_brand_level_diff_rank + 
                total_orders_sku_level_diff_rank + sku_level_spend_diff_rank + rolling_avg_days_between_orders_sku_level_diff_rank + rolling_avg_sku_level_spend_per_order_sku_level_diff_rank 
                AS sum_of_diff_ranks
            FROM ranked_diff 
        )
        SELECT * FROM rank_sum
    ;

    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history_test_vs_control_step_three` WHERE booking_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history_test_vs_control_step_three`
        with min_diffs AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (PARTITION BY booking_id, CENTROID_ID, exposure_crn ORDER BY sum_of_diff_ranks) AS row_num_test 
            FROM `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history_test_vs_control_step_two`
            WHERE booking_id = current_campaign_global_var
            ORDER BY sum_of_diff_ranks
        )
        SELECT 
            *
        FROM min_diffs 
        WHERE row_num_test = 1
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
    SELECT  
        current_campaign_global_var AS campaign_id,
        "11" AS query_step,
        "Clustered Test & Control Pairwise Matching" AS query_type,
        query_start_time, 
        query_end_time, 
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;


    SET query_start_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    DELETE FROM `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_uplift_test_vs_control_shopper_level` WHERE booking_id = current_campaign_global_var;
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_uplift_test_vs_control_shopper_level`
        SELECT 
            final_pairs.booking_id, 

            final_pairs.exposure_crn, 
            total_exposed_spend_sku_level_7_day,
            total_exposed_spend_brand_level_7_day,
            
            final_pairs.candidate_comparison_shopper, 
            total_not_exposed_spend_sku_level_7_day,
            total_not_exposed_spend_brand_level_7_day,

            total_exposed_spend_sku_level_7_day - total_not_exposed_spend_sku_level_7_day AS total_spend_sku_level_7_day_uplift,
            total_exposed_spend_brand_level_7_day - total_not_exposed_spend_brand_level_7_day AS total_spend_brand_level_7_day_uplift

        FROM `gcp-wow-cart-data-dev-d4d7.davide.ces_exposure_plus_history_test_vs_control_step_three` final_pairs 
        
        -- Exposed/Test Shopper - Sku Level Sales, 7 Days
        LEFT JOIN (
            SELECT 
                booking_id,
                converted_shopper_identification_number, 
                SUM(tot_net_incld_gst) AS total_exposed_spend_sku_level_7_day
            FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_sales_7_days_sku_level_v2`
            GROUP BY 1,2
        )  exposed_campaign_conversions_sku_7_day
            ON final_pairs.booking_id = exposed_campaign_conversions_sku_7_day.booking_id 
            AND final_pairs.exposure_crn = exposed_campaign_conversions_sku_7_day.converted_shopper_identification_number 


        -- Not Exposed/Control Shopper - Sku Level Sales, 7 Days
        LEFT JOIN (
            SELECT 
                booking_id,
                converted_shopper_identification_number, 
                SUM(tot_net_incld_gst) AS total_not_exposed_spend_sku_level_7_day
            FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_sales_7_days_sku_level_v2`
            GROUP BY 1,2
        )  not_exposed_campaign_conversions_sku_7_day
            ON final_pairs.booking_id = not_exposed_campaign_conversions_sku_7_day.booking_id 
            AND final_pairs.candidate_comparison_shopper = not_exposed_campaign_conversions_sku_7_day.converted_shopper_identification_number 


        -- Exposed/Test Shopper - Brand Level Sales, 7 Days
        LEFT JOIN (
            SELECT 
                booking_id,
                converted_shopper_identification_number, 
                SUM(tot_net_incld_gst) AS total_exposed_spend_brand_level_7_day
            FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_sales_7_days_brand_level_v2` 
            WHERE exposure_to_purchase_window = "7 Days" 
            GROUP BY 1,2
        )  exposed_campaign_conversions_brand_7_day
            ON final_pairs.booking_id = exposed_campaign_conversions_brand_7_day.booking_id 
            AND final_pairs.exposure_crn = exposed_campaign_conversions_brand_7_day.converted_shopper_identification_number 



        -- Not Exposed/Control Shopper - Brand Level Sales, 7 Days
        LEFT JOIN (
            SELECT 
                booking_id,
                converted_shopper_identification_number, 
                SUM(tot_net_incld_gst) AS total_not_exposed_spend_brand_level_7_day
            FROM `gcp-wow-cart-data-dev-d4d7.davide.campaign_exposed_sales_7_days_brand_level_v2` 
            WHERE exposure_to_purchase_window = "7 Days" 
            GROUP BY 1,2
        )  not_exposed_campaign_conversions_brand_7_day
            ON final_pairs.booking_id = not_exposed_campaign_conversions_brand_7_day.booking_id 
            AND final_pairs.candidate_comparison_shopper = not_exposed_campaign_conversions_brand_7_day.converted_shopper_identification_number 

        WHERE final_pairs.booking_id = current_campaign_global_var
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
    SELECT  
        current_campaign_global_var AS campaign_id,
        "13" AS query_step,
        "Shopper Level Test vs Control Summary" AS query_type,
        query_start_time, 
        query_end_time, 
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) AS query_duration_in_seconds,
        DATETIME_DIFF(query_end_time, query_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;

    SET query_end_time = DATETIME(CURRENT_TIMESTAMP(), 'Australia/Sydney');
    INSERT INTO `gcp-wow-cart-data-dev-d4d7.davide.cartology_dx_logs`
    SELECT  
        current_campaign_global_var AS campaign_id,
        "14" AS query_step,
        "Total Campaign Analysis Runtime" AS query_type,
        query_start_time, 
        query_end_time, 
        DATETIME_DIFF(query_end_time, campaign_run_start_time, SECOND) AS query_duration_in_seconds,
        DATETIME_DIFF(query_end_time, campaign_run_start_time, SECOND) / 60 AS query_duration_in_minutes
    ;



  SET start_index = start_index + 1;
END LOOP;