-- ######################################################################################################################################## 
-- All commented out sections have already been run and dont need to be re-run unless expanding date ranges 



    CREATE OR REPLACE TABLE `gcp-wow-cart-data-dev-d4d7.davide.test_transactions` AS (
        SELECT
            "Instore Order" AS order_context,
            start_txn_date,
            start_txn_time,
            "CRN" as shopper_identification_method,
            lylty_card_detail.crn AS shopper_identification_number,
            article_sales_summary.Article AS article,
            am.BrandDescription AS brand,
            am.SubcatDescription,
            article_sales_summary.prod_nbr,
            basket_key,
            tot_net_incld_gst
        
        -- transaction table
        FROM `gcp-wow-food-wlx-digaspt-dev.wdp_tables.article_sales_summary` AS article_sales_summary

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
        ) am ON am.Article = article_sales_summary.Article
        
        -- loyalty card
        LEFT JOIN `gcp-wow-food-wlx-digaspt-dev.wdp_tables.lylty_card_detail` AS lylty_card_detail
            ON article_sales_summary.lylty_card_nbr = lylty_card_detail.lylty_card_nbr
        

        WHERE
            article_sales_summary.start_txn_date >= DATE("2022-10-01")
        AND 
            article_sales_summary.start_txn_date <= CURRENT_DATE("Australia/Sydney")
        AND 
            LENGTH(article_sales_summary.lylty_card_nbr) > 3 
        AND 
            LENGTH(lylty_card_detail.crn) > 3 
        AND
            article_sales_summary.division_nbr IN (1005, 1030)
        AND 
            LOWER(article_sales_summary.void_flag) = 'n' 
        AND 
            LOWER(article_sales_summary.SalesChannelDescription) <> "online"
        
        
        UNION ALL 

        SELECT 
            "Online Order" AS order_context,
            CAST(oh.DisColDateTime AS DATE) AS start_txn_date,
            od.CreateDateTime AS start_txn_time, 
            "CRN" AS shopper_identification_method, 
            crn_map.CustomerRegistrationNumber AS shopper_identification_number,      
            CAST(od.StockCode as STRING) AS article,
            am2.BrandDescription AS brand,
            am2.SubcatDescription,
            "" AS prod_nbr, 
            CAST(OriginalOrderNumber AS STRING) AS basket_key, 
            od.OrderLineAmount as tot_net_incld_gst
    FROM `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_online_view_smkt.ecf_smkt_order_header_v` as oh
    INNER JOIN `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_online_view_smkt.ecf_smkt_order_detail_v` as od
        ON oh.OrderNumber = od.OrderNumber
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
        ) am2 ON am2.Article = CAST(od.StockCode as STRING)
    LEFT JOIN `gcp-wow-ent-im-wowx-cust-prod.adp_wowx_dm_customer_view.customer_online_shopper_v` crn_map 
        ON CAST(crn_map.ShopperID AS STRING) = CAST(oh.CustomerNumber AS STRING)
    WHERE CAST(oh.DisColDateTime as DATE) >= DATE("2022-10-01")

);


CREATE OR REPLACE TABLE `gcp-wow-cart-data-dev-d4d7.davide.carto_campaigns`  AS (
    with step_one AS (
        SELECT
            booking_number,
            campaign_start_date,
            campaign_end_date,
            line_name,
            CASE
                WHEN media_location LIKE ('%TV%') THEN 'Mass_Social_Media'
                WHEN media_type IN (
                    'Counter Card',
                    'POC',
                    'Spotify',
                    'YouTube',
                    'Social',
                    'POC',
                    'Social - Wrapper',
                    'Build Capacity'
                ) THEN 'Mass_Social_Media'
                WHEN media_location IN ('Category', 'Aisle', 'Store', 'Screens', 'Emma Reach') THEN 'in-store'
                WHEN media LIKE ('%FB Post%') THEN 'Mass_Social_Media'
                WHEN media_type = 'Always On eDMs' THEN 'eDM'
                WHEN media_type = 'Single Send eDM' THEN 'eDM'
                WHEN media_type = 'BRANDED SHOP' THEN 'Branded Shop'
                WHEN media_type = 'CONTENT CARD' THEN 'Content Card'
                WHEN media_type = 'Digital Standalone Catalogue' THEN 'Catalogue'
                WHEN media_type = 'Display Recipes - Wrapper' THEN 'Display Recipes'
                WHEN media_type = 'RECIPE PACKAGE' THEN 'Display Recipes'
                WHEN media_type = 'Recipe Package' THEN 'Display Recipes'
                WHEN media_type = 'Standalone Catalogue - Double Spot' THEN 'Catalogue'
                WHEN media_type = 'Standalone Catalogue - Single Spot' THEN 'Catalogue'
                WHEN media_type = 'Standalone Catalogue- Third Page Vertical' THEN 'Catalogue'
                WHEN media_type = 'Catalogue Card' THEN 'Catalogue'
                WHEN media_type = 'BF Meal' THEN 'Others'
                WHEN media_type = 'Sampling' THEN 'Others'
                WHEN media_type = 'Bus Stop' THEN 'in-store'
                WHEN media_type = 'Display Promo Tile' THEN 'in-store'
                WHEN media_type = 'Aisle Fin' THEN 'in-store'
                WHEN media_type = 'Additional Production Fees' THEN 'Others'
                WHEN media_type = 'Event' THEN 'Others'
                WHEN media_type = 'National Radio' THEN 'Mass_Social_Media'
                WHEN media_type = 'Chilled Bus Stop & Decal Package' THEN 'in-store'
                WHEN media_type = 'National TV' THEN 'Mass_Social_Media'
                WHEN media_type = 'Press' THEN 'Mass_Social_Media'
                WHEN media_type = 'Content Card' THEN 'Others'
                WHEN media_type = 'Competition Management Fee' THEN 'Others'
                WHEN media_type = 'Prize pool' THEN 'Others'
                WHEN media_type = 'Chilled Bus Stop & Decal Package' THEN 'in-store'
                WHEN media_type = 'Chilled Fin Package' THEN 'in-store'
                WHEN media_type = 'Editorial' THEN 'Mass_Social_Media'
                WHEN media_type = 'BUS STOP' THEN 'in-store'
                WHEN media_type = 'Door Take' THEN 'in-store'
                WHEN media_type = 'Freezer Package' THEN 'in-store'
                WHEN media_type = 'PL - Mag' THEN 'Others'
                WHEN media_type = 'PL - Fresh Mag' THEN 'Others'
                WHEN media_type = 'PL - Demos' THEN 'Others'
                WHEN media_type = 'Pelmet' THEN 'in-store'
                WHEN media_type = 'Overlay Card' THEN 'in-store'
                WHEN media_type = 'Mass/Social Media' THEN 'Others'
                WHEN lower(media_type) like "%catalogue%" THEN 'Catalogue'
                WHEN LOWER(media_type) LIKE "%homepage%" THEN "Home Page"
                ELSE
                media_type
            END
            AS media_type,
            individual_product_string,
            MIN(media_start_date) AS media_start_date,
            MAX(media_end_date) AS media_end_date,
            SUM(quantity) * SUM(media_spend) AS total_media_spend

        FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns`,
        UNNEST(SPLIT(quoteline_sku, ',')) AS individual_product_string

        WHERE
            
            business_unit = 'Supermarket'
            AND booking_number IS NOT NULL
            AND quote_status NOT IN ('Not Approved','Draft','For Review','Sent')
            AND media_location NOT IN ('Category','Aisle','Store','Screens','Emma Reach')
            AND media_location NOT LIKE ('%TV%')
            AND media_type NOT IN ('Counter Card','POC','Spotify','YouTube','Social','POC','Social - Wrapper')
            AND (media_type <> 'Event')
            AND (media_location <> 'None' OR media LIKE ('%FB Post%') OR media LIKE ('%Branded%') OR media LIKE ('%Segments%'))
            AND media NOT IN ('Competition Management Fee','Additional Production Fees')
        AND media_start_date >= DATE("2023-10-01")
        AND media_end_date < CURRENT_DATE("Australia/Sydney")
        GROUP BY
            1,2,3,4,5,6
    ) 

    SELECT * 
    FROM step_one
    WHERE
        media_type NOT IN ('in-store','Others','eDM')
        AND media_type NOT LIKE ('%Woolworths Rewards%')
        AND media_type NOT LIKE ('%SAMPLING%') 
        AND media_type NOT LIKE "%Road Block%"
        AND media_type <> 'Radio'
        AND media_type <> 'Mass_Social_Media'
        AND LOWER(media_type) NOT LIKE "%branded shop%"

);

-- 2. Adobe Events
CREATE OR REPLACE TABLE `gcp-wow-cart-data-dev-d4d7.davide.adobe_carto_catalogue_homepage_ciid_search_term_test`  AS (
    with 
    
    cartology_campaign_asset_impressions_in_adobe AS (
        SELECT 
            date_time,
            shopper_id_map.shopper_id, 
            crn_map.crn, 
            post_evar6 AS page_name,
            CASE WHEN SPLIT(post_evar6,":")[SAFE_OFFSET(2)] = 'catalogue' OR LOWER(post_evar7) = 'catalogue' THEN 1 ELSE 0 END AS catalogue_flag,    
            CASE WHEN post_evar6 = 'ww-sm:homepage' THEN 1 ELSE 0 END AS homepage_flag, 
            CASE WHEN post_mvvar2 IS NOT NULL THEN SPLIT(post_mvvar2,":")[SAFE_OFFSET(0)] ELSE "Not Found" END AS booking_id,
            CASE WHEN post_mvvar2 IS NOT NULL THEN SPLIT(post_mvvar2,":")[SAFE_OFFSET(3)] ELSE NULL END AS ci_id,
            post_evar51 AS search_term1,
            post_evar52 AS search_term2,
            post_evar53 AS search_term3
            
        FROM `gcp-wow-food-wlx-digaspt-dev.prod_adobe_data.grs_adobe` adobe

        -- shopper id
        LEFT JOIN `gcp-wow-rwds-ai-mlt-evs-prod.event_store.tealium_visitorid_shopperid_map_window` shopper_id_map
            ON adobe.prop9 = shopper_id_map.visitor_id
            AND adobe.date_time BETWEEN shopper_id_map.effective_time AND shopper_id_map.expiry_time
        
        -- crn
        LEFT JOIN `gcp-wow-rwds-ai-mlt-evs-prod.event_store.tealium_visitorid_crn_map_window` crn_map
        ON adobe.prop9 = crn_map.visitor_id
            AND adobe.date_time BETWEEN crn_map.effective_time and crn_map.expiry_time

        WHERE 
            site_name = "Supermarkets"
        AND 
            DATE(date_time) >= DATE("2023-10-01")
        AND    
            DATE(date_time) < DATE("2023-10-02") --CURRENT_DATE("Australia/Sydney")
        AND         
            (
                -- Catalogue
                (SPLIT(post_evar6,":")[SAFE_OFFSET(2)] = 'catalogue' OR LOWER(post_evar7) = 'catalogue')
                OR 
                -- Homepage
                (post_evar6 = 'ww-sm:homepage') 
                OR 
                -- ci_id 
                (SPLIT(post_mvvar2,":")[SAFE_OFFSET(3)] IS NOT NULL)
            )
        AND      
            post_channel NOT LIKE "Order Confirmation Section"
        AND 
            (shopper_id_map.shopper_id IS NOT NULL OR crn_map.crn IS NOT NULL)
    
    ) 

    SELECT * FROM cartology_campaign_asset_impressions_in_adobe 

);


CREATE OR REPLACE TABLE `gcp-wow-cart-data-dev-d4d7.davide.adobe_carto_catalogue_homepage_ciid_control`AS (
        SELECT DISTINCT
            DATE(date_time) date,
            shopper_id_map.shopper_id, 
            crn_map.crn
            
        FROM `gcp-wow-food-wlx-digaspt-dev.prod_adobe_data.grs_adobe` adobe

        -- shopper id
        LEFT JOIN `gcp-wow-rwds-ai-mlt-evs-prod.event_store.tealium_visitorid_shopperid_map_window` shopper_id_map
            ON adobe.prop9 = shopper_id_map.visitor_id
            AND adobe.date_time BETWEEN shopper_id_map.effective_time AND shopper_id_map.expiry_time
        
        -- crn
        LEFT JOIN `gcp-wow-rwds-ai-mlt-evs-prod.event_store.tealium_visitorid_crn_map_window` crn_map
            ON adobe.prop9 = crn_map.visitor_id
            AND adobe.date_time BETWEEN crn_map.effective_time and crn_map.expiry_time

        WHERE 
            site_name = "Supermarkets"
        AND 
            DATE(date_time) >= DATE("2023-10-01")
        AND    
            DATE(date_time) < CURRENT_DATE("Australia/Sydney")
        AND         
            (
                -- Catalogue
                (SPLIT(post_evar6,":")[SAFE_OFFSET(2)] = 'catalogue' OR LOWER(post_evar7) = 'catalogue')
                OR 
                -- Homepage
                (post_evar6 = 'ww-sm:homepage') 
                OR 
                -- ci_id 
                (SPLIT(post_mvvar2,":")[SAFE_OFFSET(3)] IS NOT NULL)
            )
        AND      
            post_channel NOT LIKE "Order Confirmation Section"
        AND 
            (shopper_id_map.shopper_id IS NOT NULL OR crn_map.crn IS NOT NULL)

);

-- 3. Adobe Events + Cartology Campaign Info
-- Approx 33mins to run
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

