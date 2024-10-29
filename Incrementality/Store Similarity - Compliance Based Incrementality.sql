



-- Store Similarity Ranking for Cartology Incrementality Compliance Based Retro

CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_store_comparison_of_campaign_impact_step_one AS
    with step_one AS (
        SELECT 
            logs.campaign_id,
            MAX(query_step) AS latest_step, 
            MIN(query_start_time) AS run_start_time,
            MAX(query_end_time) AS run_end_time, 
            MAX(query_duration_in_seconds) AS query_duration_in_seconds,
            MAX(query_duration_in_minutes) AS query_duration_in_minutes,
        FROM `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_run_logs` logs
        WHERE query_end_time >= "2024-06-09 18:07:39.062239"
        AND query_step = "7"
        GROUP BY 1
    )

    SELECT 
        pre.campaign_id, 
        campaigns.campaign_start_date,
        campaigns.campaign_end_date,
        pre.store_id, 
        pre.subcats,
        pre.media_types_string, 
        pre.compliant_media_types,
        pre.non_compliant_media_types, 
        ARRAY_TO_STRING(pre.campaign_skus, ", ") AS campaign_skus, 
        
        pre.n_days_pre_period,
        during.n_days_campaign_period,
        post.n_days_post_period,
        
        pre.total_sales_pre_period, 
        during.total_sales_campaign_period,
        post.total_sales_post_period,
        
        pre.total_baskets_pre_period,
        during.total_baskets_campaign_period,
        post.total_baskets_post_period,

        SAFE_DIVIDE(pre.total_sales_pre_period, 7) AS pre_period_daily_avg_sales, 
        SAFE_DIVIDE(during.total_sales_campaign_period, 7) AS campaign_period_daily_avg_sales,
        SAFE_DIVIDE(post.total_sales_post_period, 7) AS post_period_daily_avg_sales, 

        
        
        SAFE_DIVIDE(pre.total_baskets_pre_period, 7) AS pre_period_daily_avg_baskets,  
        SAFE_DIVIDE(during.total_baskets_campaign_period, 7) AS campaign_period_daily_avg_baskets, 
        SAFE_DIVIDE(post.total_baskets_post_period, 7) AS post_period_daily_avg_baskets,

        all_baskets_with_or_without_promoted_skus_pre_campaign,
        all_baskets_with_or_without_promoted_skus_during_campaign,
        all_baskets_with_or_without_promoted_skus_post_campaign, 

        SAFE_DIVIDE(total_baskets_pre_period , all_baskets_with_or_without_promoted_skus_pre_campaign) AS perc_baskets_with_item_pre_campaign,
        SAFE_DIVIDE(total_baskets_campaign_period , all_baskets_with_or_without_promoted_skus_during_campaign) AS perc_baskets_with_item_during_campaign,
        SAFE_DIVIDE(total_baskets_post_period , all_baskets_with_or_without_promoted_skus_post_campaign) AS perc_baskets_with_item_post_campaign

    FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_four_pre_campaign_sales pre

    LEFT JOIN (SELECT DISTINCT booking_number, campaign_start_date, campaign_end_date FROM `gcp-wow-cart-data-prod-d710.cdm.dim_cartology_campaigns`) campaigns 
        ON pre.campaign_id = campaigns.booking_number 
        AND pre.campaign_start_date = campaigns.campaign_start_date     

    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_five_during_campaign_sales during 
        ON pre.campaign_id = during.campaign_id 
        AND pre.store_id = during.store_id

    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_step_six_post_campaign_sales post 
        ON pre.campaign_id = post.campaign_id 
        AND pre.store_id = post.store_id


    WHERE pre.campaign_id IN (SELECT DISTINCT campaign_id FROM step_one)
    AND pre.n_days_pre_period = 7 
    AND during.n_days_campaign_period = 7
;



CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_store_comparison_of_prior_campaign_impact_medians AS (
    with step_one AS (
        SELECT 
            *, 
            PERCENTILE_CONT(n_comparison_campaign_ids, 0.5) OVER (PARTITION BY campaign_id, store_id) AS median_comparison_campaign_ids
        FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_store_comparison_of_prior_campaign_impact
        WHERE avg_of_abs_diff_in_perc_change_in_sales IS NOT NULL
    ),
    step_two AS (
    SELECT 
        *,
        ROW_NUMBER() OVER(PARTITION BY store_id, campaign_id ORDER BY avg_of_abs_diff_in_perc_change_in_sales) AS similarity_ranking
    FROM step_one 
    WHERE n_comparison_campaign_ids >= median_comparison_campaign_ids
    )
    SELECT 
        campaign_id, 
        campaign_start_date, 
        store_id, 
        comparison_store_id, 
        n_comparison_campaign_ids, 
        median_comparison_campaign_ids, 
        avg_of_abs_diff_in_perc_change_in_sales, 
        avg_of_abs_diff_in_perc_change_in_transaction_share, 
        similarity_ranking
    FROM step_two 
    ORDER BY campaign_id, store_id, similarity_ranking 
);


CREATE OR REPLACE TABLE `gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_store_comparison_of_prior_campaign_impact_medians_compliance` AS (
    SELECT 
        meds.*,
        store_compliance.subcats,
        store_compliance.campaign_skus,
        store_compliance.compliant_media_types,
        store_compliance.non_compliant_media_types,
        comparison_store_compliance.compliant_media_types AS comparison_compliant_media_types,
        comparison_store_compliance.non_compliant_media_types AS comparison_non_compliant_media_types
    FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_store_comparison_of_prior_campaign_impact_medians meds
    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_store_comparison_of_campaign_impact_step_one store_compliance 
        ON meds.campaign_id = store_compliance.campaign_id 
        AND meds.store_id = store_compliance.store_id 
    LEFT JOIN gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_store_comparison_of_campaign_impact_step_one comparison_store_compliance 
        ON meds.campaign_id = comparison_store_compliance.campaign_id 
        AND meds.comparison_store_id = comparison_store_compliance.store_id
);



CREATE OR REPLACE TABLE gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_campaign_comparison_results AS (
    with step_one AS (
        SELECT 
            compliance.*,
            media_split_cohorts.test_group,
            media_split_cohorts.media_types_string
        FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_store_comparison_of_prior_campaign_impact_medians_compliance compliance
        LEFT JOIN (
            SELECT DISTINCT 
                campaign_id, 
                store_id, 
                CONCAT("GROUP", media_split_cohort) AS test_group,
                media_types_string 
            FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_compliant_cohorts_and_stockouts_since_2023
        ) media_split_cohorts 
            ON compliance.campaign_id = media_split_cohorts.campaign_id 
            AND compliance.store_id = media_split_cohorts.store_id  

        LEFT JOIN (
            SELECT DISTINCT 
                campaign_id, 
                store_id, 
                CONCAT("GROUP", media_split_cohort) AS comparison_test_group,
                media_types_string 
            FROM gcp-wow-cart-data-dev-d4d7.davide.cartology_incrementality_retro_analysis_compliant_cohorts_and_stockouts_since_2023
        ) comparison_media_split_cohorts 
            ON compliance.campaign_id = comparison_media_split_cohorts.campaign_id 
            AND compliance.comparison_store_id = comparison_media_split_cohorts.store_id  
       
        WHERE 
            compliant_media_types <> comparison_compliant_media_types
        AND 
            media_split_cohorts.test_group = comparison_media_split_cohorts.comparison_test_group
        ORDER BY campaign_id, store_id, similarity_ranking
    ) 
    , step_two AS (
        SELECT 
            campaign_id, 
            store_id, 
            MIN(similarity_ranking) AS most_similar_rank
        FROM step_one 
        GROUP BY 1,2
    )
    SELECT 
        step_one.*
    FROM step_one 
    INNER JOIN step_two 
        ON step_two.campaign_id = step_one.campaign_id 
        AND step_two.store_id = step_one.store_id 
        AND step_two.most_similar_rank = step_one.similarity_ranking
);