select distinct baseline_question_id,id,variation_text,image_link,strategy from qn_variations_hp;


with hpq as (select id, baseline_question_id, variation_text, strategy, answer, image_link, created_at
             from qn_variations_hp where answer not like '%generation failed%' and model_id='google/medgemma-4b-it'),
     bq as (select id,
                   study_id,
                   subject_id,
                   split,
                   gender,
                   age,
                   race,
                   answer,
                   question_id,
                   question,
                   image,
                   condition_type,
                   attack_category,
                   adversarial_prompt
            from benchmark_qns b
           )
select distinct bq.question_id,hpq.id,
       bq.question        as baseline_question,
       bq.answer             ground_truth,
       hpq.variation_text as variant_question,
       hpq.answer         as variant_answer
       -- hpq.image_link
from hpq
         join bq on hpq.baseline_question_id = bq.question_id order by question_id;