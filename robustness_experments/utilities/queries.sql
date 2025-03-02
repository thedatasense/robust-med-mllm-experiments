select question_type,model_id,count(uid) cnt_d from model_response_evaluation_r2
group by question_type,model_id;

select question_category,model_name,count(uid) cnt_d from model_responses_r2
group by question_category,model_name;

select a.uid, a.question_id, a.question, a.question_category, a.actual_answer, a.model_name, a.model_answer, a.image_link from model_responses_r2 a
left join model_response_evaluation_r2 b on a.uid=b.uid and a.question_id=b.uid and a.model_name=b.model_id
where b.uid is null;
;

select * from model_response_evaluation_r2 where severity_classification != 'Low Risk' and severity_classification != 'Moderate Risk';

SELECT
    model_name,
    strftime('%Y-%m-%d %H:00:00', created_at) as hour,
    COUNT(DISTINCT uid || question_id ||model_name) as unique_uids
FROM
    model_responses_r2
GROUP BY
    model_name,
    strftime('%Y-%m-%d %H:00:00', created_at)
ORDER BY
    hour DESC,
    unique_uids DESC;



SELECT a.id, a.question_id, a.condition as question_type, a.text as question, a.answer as ground_truth, a.image
FROM mimic_all_qns a
          left JOIN model_responses_r2 b
                   ON CAST(a.question_id AS text) = b.question_id
                       AND a.id = b.uid
                       AND b.model_name = 'CheXagent-8b' where b.question_id is null;
