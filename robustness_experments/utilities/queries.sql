select question_type,model_id,count(uid) cnt_d from model_response_evaluation_r2
group by question_type,model_id;

select question_category,model_name,count(uid) cnt_d from model_responses_r2
group by question_category,model_name;

select a.uid, a.question_id, a.question, a.question_category, a.actual_answer, a.model_name, a.model_answer, a.image_link from model_responses_r2 a
left join model_response_evaluation_r2 b on a.uid=b.uid and a.question_id=b.uid and a.model_name=b.model_id
where b.uid is null;
;

select * from model_response_evaluation_r2 where severity_classification != 'Low Risk' and severity_classification != 'Moderate Risk';
