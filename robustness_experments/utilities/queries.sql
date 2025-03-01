select question_type,model_id,count(uid) cnt_d from model_response_evaluation_r2
group by question_type,model_id;

select question_category,model_name,count(uid) cnt_d from model_responses_r2
group by question_category,model_name;