set -e
for exp_id in adaptive_n_back alcohol_drugs_survey angling_risk_task_always_sunny attention_network_task bickel_titrator bis11_survey bis_bas_survey brief_self_control_survey choice_reaction_time cognitive_reflection_survey columbia_card_task_cold columbia_card_task_hot demographics_survey dickman_survey dietary_decision digit_span directed_forgetting discount_titrate dospert_eb_survey dospert_rp_survey dospert_rt_survey dot_pattern_expectancy eating_survey erq_survey five_facet_mindfulness_survey future_time_perspective_survey go_nogo grit_scale_survey hierarchical_rule holt_laury_survey impulsive_venture_survey information_sampling_task k6_survey keep_track kirby leisure_time_activity_survey local_global_letter mindful_attention_awareness_survey motor_selective_stop_signal mpq_control_survey probabilistic_selection psychological_refractory_period_two_choices ravens recent_probes selection_optimization_compensation_survey self_regulation_survey sensation_seeking_survey shape_matching shift_task simon simple_reaction_time spatial_span stim_selective_stop_signal stop_signal stroop ten_item_personality_survey theories_of_willpower_survey threebytwo time_perspective_survey tower_of_london two_stage_decision upps_impulsivity_survey writing_task
do
sed -e "s/{EXP_ID}/$exp_id/g" -e "s/{SUIBSET}/mturk_complete/g" calculate_exp_DVs.batch | sbatch -p russpold
done
