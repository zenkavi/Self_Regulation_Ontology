from selfregulation.utils.results_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

fmri_contrasts = ['ANT_orienting_network',
 'ANT_conflict_network',
 'ANT_response_time',
 'CCTHot_EV',
 'CCTHot_risk',
 'CCTHot_response_time',
 'discountFix_subjective_value',
 'discountFix_LL_vs_SS',
 'discountFix_response_time',
 'DPX_BX-BY',
 'DPX_AY-BY',
 'DPX_AY-BX',
 'DPX_BX-AY',
 'DPX_response_time',
 'motorSelectiveStop_crit_stop_success-crit_go',
 'motorSelectiveStop_crit_stop_failure-crit_go',
 'motorSelectiveStop_crit_go-noncrit_nosignal',
 'motorSelectiveStop_noncrit_signal-noncrit_nosignal',
 'motorSelectiveStop_crit_stop_success-crit_stop_failure',
 'motorSelectiveStop_crit_stop_failure-crit_stop_success',
 'motorSelectiveStop_crit_stop_success-noncrit_signal',
 'motorSelectiveStop_crit_stop_failure-noncrit_signal',
 'stroop_incongruent-congruent',
 'stroop_response_time',
 'surveyMedley_response_time',
 'twoByTwo_cue_switch_cost_100',
 'twoByTwo_cue_switch_cost_900',
 'twoByTwo_task_switch_cost_100',
 'twoByTwo_task_switch_cost_900',
 'twoByTwo_response_time',
 'WATT3_search_depth']
 
 
 fmri_ontology_mapping = {
     'ANT_conflict_network': 'attention_network_task.conflict_hddm_drift',
     'ANT_orienting_network': 'attention_network_task.orienting_hddm_drift',
     'DPX_AY-BY': 'dot_pattern_expectancy.AY-BY_hddm_drift',
     'DPX_BX-BY': 'dot_pattern_expectancy.BX-BY_hddm_drift',
     'stroop_incongruent-congruent': 'stroop.stroop_hddm_drift',
     'twoByTwo_cue_switch_cost_100': 'threebytwo.cue_switch_cost_hddm_drift',
     'twoByTwo_task_switch_cost_100': 'threebytwo.task_switch_cost_hddm_drift'
 }

dataset = get_recent_dataset()