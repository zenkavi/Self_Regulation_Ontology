/* ************************************ */
/* Define helper functions */
/* ************************************ */
var getInstructFeedback = function() {
	return '<div class = centerbox><p class = center-block-text>' + feedback_instruct_text +
		'</p></div>'
}

var post_trial_gap = function() {
	var curr_trial = jsPsych.progress().current_trial_global
	return 3500 - jsPsych.data.getData()[curr_trial - 1].rt - jsPsych.data.getData()[curr_trial - 4].block_duration
}

var get_RT = function() {
	var curr_trial = jsPsych.progress().current_trial_global
	return jsPsych.data.getData()[curr_trial].rt
}

var getInstructFeedback = function() {
	return '<div class = centerbox><p class = center-block-text>' + feedback_instruct_text +
		'</p></div>'
}



/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// task specific variables
/* set up stim: location (2) * cue (4) * direction (2) * condition (3) */
var locations = ['up', 'down']
var cues = ['nocue', 'center', 'double', 'spatial']
var current_trial = 0
var exp_stage = 'test'
var test_stimuli = []
var choices = [66, 89]
var path = '/static/experiments/attention_network_task/images/'
var images = [path + 'right_arrow.png', path + 'left_arrow.png', path + 'no_arrow.png']
//preload
jsPsych.pluginAPI.preloadImages(images)

for (l = 0; l < locations.length; l++) {
	var loc = locations[l]
	for (ci = 0; ci < cues.length; ci++) {
		var c = cues[ci]
		stims = [{
			stimulus: '<div class = centerbox><div class = ANT_text>+</div></div><div class = ANT_' + loc +
				'><img class = ANT_img src = ' + images[2] + '></img><img class = ANT_img src = ' + images[2] + '></img><img class = ANT_img src = ' + images[1] + '></img><img class = ANT_img src = ' + images[2] + '></img><img class = ANT_img src = ' + images[2] + '></img></div></div>',
			data: {
				correct_response: 37,
				flanker_middle_direction: 'left',
				flanker_type: 'neutral',
				flanker_location: loc,
				cue: c, 
				trial_id: 'stim'
			}
		}, {
			stimulus: '<div class = centerbox><div class = ANT_text>+</div></div><div class = ANT_' + loc +
				'><img class = ANT_img src = ' + images[1] + '></img><img class = ANT_img src = ' + images[1] + '></img><img class = ANT_img src = ' + images[1] + '></img><img class = ANT_img src = ' + images[1] + '></img><img class = ANT_img src = ' + images[1] + '></img></div></div>',
			data: {
				correct_response: 37,
				flanker_middle_direction: 'left',
				flanker_type: 'congruent',
				flanker_location: loc,
				cue: c, 
				trial_id: 'stim'
			}
		}, {
			stimulus: '<div class = centerbox><div class = ANT_text>+</div></div><div class = ANT_' + loc +
				'><img class = ANT_img src = ' + images[0] + '></img><img class = ANT_img src = ' + images[0] + '></img><img class = ANT_img src = ' + images[1] + '></img><img class = ANT_img src = ' + images[0] + '></img><img class = ANT_img src = ' + images[0] + '></img></div></div>',
			data: {
				correct_response: 37,
				flanker_middle_direction: 'left',
				flanker_type: 'incongruent',
				flanker_location: loc,
				cue: c, 
				trial_id: 'stim'
			}
		}, {
			stimulus: '<div class = centerbox><div class = ANT_text>+</div></div><div class = ANT_' + loc +
				'><img class = ANT_img src = ' + images[2] + '></img><img class = ANT_img src = ' + images[2] + '></img><img class = ANT_img src = ' + images[0] + '></img><img class = ANT_img src = ' + images[2] + '></img><img class = ANT_img src = ' + images[2] + '></img></div></div>',
			data: {
				correct_response: 39,
				flanker_middle_direction: 'right',
				flanker_type: 'neutral',
				flanker_location: loc,
				cue: c, 
				trial_id: 'stim'
			}
		}, {
			stimulus: '<div class = centerbox><div class = ANT_text>+</div></div><div class = ANT_' + loc +
				'><img class = ANT_img src = ' + images[0] + '></img><img class = ANT_img src = ' + images[0] + '></img><img class = ANT_img src = ' + images[0] + '></img><img class = ANT_img src = ' + images[0] + '></img><img class = ANT_img src = ' + images[0] + '></img></div></div>',
			data: {
				correct_response: 39,
				flanker_middle_direction: 'right',
				flanker_type: 'congruent',
				flanker_location: loc,
				cue: c, 
				trial_id: 'stim'
			}
		}, {
			stimulus: '<div class = centerbox><div class = ANT_text>+</div></div><div class = ANT_' + loc +
				'><img class = ANT_img src = ' + images[1] + '></img><img class = ANT_img src = ' + images[1] + '></img><img class = ANT_img src = ' + images[0] + '></img><img class = ANT_img src = ' + images[1] + '></img><img class = ANT_img src = ' + images[1] + '></img></div></div>',
			data: {
				correct_response: 39,
				flanker_middle_direction: 'right',
				flanker_type: 'incongruent',
				flanker_location: loc,
				cue: c, 
				trial_id: 'stim'
			}
		}]
		for (i = 0; i < stims.length; i++) {
			test_stimuli.push(stims[i])
		}
	}
}

/* set up repeats for three test blocks */
var block1_trials = jsPsych.randomization.repeat($.extend(true, [], test_stimuli), 1, true);
var block2_trials = jsPsych.randomization.repeat($.extend(true, [], test_stimuli), 1, true);
var block3_trials = jsPsych.randomization.repeat($.extend(true, [], test_stimuli), 1, true);
var blocks = [block1_trials, block2_trials, block3_trials]


/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
/* define static blocks */
 var test_intro_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div style = "font-size: 40px" class = center-text>Starting the test in 5 seconds. Get ready!</p></div>',
	is_html: true,
	choices: 'none',
	timing_stim: 5000, 
	timing_response: 5000,
	data: {
		trial_id: "rest block"
	},
	timing_post_trial: 1000
};

var rest_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div style = "font-size: 40px" class = center-text>Take a break! We will start in 10 seconds.</p></div>',
	is_html: true,
	choices: 'none',
	timing_stim: 10000, 
	timing_response: 1000,
	data: {
		trial_id: "rest block"
	},
	timing_post_trial: 1000
};

var end_block = {
	type: 'poldrack-text',
	text: '<div class = centerbox><p class = center-block-text>Thanks for completing this task!</p><p class = center-block-text>Press <strong>enter</strong> to continue.</p></div>',
	cont_key: [13],
	data: {
		trial_id: "end",
    	exp_id: 'attention_network_task'
	},
	timing_response: 180000,
	timing_post_trial: 0
};

var feedback_instruct_text =
	'Welcome to the experiment. This experiment will take about 15 minutes. Press <strong>enter</strong> to begin.'
var feedback_instruct_block = {
	type: 'poldrack-text',
	cont_key: [13],
	text: getInstructFeedback,
	data: {
		trial_id: 'instruction'
	},
	timing_post_trial: 0,
	timing_response: 180000
};
/// This ensures that the subject does not read through the instructions too quickly.  If they do it too quickly, then we will go over the loop again.
var instructions_block = {
	type: 'poldrack-instructions',
	pages: [
		'<div class = centerbox><p class = block-text>In this experiment you will see groups of five arrows and dashes pointing left or right (e.g &larr; &larr; &larr; &larr; &larr;, or &mdash; &mdash; &rarr; &mdash; &mdash;) presented randomly at the top or bottom of the screen.</p><p class = block-text>Your job is to indicate which way the central arrow is pointing by pressing the corresponding arrow key.</p></p></p></div>',
		'<div class = centerbox><p class = block-text>Before the arrows and dashes come up, an * will occasionally come up somewhere on the screen.</p><p class = block-text>Irrespective of whether or where the * appears, it is important that you respond as quickly and accurately as possible by pressing the arrow key corresponding to the direction of the center arrow.</p><p class = block-text>After you end instructions we will start with practice. During practice you will receive feedback about whether your responses are correct. You will not receive feedback during the rest of the experiment.</p></div>'
	],
	allow_keys: false,
	data: {
		trial_id: 'instruction'
	},
	show_clickable_nav: true,
	timing_post_trial: 1000
};


var fixation = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = ANT_text>+</div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: 'fixation'
	},
	timing_post_trial: 0,
	timing_stim: 400,
	timing_response: 400,
	on_finish: function() {
		jsPsych.data.addDataToLastTrial({
			exp_stage: exp_stage
		})
	}
}

var no_cue = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = ANT_text>+</div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: 'nocue'
	},
	timing_post_trial: 0,
	timing_stim: 100,
	timing_response: 100,
	on_finish: function() {
		jsPsych.data.addDataToLastTrial({
			exp_stage: exp_stage
		})
	}
}

var center_cue = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = ANT_centercue_text>*</div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: 'centercue'
	},
	timing_post_trial: 0,
	timing_stim: 100,
	timing_response: 100,
	on_finish: function() {
		jsPsych.data.addDataToLastTrial({
			exp_stage: exp_stage
		})
	}

}

var double_cue = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = ANT_text>+</div></div><div class = ANT_down><div class = ANT_text>*</div></div><div class = ANT_up><div class = ANT_text>*</div><div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: 'doublecue'
	},
	timing_post_trial: 0,
	timing_stim: 100,
	timing_response: 100,
	on_finish: function() {
		jsPsych.data.addDataToLastTrial({
			exp_stage: exp_stage
		})
	}
}

/* set up ANT experiment */
var attention_network_task_experiment = [];
attention_network_task_experiment.push(instructions_block);
setup_fmri_intro(attention_network_task_experiment, choices)

/* Set up ANT main task */
var trial_num = 0
for (b = 0; b < blocks.length; b++) {
	attention_network_task_experiment.push(test_intro_block);
	var block = blocks[b]
	for (i = 0; i < block.data.length; i++) {
		var trial_num = trial_num + 1
		var first_fixation_gap = Math.floor(Math.random() * 1200) + 400;
		var first_fixation = {
			type: 'poldrack-single-stim',
			stimulus: '<div class = centerbox><div class = ANT_text>+</div></div>',
			is_html: true,
			choices: 'none',
			data: {

				trial_id: "fixation",
				exp_stage: 'test'
			},
			timing_post_trial: 0,
			timing_stim: first_fixation_gap,
			timing_response: first_fixation_gap
		}
		attention_network_task_experiment.push(first_fixation)

		if (block.data[i].cue == 'nocue') {
			attention_network_task_experiment.push(no_cue)
		} else if (block.data[i].cue == 'center') {
			attention_network_task_experiment.push(center_cue)
		} else if (block.data[i].cue == 'double') {
			attention_network_task_experiment.push(double_cue)
		} else {
			var spatial_cue = {
				type: 'poldrack-single-stim',
				stimulus: '<div class = centerbox><div class = ANT_text>+</div></div><div class = centerbox><div class = ANT_' + block.data[i].flanker_location +
					'><div class = ANT_text>*</p></div></div>',
				is_html: true,
				choices: 'none',
				data: {

					trial_id: "spatialcue",
					exp_stage: 'test'
				},
				timing_post_trial: 0,
				timing_stim: 100,
				timing_response: 100
			}
			attention_network_task_experiment.push(spatial_cue)
		}
		attention_network_task_experiment.push(fixation)

		block.data[i].trial_num = trial_num
		var ANT_trial = {
			type: 'poldrack-single-stim',
			stimulus: block.stimulus[i],
			is_html: true,
			choices: choices,
			data: block.data[i],
			timing_response: 1700,
			timing_stim: 1700,
			response_ends_trial: true,
			timing_post_trial: 0,
			on_finish: function(data) {
				correct = data.key_press === data.correct_response
				jsPsych.data.addDataToLastTrial({
					correct: correct,
					exp_stage: exp_stage
				})
			}
		}
		attention_network_task_experiment.push(ANT_trial)

		var last_fixation = {
			type: 'poldrack-single-stim',
			stimulus: '<div class = centerbox><div class = ANT_text>+</div></div>',
			is_html: true,
			choices: 'none',
			data: {

				trial_id: "fixation",
				exp_stage: 'test'
			},
			timing_post_trial: 0,
			timing_stim: post_trial_gap,
			timing_response: post_trial_gap,
		}
		attention_network_task_experiment.push(last_fixation)
	}
	attention_network_task_experiment.push(rest_block)
}
attention_network_task_experiment.push(end_block)