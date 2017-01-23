/* ************************************ */
/* Define helper functions */
/* ************************************ */
var ITIs = [0.2,0.0,0.9,0.1,0.1,0.5,0.3,0.8,0.3,0.2,0.0,0.3,0.2,0.0,0.0,0.2,0.1,0.6,0.2,0.4,0.0,0.3,0.2,0.9,0.1,0.1,0.3,0.0,0.5,0.1,0.7,0.1,0.3,0.0,0.0,0.0,0.3,0.0,0.1,0.0,0.1,0.5,0.3,0.3,0.3,0.0,0.0,0.7,1.5,0.0,0.3,0.0,0.0,0.1,0.2,0.4,0.1,0.0,0.2,0.4,0.2,0.1,0.0,0.2,0.0,0.1,0.0,0.5,0.1,0.1,0.1,0.4,0.2,0.1,0.0,0.1,0.3,0.1,0.4,0.2,0.0,0.2,0.5,0.0,0.0,0.4,0.1,0.2,0.2,0.1,0.2,0.0,0.3,0.0,0.0,0.1,0.3,0.1,0.2,0.0,0.1,0.0,0.1,0.3,0.2,0.5,0.1,0.0,0.0,0.1,0.3,1.0,0.3,0.1,0.4,0.2,0.0,0.0,0.3,0.8,0.0,0.0,0.0,0.0,0.4,0.2,0.1,0.6,0.0,0.1,0.2,0.0,0.1,0.0,0.4,0.6,0.1,0.3,0.0,0.7,0.0,1.0,0.0,0.4,0.2,0.1,0.5,0.1,0.3,0.3,0.4,0.3,0.5,0.8,0.1,0.2,0.1,0.1,0.2,0.2,0.0,0.0,0.1,0.0,0.0,0.2,0.4,0.0,0.2,0.1,0.2,0.5,0.0,0.0,0.5,0.0,0.0,0.4,1.0,0.2,0.1,0.0,0.0,0.0,0.2,0.0,0.0,0.1,0.0,0.3,0.3,0.1]
var get_ITI = function() {
  return 2100 + ITIs.shift()
 }

var getPracticeTrials = function() {
	var practice_stim = jsPsych.randomization.repeat($.extend(true, [], base_practice_stim), 1, true)
	var practice_trials = []
	for (i=0; i<practice_length; i++) {
		practice_trials.push(no_cue)
		if (practice_stim.data[i].cue == 'nocue') {
			
		} else if (practice_stim.data[i].cue == 'center') {
			practice_trials.push(center_cue)
		} else {
			var spatial_cue = {
				type: 'poldrack-single-stim',
				stimulus: '<div class = centerbox><div class = ANT_' + practice_stim.data[i].flanker_location +
					'><div class = ANT_text>*</p></div></div>',
				is_html: true,
				choices: 'none',
				data: {

					trial_id: "spatialcue",
					exp_stage: exp_stage
				},
				timing_post_trial: 0,
				timing_stim: 100,
				timing_response: 100
			}
			practice_trials.push(spatial_cue)
		}
		practice_trials.push(fixation)

		var practice_ANT_trial = {
			type: 'poldrack-categorize',
			stimulus: practice_stim.stimulus[i],
			is_html: true,
			key_answer: practice_stim.data[i].correct_response,
			correct_text: '<div class = centerbox><div style="color:#4FE829"; class = center-text>Correct!</div></div>',
			incorrect_text: '<div class = centerbox><div style="color:red"; class = center-text>Incorrect</div></div>',
			timeout_message: '<div class = centerbox><div class = center-text>Respond faster!</div></div>',
			choices: choices,
			data: practice_stim.data[i],
			timing_response: 1700,
			timing_stim: 1700,
			response_ends_trial: false,
			timing_feedback_duration: 1000,
			show_stim_with_feedback: false,
			timing_post_trial: 500,
			on_finish: function(data) {
				jsPsych.data.addDataToLastTrial({
					exp_stage: exp_stage
				})
				console.log('Trial: ' + current_trial +
              		'\nCorrect Response? ' + data.correct + ', RT: ' + data.rt)
			}
		}
		practice_trials.push(practice_ANT_trial)
	}
	return practice_trials
}


/* ************************************ */
/* Define experimental variables */
/* ************************************ */
var practice_repeats = 0
// task specific variables
var practice_length = 12
var num_blocks = 2
var block_length = 96

var current_trial = 0
var exp_stage = 'practice'
var choices = [89, 71]
var path = '/static/experiments/attention_network_task/images/'
var images = [path + 'left_arrow.png', path + 'right_arrow.png', path + 'no_arrow.png']
//preload
jsPsych.pluginAPI.preloadImages(images)


/* set up stim: location (2) * cue (3) * direction (2) * condition (2) */
var base_practice_stim = []
var test_stim = [[],[],[],[],[],[]] // track each cue/condition separately
var locations = ['up', 'down']
var cues = ['nocue', 'center', 'spatial']
var directions = ['left', 'right']
var conditions = ['congruent', 'incongruent']

for (ci = 0; ci < cues.length; ci++) {
	var c = cues[ci]
	for (coni = 0; coni < conditions.length; coni++) {
		var condition = conditions[coni]
		for (d = 0; d < directions.length; d++) {
			var center_image = images[d]
			var direction = directions[d]
			var side_image = ''
			if (condition == 'incongruent') {
				var side_image = images[1-d]
			} else {
				side_image = images[d]
			}
			for (l = 0; l < locations.length; l++) {
				var loc = locations[l]
				var stim = {
					stimulus: '<div class = centerbox><div class = ANT_text>+</div></div><div class = ANT_' + loc +
						'><img class = ANT_img src = ' + side_image + '></img><img class = ANT_img src = ' + side_image + '></img><img class = ANT_img src = ' + center_image + '></img><img class = ANT_img src = ' + side_image + '></img><img class = ANT_img src = ' + side_image + '></img></div></div>',
					data: {
						correct_response: choices[d],
						flanker_middle_direction: direction,
						flanker_type: condition,
						flanker_location: loc,
						cue: c, 
						trial_id: 'stim'
					}
				}
				base_practice_stim.push(stim)
				test_stim[ci*2+coni].push(stim)
			}
		}
	}
}

for (var i=0; i<test_stim.length; i++) {
	test_stim[i] = jsPsych.randomization.repeat(test_stim[i], block_length*num_blocks/24)
}
// set up stim order based on optimized trial sequence
var stim_index = [3,5,4,2,3,1,1,0,1,3,5,5,3,2,2,5,1,4,5,3,0,2,0,3,3,1,5,3,4,4,0,2,4,1,2,1,1,0,1,5,3,2,4,5,0,0,2,5,2,0,2,1,5,3,4,1,0,2,2,3,2,5,4,2,0,4,3,5,1,3,5,0,4,4,4,5,5,2,3,1,4,0,1,2,2,0,5,4,4,1,4,2,3,1,0,1,5,2,1,1,3,5,1,4,0,4,1,2,5,3,1,2,4,0,4,0,4,3,2,1,5,5,5,3,0,0,0,0,5,2,4,4,2,3,0,4,5,1,3,1,2,4,5,2,4,0,2,0,1,1,4,0,0,3,3,4,1,2,1,0,2,4,4,3,3,0,1,1,1,0,4,5,0,4,5,5,2,4,3,5,5,5,2,3,0,3,0,1,1,2,0,1]
var ordered_stim = []
for (var i=0; i<stim_index.length; i++) {
	ordered_stim.push(test_stim[stim_index[i]].shift())
}

/* set up repeats for test blocks */
var blocks = []
for (b = 0; b < num_blocks; b++) {
	blocks.push(ordered_stim.slice(b*block_length,(b+1)*block_length))
}



/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
/* define static blocks */
 var test_intro_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = center-text>Get ready!</div></div>',
	is_html: true,
	choices: 'none',
	timing_stim: 1500, 
	timing_response: 1500,
	data: {
		trial_id: "test_start_block"
	},
	timing_post_trial: 500,
	on_finish: function() {
		exp_stage = 'test'
	}
};

var rest_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = center-text>Take a break!<br>Next run will start in a moment</div></div>',
	is_html: true,
	choices: 'none',
	timing_response: 7500,
	data: {
		trial_id: "rest_block"
	},
	timing_post_trial: 1000
};

 var end_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = center-text><i>Fin</i></div></div>',
	is_html: true,
	choices: [32],
	timing_response: -1,
	response_ends_trial: true,
	data: {
		trial_id: "end",
		exp_id: 'attention_network_task'
	},
	timing_post_trial: 0
};

 var instructions_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = center-text>Indicate which direction the center arrow is pointing using the left (index) and right (middle) keys.</div>',
	is_html: true,
	choices: 'none',
	timing_stim: 9500, 
	timing_response: 9500,
	data: {
		trial_id: "instructions",
	},
	timing_post_trial: 500
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

/* Set up practice trials */
var practice_trials = getPracticeTrials()
var practice_loop = {
	timeline: practice_trials,
	loop_function: function(data) {
		practice_repeats+=1
		total_trials = 0
		correct_trials = 0
		for (var i = 0; i < data.length; i++) {
			if (data[i].trial_id == 'stim') {
				total_trials+=1
				if (data[i].correct == true) {
					correct_trials+=1
				}
			}
		}
		console.log('Practice Block Accuracy: ', correct_trials/total_trials)
		if (correct_trials/total_trials > .75 || practice_repeats == 3) {
			return false
		} else {
			practice_trials = getPracticeTrials()
			return true
		}
	}
};

/* set up ANT experiment */
var attention_network_task_experiment = [];
attention_network_task_experiment.push(instructions_block);
attention_network_task_experiment.push(practice_loop)
setup_fmri_intro(attention_network_task_experiment, choices)

/* Set up test trials */
var trial_num = 0
for (b = 0; b < blocks.length; b++) {
	attention_network_task_experiment.push(test_intro_block);
	var block = blocks[b]
	for (i = 0; i < block.data.length; i++) {
		var trial_num = trial_num + 1

		if (block.data[i].cue == 'nocue') {
			attention_network_task_experiment.push(no_cue)
		} else if (block.data[i].cue == 'center') {
			attention_network_task_experiment.push(center_cue)
		} else {
			var spatial_cue = {
				type: 'poldrack-single-stim',
				stimulus: '<div class = centerbox><div class = ANT_' + block.data[i].flanker_location +
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
			timing_response: get_ITI,
			timing_stim: 1700,
			response_ends_trial: false,
			timing_post_trial: 0,
			prompt: '<div class = centerbox><div class = ANT_text>+</div></div>',
			on_finish: function(data) {
				correct = data.key_press === data.correct_response
				console.log('Trial: ' + current_trial +
              '\nCorrect Response? ' + correct + ', RT: ' + data.rt)
				jsPsych.data.addDataToLastTrial({ 
					correct: correct,
					exp_stage: exp_stage
				})
			}
		}
		attention_network_task_experiment.push(ANT_trial)

	}
	if (b < (blocks.length-1)) {
		attention_network_task_experiment.push(rest_block)
	}
}
attention_network_task_experiment.push(end_block)