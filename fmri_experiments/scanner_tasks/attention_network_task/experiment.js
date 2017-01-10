/* ************************************ */
/* Define helper functions */
/* ************************************ */
var get_ITI = function() {
  // ref: https://gist.github.com/nicolashery/5885280
  function randomExponential(rate, randomUniform) {
    // http://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates
    rate = rate || 1;

    // Allow to pass a random uniform value or function
    // Default to Math.random()
    var U = randomUniform;
    if (typeof randomUniform === 'function') U = randomUniform();
    if (!U) U = Math.random();

    return -Math.log(U) / rate;
  }
  gap = randomExponential(1/2)*260
  if (gap > 10000) {
    gap = 10000
  } else if (gap < 0) {
  	gap = 0
  } else {
  	gap = Math.round(gap/1000)*1000
  }
  return 2100 + gap //1700 (stim time) + 400 (minimum ITI)
 }

var getPracticeTrials = function() {
	var practice_stim = jsPsych.randomization.repeat($.extend(true, [], test_stimuli), 1, true)
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
			on_finish: function() {
				jsPsych.data.addDataToLastTrial({
					exp_stage: exp_stage
				})
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
var test_stimuli = []
var choices = [89, 71]
var path = '/static/experiments/attention_network_task/images/'
var images = [path + 'left_arrow.png', path + 'right_arrow.png', path + 'no_arrow.png']
//preload
jsPsych.pluginAPI.preloadImages(images)

/* set up stim: location (2) * cue (3) * direction (2) * condition (2) */
var locations = ['up', 'down']
var cues = ['nocue', 'center', 'spatial']
var directions = ['left', 'right']
var conditions = ['congruent', 'incongruent']
for (l = 0; l < locations.length; l++) {
	var loc = locations[l]
	for (ci = 0; ci < cues.length; ci++) {
		var c = cues[ci]
		for (d = 0; d < directions.length; d++) {
			var center_image = images[d]
			var direction = directions[d]
			for (coni = 0; coni < conditions.length; coni++) {
				var condition = conditions[coni]
				var side_image = ''
				if (condition == 'incongruent') {
					var side_image = images[1-d]
				} else {
					side_image = images[d]
				}
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
				test_stimuli.push(stim)
			}
		}
	}
}


/* set up repeats for test blocks */
var blocks = []
for (b = 0; b < num_blocks; b++) {
	blocks.push(jsPsych.randomization.repeat($.extend(true, [], test_stimuli), block_length/test_stimuli.length, true))
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
				console.log('Trial: ' + data.trial_num +
							'\nCorrect Response? ' + correct + '\n')
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