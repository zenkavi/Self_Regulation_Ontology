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
  gap = randomExponential(1/2)*1000
  if (gap > 5000) {
    gap = 5000
  } else if (gap < 500) {
  	gap = 500
  }
  return gap
}

/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// task specific variables
var choices = [89,71,82]
var congruent_stim = [{
	stimulus: '<div class = centerbox><div class = stroop-stim style = "color:red">RED</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'congruent',
		stim_color: 'red',
		stim_word: 'red',
		correct_response: choices[0]
	},
	key_answer: choices[0]
}, {
	stimulus: '<div class = centerbox><div class = stroop-stim style = "color:blue">BLUE</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'congruent',
		stim_color: 'blue',
		stim_word: 'blue',
		correct_response: choices[1]
	},
	key_answer: choices[1]
}, {
	stimulus: '<div class = centerbox><div class = stroop-stim style = "color:green">GREEN</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'congruent',
		stim_color: 'green',
		stim_word: 'green',
		correct_response: choices[2]
	},
	key_answer: choices[2]
}];

var incongruent_stim = [{
	stimulus: '<div class = centerbox><div class = stroop-stim style = "color:red">BLUE</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'red',
		stim_word: 'blue',
		correct_response: choices[0]
	},
	key_answer: choices[0]
}, {
	stimulus: '<div class = centerbox><div class = stroop-stim style = "color:red">GREEN</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'red',
		stim_word: 'green',
		correct_response: choices[0]
	},
	key_answer: choices[0]
}, {
	stimulus: '<div class = centerbox><div class = stroop-stim style = "color:blue">RED</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'blue',
		stim_word: 'red',
		correct_response: choices[1]
	},
	key_answer: choices[1]
}, {
	stimulus: '<div class = centerbox><div class = stroop-stim style = "color:blue">GREEN</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'blue',
		stim_word: 'green',
		correct_response: choices[1]
	},
	key_answer: choices[1]
}, {
	stimulus: '<div class = centerbox><div class = stroop-stim style = "color:green">RED</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'green',
		stim_word: 'red',
		correct_response: choices[2]
	},
	key_answer: choices[2]
}, {
	stimulus: '<div class = centerbox><div class = stroop-stim style = "color:green">BLUE</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'green',
		stim_word: 'blue',
		correct_response: choices[2]
	},
	key_answer: choices[2]
}];
var stims = [].concat(congruent_stim, congruent_stim, incongruent_stim)
var exp_len = 96
var test_stims = jsPsych.randomization.repeat(stims, exp_len / 12, true)
var exp_stage = 'test'
var current_trial = 1

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
/* define static blocks */
var instructions_block = {
	type: 'poldrack-text',
	data: {
		trial_id: "instruction"
	},
	text: '<div class = center-text>Respond to the <strong>ink color</strong> of the word!<br><br><span style = "color:red;padding-left:40px">WORD</span>: index finger<br><span style = "color:green;padding-left:80px">WORD</span>: middle finger<br><span style = "color:blue">WORD</span>: ring finger</div>',
	cont_key: [32],
	timing_post_trial: 1000
};

var start_test_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text>Get ready!</p></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 1500, 
  timing_response: 1500,
  data: {
    trial_id: "test_start_block"
  },
  timing_post_trial: 500
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
		exp_id: 'stroop'
	},
	timing_post_trial: 0
};


var fixation_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = fixation>+</div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: "fixation"
	},
	timing_post_trial: 0,
	timing_stim: -1,
	timing_response: get_ITI,
	on_finish: function() {
		jsPsych.data.addDataToLastTrial({'exp_stage': exp_stage})
	},
}

/* create experiment definition array */
stroop_experiment = []
stroop_experiment.push(instructions_block)
setup_fmri_intro(stroop_experiment, choices)

stroop_experiment.push(start_test_block)
	/* define test trials */
for (i = 0; i < exp_len; i++) {
	stroop_experiment.push(fixation_block)
	var test_block = {
		type: 'poldrack-categorize',
		stimulus: test_stims.stimulus[i],
		data: test_stims.data[i],
		key_answer: test_stims.key_answer[i],
		is_html: true,
		correct_text: '<div class = fb_box><div class = center-text>Correct!</div></div>',
		incorrect_text: '<div class = fb_box><div class = center-text>Incorrect</div></div>',
		timeout_message: '<div class = fb_box><div class = center-text>Respond Faster!</div></div>',
		choices: choices,
		timing_response: 1500,
		timing_stim: -1,
		timing_feedback_duration: 500,
		show_stim_with_feedback: true,
		timing_post_trial: 250,
		on_finish: function(data) {
			console.log('Trial Num: ', current_trial)
			console.log('Correct? ', data.correct)
			jsPsych.data.addDataToLastTrial({
				trial_id: 'stim',
				trial_num: current_trial,
				exp_stage: 'test'
			})
			current_trial += 1
		}
	}
	stroop_experiment.push(test_block)
}
stroop_experiment.push(end_block)