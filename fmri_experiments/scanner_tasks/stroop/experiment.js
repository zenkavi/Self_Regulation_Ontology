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
  gap = randomExponential(1/2)*200
  if (gap > 10000) {
    gap = 10000
  } else if (gap < 0) {
  	gap = 0
  } else {
  	gap = Math.round(gap/1000)*1000
  }
  return 2000 + gap //1500 (stim time) + 500 (minimum ITI)
 }

/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// task specific variables
var choices = [89, 71, 82]
var congruent_stim = [{
	stimulus: '<div class = stroopbox><div class = stroop-stim style = "color:red">RED</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'congruent',
		stim_color: 'red',
		stim_word: 'red',
		correct_response: choices[0]
	},
	key_answer: choices[0]
}, {
	stimulus: '<div class = stroopbox><div class = stroop-stim style = "color:blue">BLUE</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'congruent',
		stim_color: 'blue',
		stim_word: 'blue',
		correct_response: choices[1]
	},
	key_answer: choices[1]
}, {
	stimulus: '<div class = stroopbox><div class = stroop-stim style = "color:green">GREEN</div></div>',
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
	stimulus: '<div class = stroopbox><div class = stroop-stim style = "color:red">BLUE</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'red',
		stim_word: 'blue',
		correct_response: choices[0]
	},
	key_answer: choices[0]
}, {
	stimulus: '<div class = stroopbox><div class = stroop-stim style = "color:red">GREEN</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'red',
		stim_word: 'green',
		correct_response: choices[0]
	},
	key_answer: choices[0]
}, {
	stimulus: '<div class = stroopbox><div class = stroop-stim style = "color:blue">RED</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'blue',
		stim_word: 'red',
		correct_response: choices[1]
	},
	key_answer: choices[1]
}, {
	stimulus: '<div class = stroopbox><div class = stroop-stim style = "color:blue">GREEN</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'blue',
		stim_word: 'green',
		correct_response: choices[1]
	},
	key_answer: choices[1]
}, {
	stimulus: '<div class = stroopbox><div class = stroop-stim style = "color:green">RED</div></div>',
	data: {
		trial_id: 'stim',
		condition: 'incongruent',
		stim_color: 'green',
		stim_word: 'red',
		correct_response: choices[2]
	},
	key_answer: choices[2]
}, {
	stimulus: '<div class = stroopbox><div class = stroop-stim style = "color:green">BLUE</div></div>',
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
var practice_len = 12
var practice_stims  = jsPsych.randomization.repeat(stims, practice_len / 12)
var test_stims = jsPsych.randomization.repeat(stims, exp_len / 12)
var exp_stage = 'practice'
var current_trial = 1

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
var instructions_prompt = '<div class=prompt_box><span style = "color:red;padding-right:40px">Index</span><span style = "color:green;">Middle</span><span style = "color:blue;padding-left:40px">Ring</span></div>'

/* define static blocks */
var instructions_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text>Respond to the <strong>ink color</strong> of the word!<br><br><span style = "color:red;padding-left:40px">WORD</span>: Up<br><span style = "color:green;padding-left:60px">WORD</span>: Left<br><span style = "color:blue;padding-left:95px">WORD</span>: Right<br><br>We will start with practice</div></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 10000, 
  timing_response: 10000,
  data: {
    trial_id: "instructions",
  },
  timing_post_trial: 500
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
  timing_post_trial: 500,
  on_finish: function() {
  	current_trial = 0
  	exp_stage = 'test'
  }
};

 var end_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = center-text><i>Fin</i></div></div>',
	is_html: true,
	choices: 'none',
	timing_stim: 3000, 
	timing_response: 3000,
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
	timing_response: 500
}

var practice_fixation_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = fixation>+</div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: "fixation"
	},
	timing_post_trial: 0,
	timing_stim: -1,
	timing_response: 250,
	prompt: instructions_prompt
}

practice = []
for (var i=0; i<practice_stims.length; i++) {
	var practice_block = {
		timeline: [practice_stims[i]],
		type: 'poldrack-categorize',
		is_html: true,
		choices: choices,
		timing_response: 1500,
		timing_stim: 1500,
		timing_post_trial: 0,
		prompt: '<div class = centerbox><div class = fixation>+</div></div>',
		timing_feedback_duration: 500,
		show_stim_with_feedback: true,
		correct_text: '<div class = fb_box><div class = center-text><font size = 20>Correct!</font></div></div>',
		incorrect_text: '<div class = fb_box><div class = center-text><font size = 20>Incorrect</font></div></div>',
		timeout_message: '<div class = fb_box><div class = center-text><font size = 20>Respond Faster!</font></div></div>',
		prompt: instructions_prompt,
		on_finish: function(data) {
			var correct = false
			if (data.correct_response == data.key_press) {
				correct = true
			}
			console.log('Trial: ' + current_trial +
              '\nCorrect Response? ' + correct + '\n')
			jsPsych.data.addDataToLastTrial({
				correct: correct,
				trial_id: 'stim',
				trial_num: current_trial,
				exp_stage: exp_stage
			})
			current_trial += 1
		}
	}
	practice.push(practice_block)
	practice.push(practice_fixation_block)
}

var test_block = {
	timeline: test_stims,
	type: 'poldrack-single-stim',
	is_html: true,
	choices: choices,
	timing_response: get_ITI,
	timing_stim: 1500,
	timing_post_trial: 0,
	prompt: '<div class = centerbox><div class = fixation>+</div></div>' + instructions_prompt,
	on_finish: function(data) {
		var correct = false
		if (data.correct_response == data.key_press) {
			correct = true
		}
		console.log('Trial: ' + current_trial +
              '\nCorrect Response? ' + correct + '\n')
		jsPsych.data.addDataToLastTrial({
			correct: correct,
			trial_id: 'stim',
			trial_num: current_trial,
			exp_stage: exp_stage
		})
		current_trial += 1
	}
}

/* create experiment definition array */
stroop_experiment = []
stroop_experiment.push(instructions_block)
stroop_experiment = stroop_experiment.concat(practice)
setup_fmri_intro(stroop_experiment, choices)
stroop_experiment.push(start_test_block)
stroop_experiment.push(fixation_block)
stroop_experiment.push(test_block)
stroop_experiment.push(end_block)