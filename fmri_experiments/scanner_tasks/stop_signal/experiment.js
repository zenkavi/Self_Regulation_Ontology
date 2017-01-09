/* ************************************ */
/* Define helper functions */
/* ************************************ */
var randomDraw = function(lst) {
	var index = Math.floor(Math.random() * (lst.length))
	return lst[index]
}

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
  gap = randomExponential(1/2)*225
  if (gap > 10000) {
    gap = 10000
  } else if (gap < 0) {
  	gap = 0
  } else {
  	gap = Math.round(gap/1000)*1000
  }
  return 2250 + gap //1850 (response time) + 500 (minimum ITI)
 }


/* Staircase procedure. After each successful stop, make the stop signal delay longer (making stopping harder) */
var updateSSD = function(data) {
	if (data.SS_trial_type == 'stop') {
		if (data.rt == -1 && SSD < 850) {
			SSD = SSD + 50
		} else if (data.rt != -1 && SSD > 0) {
			SSD = SSD - 50
		}
	}
}

var getSSD = function() {
	return SSD
}

/* After each test block let the subject know their average RT and accuracy. If they succeed or fail on too many stop signal trials, give them a reminder */
var getTestFeedback = function() {
	var data = test_block_data
	var rt_array = [];
	var sum_correct = 0;
	var go_length = 0;
	var stop_length = 0;
	var num_responses = 0;
	var successful_stops = 0;
	for (var i = 0; i < data.length; i++) {
		if (data[i].SS_trial_type == "go") {
			go_length += 1
			if (data[i].rt != -1) {
				num_responses += 1
				rt_array.push(data[i].rt);
				if (data[i].key_press == data[i].correct_response) {
					sum_correct += 1
				}
			}
		} else {
			stop_length += 1
			if (data[i].rt == -1) {
				successful_stops += 1
			}
		}
	}
	var average_rt = -1;
    if (rt_array.length !== 0) {
      average_rt = math.median(rt_array);
      rtMedians.push(average_rt)
    }
	var rt_diff = 0
	if (rtMedians.length !== 0) {
		rt_diff = (average_rt - rtMedians.slice(-1)[0])
	}
	var GoCorrect_percent = sum_correct / go_length;
	var missed_responses = (go_length - num_responses) / go_length
	var StopCorrect_percent = successful_stops / stop_length
	stopAccMeans.push(StopCorrect_percent)
	var stopAverage = math.mean(stopAccMeans)

	test_feedback_text = "<br>Done with a test block. Please take this time to read your feedback and to take a short break!"
	test_feedback_text += "</p><p class = block-text><strong>Average reaction time:  " + Math.round(average_rt) + " ms. Accuracy for non-star trials: " + Math.round(GoCorrect_percent * 100)+ "%</strong>" 
	if (average_rt > RT_thresh || rt_diff > rt_diff_thresh) {
		test_feedback_text +=
			'</p><p class = block-text>You have been responding too slowly, please respond to each shape as quickly and as accurately as possible.'
	}
	if (missed_responses >= missed_response_thresh) {
		test_feedback_text +=
			'</p><p class = block-text><strong>We have detected a number of trials that required a response, where no response was made.  Please ensure that you are responding to each shape, unless a star appears.</strong>'
	}
	if (GoCorrect_percent < accuracy_thresh) {
		test_feedback_text += '</p><p class = block-text>Your accuracy is too low. Remember, the correct keys are as follows: ' + prompt_text
	}
	if (StopCorrect_percent < (0.5-stop_thresh) || stopAverage < 0.45){
			 	test_feedback_text +=
			 		'</p><p class = block-text><strong>Remember to try and withhold your response when you see a stop signal.</strong>'	
	} else if (StopCorrect_percent > (0.5+stop_thresh) || stopAverage > 0.55){
	 	test_feedback_text +=
	 		'</p><p class = block-text><strong>Remember, do not slow your responses to the shape to see if a star will appear before you respond.  Please respond to each shape as quickly and as accurately as possible.</strong>'
	}

	return '<div class = centerbox><p class = block-text>' + test_feedback_text + '</p></div>'
}

/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// task specific variables
// Define and load images
var prefix = '/static/experiments/stop_signal/images/'
var images = [prefix + 'pentagon.png', prefix + 'hourglass.png', prefix + 'tear.png', prefix +
	'square.png'
]
jsPsych.pluginAPI.preloadImages(images);
/* Stop signal delay in ms */
var SSD = 250
var stop_signal =
	'<div class = coverbox></div><div class = stopbox><div class = centered-shape id = stop-signal></div><div class = centered-shape id = stop-signal-inner></div></div>'

/* Instruction Prompt */
var possible_responses = [
	["left button", 89],
	["right button", 71]
]
var choices = [possible_responses[0][1], possible_responses[1][1]]
var correct_responses = jsPsych.randomization.shuffle([possible_responses[0], possible_responses[0],
	possible_responses[1], possible_responses[1]
])

var tab = '&nbsp&nbsp&nbsp&nbsp'

var prompt_text = '<ul list-text><li><img class = prompt_stim src = ' + images[0] + '></img>' + tab +
	correct_responses[0][0] + '</li><li><img class = prompt_stim src = ' + images[1] + '></img>' + tab +
	correct_responses[1][0] + ' </li><li><img class = prompt_stim src = ' + images[2] + '></img>   ' +
	tab + correct_responses[2][0] +
	' </li><li><img class = prompt_stim src = ' + images[3] + '></img>' + tab + correct_responses[3][0] +
	' </li></ul>'

/* Global task variables */
var current_trial = 0
var rtMedians = []
var stopAccMeans =[]	
var RT_thresh = 1000
var rt_diff_thresh = 75
var missed_response_thresh = 0.1
var accuracy_thresh = 0.8
var stop_thresh = 0.2	
var exp_len = 125
var num_blocks = 3
var block_len = exp_len/num_blocks
var test_block_data = []

/* Define stims */
var stims = [{
	stimulus: '<div class = coverbox></div><div class = shapebox><img class = stim src = ' + images[0] + '></img></div>',
	data: {
		correct_response: correct_responses[0][1],
		trial_id: 'stim',
	}
}, {
	stimulus: '<div class = coverbox></div><div class = shapebox><img class = stim src = ' + images[1] + '></img></div>',
	data: {
		correct_response: correct_responses[1][1],
		trial_id: 'stim',
	}
}, {
	stimulus: '<div class = coverbox></div><div class = shapebox><img class = stim src = ' + images[2] + '></img></div>',
	data: {
		correct_response: correct_responses[2][1],
		trial_id: 'stim',
	}
}, {
	stimulus: '<div class = coverbox></div><div class = shapebox><img class = stim src = ' + images[3] + '></img></div>',
	data: {
		correct_response: correct_responses[3][1],
		trial_id: 'stim',
	}
}]



//setup test sequence
trials = jsPsych.randomization.repeat(stims, exp_len/4).slice(0,exp_len).slice(0,125)
var stop_trials = jsPsych.randomization.repeat(['stop', 'stop', 'go', 'go', 'go'], exp_len /5)
for (i=0; i<trials.length; i++) {
	trials[i]['SS_trial_type'] = stop_trials[i]
}
var blocks = []
for (b=0; b<num_blocks; b++) {
	blocks.push(trials.slice(block_len*b, (block_len*b+block_len)))
}

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
/* define static blocks  */
var task_setup_block = {
	type: 'survey-text',
	data: {
		trial_id: "task_setup"
	},
	questions: [
		[
			"<p class = center-block-text>Experimenter Setup</p>"
		]
	], on_finish: function(data) {
		SSD = parseInt(data.responses.slice(7, 10))
	}
}

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
	choices: 'none',
	timing_stim: 3000, 
	timing_response: 3000,
	data: {
		trial_id: "end",
		exp_id: 'stop_signal'
	},
	timing_post_trial: 0
};

 var instructions_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><p class = block-text>Only one key is correct for each shape. The correct keys are as follows:' + prompt_text +
		'</p><p class = block-text>Do not respond if you see the black star!</p></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 20000, 
  timing_response: 20000,
  data: {
    trial_id: "instructions",
  },
  timing_post_trial: 0
};


var fixation_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = fixation>+</div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: "fixation",
		exp_stage: "test"
	},
	timing_post_trial: 0,
	timing_response: 500
}


/* set up feedback blocks */
var test_feedback_block = {
  type: 'poldrack-single-stim',
  stimulus: getTestFeedback,
  is_html: true,
  choices: 'none',
  timing_stim: 12500, 
  timing_response: 12500,
  data: {
    trial_id: "test_feedback"
  },
  timing_post_trial: 1000,
  on_finish: function() {
  	test_block_data = []
  }
};


/* ************************************ */
/* Set up experiment */
/* ************************************ */

var stop_signal_experiment = []
stop_signal_experiment.push(task_setup_block);
stop_signal_experiment.push(instructions_block);
setup_fmri_intro(stop_signal_experiment, choices)

/* Test blocks */
// Loop through each trial within the block
for (b = 0; b < num_blocks; b++) {
	stop_signal_experiment.push(start_test_block)
	stop_signal_experiment.push(fixation_block)
	var stop_signal_block = {
		type: 'stop-signal',
		timeline: blocks[b], 
		SS_stimulus: stop_signal,
		is_html: true,
		choices: choices,
		timing_stim: 850,
		timing_response: get_ITI,
		SSD: getSSD,
		timing_SS: 500,
		timing_post_trial: 0,
		prompt: '<div class = centerbox><div class = fixation>+</div></div>',
		on_finish: function(data) {
			correct = false
			if (data.key_press == data.correct_response) {
				correct = true
			}
			updateSSD(data)
			jsPsych.data.addDataToLastTrial({
				exp_stage: 'test',
				trial_num: current_trial,
				correct: correct
			})
			current_trial += 1
			test_block_data.push(data)
			console.log('Trial: ' + current_trial +
              '\nCorrect Response? ' + correct + '\n')
		}
	}
	stop_signal_experiment.push(stop_signal_block)
	if ((b+1)<num_blocks) {
		stop_signal_experiment.push(test_feedback_block)
	}
}

stop_signal_experiment.push(end_block)