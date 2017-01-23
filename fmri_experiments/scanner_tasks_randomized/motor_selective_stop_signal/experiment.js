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
  gap = randomExponential(1/2)*225
  if (gap > 10000) {
    gap = 10000
  } else if (gap < 0) {
  	gap = 0
  } else {
  	gap = Math.round(gap/1000)*1000
  }
  return 2250 + gap //1850 (response time) + 400 (minimum ITI)
 }


var randomDraw = function(lst) {
	var index = Math.floor(Math.random() * (lst.length))
	return lst[index]
}

/* After each test block let the subject know their average RT and accuracy. If they succeed or fail on too many stop signal trials, give them a reminder */
var getTestFeedback = function() {
	var data = test_block_data
	var rt_array = [];
	var sum_correct = 0;
	var go_length = 0;
	var stop_length = 0;
	var ignore_length = 0;
	var num_responses = 0;
	var ignore_responses = 0;
	var successful_stops = 0;
	for (var i = 0; i < data.length; i++) {
		if (data[i].condition != "stop") {
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
		if (data[i].condition == "ignore") {
			ignore_length += 1
			if (data[i].rt != -1) {
				ignore_responses += 1
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
	var ignoreRespond_percent = ignore_responses / ignore_length
	
	test_feedback_text = "<br>Done with a test block. Please take this time to read your feedback and to take a short break! Press <strong>enter</strong> to continue after you have read the feedback."
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
			 		'</p><p class = block-text><strong>Remember to try and withhold your response when you see a stop signal AND the correct key is the ' +
					stop_response[0] + '.</strong>' 

	} else if (StopCorrect_percent > (0.5+stop_thresh) || stopAverage > 0.55){
	 	test_feedback_text +=
	 		'</p><p class = block-text><strong>Remember, do not slow your responses to the shape to see if a star will appear before you respond.  Please respond to each shape as quickly and as accurately as possible.</strong>'
	}
	if (ignoreRespond_percent <= motor_thresh){
      	test_feedback_text +=
          '</p><p class = block-text> You have been stopping to both the Z and M keys.  Please make sure to <strong>stop your response only when the star appears and you were going to hit the '+ stop_response[0]+'.</strong>'
  	}

	return '<div class = centerbox><p class = block-text>' + test_feedback_text + '</p></div>'
}

/* Staircase procedure. After each successful stop, make the stop signal delay longer (making stopping harder) */
var updateSSD = function(data) {
	if (data.condition == 'stop') {
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

var getPracticeTrials = function() {
	var practice = []
	var trials = jsPsych.randomization.repeat(stimuli, practice_len/4)
	for (i=0; i<trials.length; i++) {
		trials[i]['key_answer'] = trials[i].data.correct_response
	}
	practice.push(fixation_block)
	var practice_block = {
		type: 'poldrack-categorize',
		timeline: trials, 
		is_html: true,
		choices: choices,
		timing_stim: 850,
		timing_response: 1850,
		correct_text: '<div class = feedbackbox><div style="color:#4FE829"; class = center-text>Correct!</p></div>',
		incorrect_text: '<div class = feedbackbox><div style="color:red"; class = center-text>Incorrect</p></div>',
		timeout_message: '<div class = feedbackbox><div class = center-text>Too Slow</div></div>',
		show_stim_with_feedback: false,
		timing_feedback_duration: 500,
		timing_post_trial: 250,
		on_finish: function(data) {
			jsPsych.data.addDataToLastTrial({
				exp_stage: 'practice',
				trial_num: current_trial,
			})
			current_trial += 1
			console.log('Trial: ' + current_trial +
              '\nCorrect Response? ' + data.correct + ', RT: ' + data.rt)
		}
	}
	practice.push(practice_block)
	return practice
}

/* ************************************ */
/* Define experimental variables */
/* ************************************ */
var practice_repeats = 0
// task specific variables
// Define and load images
var prefix = '/static/experiments/motor_selective_stop_signal/images/'
var images = [prefix + 'circle.png', prefix + 'rhombus.png', prefix + 'Lshape.png', prefix +
  'triangle.png'
]
jsPsych.pluginAPI.preloadImages(images);
images = jsPsych.randomization.shuffle(images)
/* Stop signal delay in ms */
var SSD = 250
var stop_signal =
	'<div class = coverbox></div><div class = stopbox><div class = centered-shape id = stop-signal></div><div class = centered-shape id = stop-signal-inner></div></div>'

/* Instruction Prompt */
var possible_responses = jsPsych.randomization.shuffle([
	["index finger", 89],
	["middle finger", 71]
])

var choices = [possible_responses[0][1], possible_responses[1][1]]

var tab = '&nbsp&nbsp&nbsp&nbsp'
var prompt_text = '<ul list-text><li><img class = prompt_stim src = ' + images[0] + '></img>' + tab +
  possible_responses[0][0] + '</li><li><img class = prompt_stim src = ' + images[1] + '></img>' +
  tab +
  possible_responses[0][0] + ' </li><li><img class = prompt_stim src = ' + images[2] + '></img>   ' +
  '&nbsp&nbsp&nbsp' + possible_responses[1][0] +
  ' </li><li><img class = prompt_stim src = ' + images[3] + '></img>' + tab + possible_responses[1][0] + ' </li></ul>'

/* Global task variables */
var current_trial = 0
var rtMedians = []
var stopAccMeans =[]
var RT_thresh = 1000
var rt_diff_thresh = 75
var missed_response_thresh = 0.1
var accuracy_thresh = 0.8
var stop_thresh = 0.2
var motor_thresh = 0.6
var stop_response = possible_responses[0]
var ignore_response = possible_responses[1]
var practice_len = 12
var test_block_data = [] // records the data in the current block to calculate feedback
var test_block_len = 50
var num_blocks = 5

/* Define Stims */
var stimuli = [{
	stimulus: '<div class = coverbox></div><div class = shapebox><img class = stim src = ' + images[0] + '></img></div>',
	data: {
		correct_response: possible_responses[0][1],
		trial_id: 'stim',
	}
}, {
	stimulus: '<div class = coverbox></div><div class = shapebox><img class = stim src = ' + images[1] + '></img></div>',
	data: {
		correct_response: possible_responses[0][1],
		trial_id: 'stim',
	}
}, {
	stimulus: '<div class = coverbox></div><div class = shapebox><img class = stim src = ' + images[2] + '></img></div>',
	data: {
		correct_response: possible_responses[1][1],
		trial_id: 'stim',
	}
}, {
	stimulus: '<div class = coverbox></div><div class = shapebox><img class = stim src = ' + images[3] + '></img></div>',
	data: {
		correct_response: possible_responses[1][1],
		trial_id: 'stim',
	}
}]

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
/* define static blocks */
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
	choices: [32],
	timing_response: -1,
	response_ends_trial: true,
	data: {
		trial_id: "end",
		exp_id: 'motor_selective_stop_signal'
	},
	timing_post_trial: 0
};

 var instructions_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><p class = block-text>Only one key is correct for each shape. The correct keys are as follows:' + prompt_text +
		'</p><p class = block-text>Do not respond if you see the red star if your response was going to be the ' + stop_response[0] + '!</p><br><p class = block-text>We will start with practice</p></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 15000, 
  timing_response: 15000,
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

/* prompt blocks are used during practice to show the instructions */

var prompt_fixation_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = shapebox><div class = fixation>+</div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: "fixation",
		exp_stage: "practice"
	},
	timing_post_trial: 0,
	timing_response: 500,
	prompt: prompt_text
}

/* set up feedback blocks */
var test_feedback_block = {
  type: 'poldrack-single-stim',
  stimulus: getTestFeedback,
  is_html: true,
  choices: 'none',
  timing_stim: 13500, 
  timing_response: 13500,
  data: {
    trial_id: "test_feedback"
  },
  timing_post_trial: 1000,
  on_finish: function() {
  	test_block_data = []
  }
};

// set up practice trials
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

/* ************************************ */
/* Set up experiment */
/* ************************************ */

var motor_selective_stop_signal_experiment = []
motor_selective_stop_signal_experiment.push(task_setup_block);
motor_selective_stop_signal_experiment.push(instructions_block);
motor_selective_stop_signal_experiment.push(practice_loop);
setup_fmri_intro(motor_selective_stop_signal_experiment, choices)

/* Test blocks */
// Loop through the multiple blocks within each condition
for (b = 0; b < num_blocks; b++) {
	stop_signal_exp_block = []
	var go_stims = jsPsych.randomization.repeat(stimuli, test_block_len*0.6 / 4, true)
	var stop_stims = jsPsych.randomization.repeat(stimuli.slice(0,2), test_block_len*0.2 / 2, true)
	var ignore_stims = jsPsych.randomization.repeat(stimuli.slice(2,4), test_block_len*0.2 / 2, true)
	var stop_trials = jsPsych.randomization.repeat(['stop', 'ignore', 'go', 'go', 'go'], test_block_len /
			5, false)
	// Loop through each trial within the block
	stop_signal_exp_block.push(start_test_block)
	stop_signal_exp_block.push(fixation_block)
	for (i = 0; i < test_block_len; i++) {
	    var stop_trial = stop_trials[i]
	    var trial_stim = ''
	    var trial_data = []
	    if (stop_trials[i] == 'ignore') {
	    	trial_stim = ignore_stims.stimulus.pop()
	    	trial_data = ignore_stims.data.pop()
	    	stop_trial = 'stop'
	    } else if (stop_trials[i] == 'stop') {
	    	trial_stim = stop_stims.stimulus.pop()
	    	trial_data = stop_stims.data.pop()
	    } else {
	    	trial_stim = go_stims.stimulus.pop()
	    	trial_data = go_stims.data.pop()
	    }
	    trial_data = $.extend({}, trial_data)
	    trial_data.condition = stop_trials[i]
		var stop_signal_block = {
			type: 'stop-signal',
			stimulus: trial_stim,
			SS_stimulus: stop_signal,
			SS_trial_type: stop_trial,
			data: trial_data,
			is_html: true,
			choices: choices,
			timing_stim: 850,
			timing_response: 1850,
			SSD: getSSD,
			timing_SS: 500,
			timing_post_trial: 0,
			on_finish: function(data) {
				correct = false
				if (data.key_press == data.correct_response) {
					correct = true
				}
				updateSSD(data)
				jsPsych.data.addDataToLastTrial({
					exp_stage: "test",
					stop_response: stop_response[1],
					trial_num: current_trial
				})
				current_trial += 1
				test_block_data.push(data)
				console.log('Trial: ' + current_trial +
              '\nCorrect Response? ' + correct + ', RT: ' + data.rt + ', SSD: ' + data.SS_delay)
			}
		}
		stop_signal_exp_block.push(stop_signal_block)
	}

	motor_selective_stop_signal_experiment = motor_selective_stop_signal_experiment.concat(
		stop_signal_exp_block)
	if ((b+1)<num_blocks) {
		motor_selective_stop_signal_experiment.push(test_feedback_block)
	}
}

motor_selective_stop_signal_experiment.push(end_block)