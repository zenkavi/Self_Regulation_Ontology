/* ************************************ */
/* Define helper functions */
/* ************************************ */
var getInstructFeedback = function() {
	return '<div class = centerbox><p class = center-block-text>' + feedback_instruct_text +
		'</p></div>'
}

/* Append gap and current trial to data and then recalculate for next trial*/

//this adds the current trial and the stims shown to the data
var appendTestData = function() {
	jsPsych.data.addDataToLastTrial({
		trial_num: current_trial,
		stim_top: stims.slice(0,3),
		stim_bottom: stims.slice(3),
		exp_stage: exp_stage
	})
};

//this adds the cue shown and trial number to data
var appendCueData = function() {
	jsPsych.data.addDataToLastTrial({
		cue: cue,
		trial_num: current_trial,
		exp_stage: exp_stage
	})
};


//this is an algorithm to choose the training set based on rules of the game (training sets are composed of any letter not presented in the last two training sets)
var getTrainingSet = function() {
	preceeding1stims = []
	preceeding2stims = []
	trainingArray = jsPsych.randomization.repeat(stimArray, 1);
	if (current_trial < 1) {
		stims = trainingArray.slice(0,6)
	} else if (current_trial == 1) {
		preceeding1stims = stims.slice()
		stims = trainingArray.filter(function(y) {
			return (jQuery.inArray(y, preceeding1stims) == -1)
		}).slice(0,6)
	} else {
		preceeding2stims = preceeding1stims.slice()
		preceeding1stims = stims.slice()
		stims = trainingArray.filter(function(y) {
			return (jQuery.inArray(y, preceeding1stims.concat(preceeding2stims)) == -1)
		}).slice(0,6)
	}
	return '<div class = centerbox><div class = fixation><span style="color:red">+</span></div></div>' +
		'<div class = topLeft><img class = forgetStim src ="' + pathSource + stims[0] + fileType +
		'"></img></div>' +
		'<div class = topMiddle><img class = forgetStim src ="' + pathSource + stims[1] + fileType +
		'"></img></div>' +
		'<div class = topRight><img class = forgetStim src ="' + pathSource + stims[2] + fileType +
		'"></img></div>' +
		'<div class = bottomLeft><img class = forgetStim src ="' + pathSource + stims[3] + fileType +
		'"></img></div>' +
		'<div class = bottomMiddle><img class = forgetStim src ="' + pathSource + stims[4] + fileType +
		'"></img></div>' +
		'<div class = bottomRight><img class = forgetStim src ="' + pathSource + stims[5] + fileType +
		'"></img></div>'
};

//returns a cue pseudorandomly, either TOP or BOT
var getCue = function() {
	var temp = Math.floor(Math.random() * 2)
	cue = cueArray[temp]
	return '<div class = centerbox><img class = forgetStim src ="' + pathSource + cue + fileType +
		'"></img></div>'
};

// Will pop out a probe type from the entire probeTypeArray and then choose a probe congruent with the probe type

var getPracticeProbe = function() {
	probeType = practiceProbeTypeArray.pop()
	var trainingArray = jsPsych.randomization.repeat(stimArray, 1);
	var lastCue = cue
	var lastSet_top = stims.slice(0,3)
	var lastSet_bottom = stims.slice(3)
	if (probeType == 'pos') {
		if (lastCue == 'BOT') {
			probe = lastSet_top[Math.floor(Math.random() * 3)]
		} else if (lastCue == 'TOP') {
			probe = lastSet_bottom[Math.floor(Math.random() * 3)]
		}
	} else if (probeType == 'neg') {
		if (lastCue == 'BOT') {
			probe = lastSet_bottom[Math.floor(Math.random() * 3)]
		} else if (lastCue == 'TOP') {
			probe = lastSet_top[Math.floor(Math.random() * 3)]
		}
	} else if (probeType == 'con') {
		newArray = trainingArray.filter(function(y) {
			return (y != lastSet_top[0] && y != lastSet_top[1] && y != lastSet_top[2] && y !=
				lastSet_bottom[0] && y != lastSet_bottom[1] && y != lastSet_bottom[2])
		})
		probe = newArray.pop()
	}
	return '<div class = centerbox><img class = forgetStim src ="' + pathSource + probe + fileType +
		'"></img></div>'
};

var getResponse = function() {
	if (cue == 'TOP') {
		if (jQuery.inArray(probe, stims.slice(3)) != -1) {
			return 37
		} else {
			return 39
		}

	} else if (cue == 'BOT') {
		if (jQuery.inArray(probe, stims.slice(0,3)) != -1) {
			return 37
		} else {
			return 39
		}
	}
}

var appendPracticeProbeData = function() {
	jsPsych.data.addDataToLastTrial({
		probe_letter: probe,
		probe_type: probeType,
		trial_num: current_trial
	})
}

var resetTrial = function() {
	current_trial = 0
	exp_stage = 'test'
}

/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// generic task variables
var run_attention_checks = false
var attention_check_thresh = 0.65
var sumInstructTime = 0 //ms
var instructTimeThresh = 0 ///in seconds
var credit_var = true

// task specific variables
var choices = [37, 39]
var exp_stage = 'practice'
var practice_length = 8
var num_trials = 24
var num_runs = 3 
var experimentLength = num_trials * num_runs
var current_trial = 0
var stimArray = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
	'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
];
var cueArray = ['TOP', 'BOT']
var probe = ''
var cue = ''
var stims = []
var preceeding1stims = []
var preceeding2stims = []
var probes = ['pos', 'pos', 'neg', 'con']
var probeTypeArray = jsPsych.randomization.repeat(probes, experimentLength / 4)
var practiceProbeTypeArray = jsPsych.randomization.repeat(probes, practice_length/2)
var stimFix = ['fixation']
var pathSource = '/static/experiments/directed_forgetting/images/'
var fileType = '.png'
var images = []
for (var i = 0; i < stimArray.length; i++) {
	images.push(pathSource + stimArray[i] + fileType)
}
images.push(pathSource + 'TOP.png')
images.push(pathSource + 'BOT.png')
	//preload images
jsPsych.pluginAPI.preloadImages(images)

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
var test_img_block1 = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = instructBox><p class = block-text>This is what a trial will look like.  The letters A, B, and C are on the top portion, while the letters D, E, and F are on the bottom portion.  After these letters disappear, a cue will be presented.  If the cue presented is <strong>TOP</strong>, then you should <strong> forget the letters A, B, and C</strong> and remember D, E, and F.  If the cue presented is <strong>BOT</strong>, then you should <strong> forget D, E, and F </strong> and remember A, B, and C.    Press <strong> enter</strong> to continue.</div>'+
		'<div class = centerbox><div class = fixation><span style="color:red">+</span></div></div>' +
		'<div class = topLeft><img class = forgetStim src ="' + pathSource + stimArray[0] + fileType +
		'"></img></div>' +
		'<div class = topMiddle><img class = forgetStim src ="' + pathSource + stimArray[1] + fileType +
		'"></img></div>' +
		'<div class = topRight><img class = forgetStim src ="' + pathSource + stimArray[2] + fileType +
		'"></img></div>' +
		'<div class = bottomLeft><img class = forgetStim src ="' + pathSource + stimArray[3] + fileType +
		'"></img></div>' +
		'<div class = bottomMiddle><img class = forgetStim src ="' + pathSource + stimArray[4] + fileType +
		'"></img></div>' +
		'<div class = bottomRight><img class = forgetStim src ="' + pathSource + stimArray[5] + fileType +
		'"></img></div>',
	is_html: true,
	choices: [13],
	data: {
		trial_id: "instruction_images"
	},
	timing_post_trial: 0,
	timing_stim: 300000,
	timing_response: 300000,
	response_ends_trial: true,
}

var test_img_block2 = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><p class = center-block-text>We will present you with 1 example.  Press <strong> enter</strong> to begin.</p></div>',
	is_html: true,
	choices: [13],
	data: {
		trial_id: "instruction_images"
	},
	timing_post_trial: 0,
	timing_stim: 300000,
	timing_response: 300000,
	response_ends_trial: true,
}

var end_block = {
	type: 'poldrack-text',
	data: {
		trial_id: "end",
		exp_id: 'directed_forgetting'
	},
	timing_response: 180000,
	text: '<div class = centerbox><p class = center-block-text>Thanks for completing this practice!</p><p class = center-block-text>Press <strong>enter</strong> to continue.</p></div>',
	cont_key: [13],
	timing_post_trial: 0,
};



var feedback_instruct_text =
	'Welcome to the practice phase for this experiment. Press <strong>enter</strong> to begin.'
var feedback_instruct_block = {
	type: 'poldrack-text',
	data: {
		trial_id: 'instruction'
	},
	cont_key: [13],
	text: getInstructFeedback,
	timing_post_trial: 0,
	timing_response: 180000
};
/// This ensures that the subject does not read through the instructions too quickly.  If they do it too quickly, then we will go over the loop again.
var instructions_block = {
	type: 'poldrack-instructions',
	data: {
		trial_id: 'instruction'
	},
	pages: [
		'<div class = centerbox><p class = block-text>In this task, on each trial you will be presented with 6 letters. You must memorize all 6 letters. </p><p class = block-text>After the presentation of 6 letters, there will be a short delay. You will then be presented with a cue, either <strong>TOP</strong> or <strong>BOT</strong>. This will instruct you to <strong>forget</strong> the 3 letters located at either the top or bottom (respectively) of the screen.</p> <p class = block-text> The three remaining letters that you must remember are called your <strong>memory set</strong>. You should remember these three letters while forgetting the other three.</p><p class = block-text>You will then be presented with a single letter. Respond with the <strong> Left</strong> arrow key if it is in the memory set, and the <strong> Right </strong> arrow key if it was not in the memory set.</p><p class = block-text>Please make sure you understand these instructions before continuing. You will see an example trial after you end the instructions.</p></div>',
	],
	allow_keys: false,
	show_clickable_nav: true,
	timing_post_trial: 1000
};

var instruction_node = {
	timeline: [feedback_instruct_block, instructions_block],
	/* This function defines stopping criteria */
	loop_function: function(data) {
		for (i = 0; i < data.length; i++) {
			if ((data[i].trial_type == 'poldrack-instructions') && (data[i].rt != -1)) {
				rt = data[i].rt
				sumInstructTime = sumInstructTime + rt
			}
		}
		if (sumInstructTime <= instructTimeThresh * 1000) {
			feedback_instruct_text =
				'Read through instructions too quickly.  Please take your time and make sure you understand the instructions.  Press <strong>enter</strong> to continue.'
			return true
		} else if (sumInstructTime > instructTimeThresh * 1000) {
			feedback_instruct_text = 'Done with instructions. Press <strong>enter</strong> to continue.'
			return false
		}
	}
}

var start_practice_block = {
	type: 'poldrack-instructions',
	data: {
		trial_id: 'instruction'
	},
	pages: [
		'<div class = centerbox><p class = block-text>As you saw, there are three letters at the top of the screen and three letters on the bottom of the screen. After a delay, the cue (TOP or BOT) tells you whether to <strong>forget</strong> the three letters at the top or bottom of the screen, respectively. The other three letters are your memory set.</p><p class = block-text>After the cue, you are shown a letter. Please respond with the <strong> Left</strong> arrow key if it is in the memory set, and the <strong> Right </strong> arrow key if it was not in the memory set.</p><p class = block-text>We will now start the practice phase.</p></div>',
	],
	allow_keys: false,
	show_clickable_nav: true,
	timing_post_trial: 1000
};



var start_fixation_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = fixation><span style="color:red">+</span></div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: "fixation"
	},
	timing_post_trial: 0,
	timing_stim: 1000,
	timing_response: 1000,
	on_finish: function() {
		jsPsych.data.addDataToLastTrial({
			exp_stage: exp_stage,
			trial_num: current_trial
		})
	}
}

var fixation_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = fixation><span style="color:red">+</span></div></div>',
	is_html: true,
	choices: 'none',
	data: {
		trial_id: "fixation"
	},
	timing_post_trial: 0,
	timing_stim: 3000,
	timing_response: 3000,
	on_finish: function() {
		jsPsych.data.addDataToLastTrial({
			exp_stage: exp_stage,
			trial_num: current_trial
		})
	}
}

var ITI_fixation_block = {
	type: 'poldrack-single-stim',
	stimulus: '<div class = centerbox><div class = fixation><span style="color:red">+</span></div></div>',
	is_html: true,
	choices: choices,
	data: {
		trial_id: "ITI_fixation"
	},
	timing_post_trial: 0,
	timing_stim: 4000,
	timing_response: 4000,
	on_finish: function() {
		jsPsych.data.addDataToLastTrial({
			exp_stage: exp_stage,
			trial_num: current_trial
		})
		current_trial = current_trial + 1
	}
}

var training_block = {
	type: 'poldrack-single-stim',
	stimulus: getTrainingSet,
	is_html: true,
	data: {
		trial_id: "stim"
	},
	choices: 'none',
	timing_post_trial: 0,
	timing_stim: 2500,
	timing_response: 2500,
	on_finish: appendTestData,
};



var cue_block = {
	type: 'poldrack-single-stim',
	stimulus: getCue,
	is_html: true,
	data: {
		trial_id: "cue",
		exp_stage: "test"
	},
	choices: false,
	timing_post_trial: 0,
	timing_stim: 1000,
	timing_response: 1000,
	on_finish: appendCueData,
};


var practice_probe_block = {
	type: 'poldrack-categorize',
	stimulus: getPracticeProbe,
	key_answer: getResponse,
	choices: choices,
	data: {trial_id: "probe", exp_stage: "practice"},
	correct_text: '<div class = bottombox><div style="color:green"; style="color:green"; class = center-text>Correct!</div></div>',
	incorrect_text: '<div class = bottombox><div style="color:red"; style="color:red"; class = center-text>Incorrect</div></div>',
	timeout_message: '<div class = bottombox><div class = center-text>no response detected</div></div>',
	timing_stim: [2000],
	timing_response: [2000],
	timing_feedback_duration: 750,
	is_html: true,
	on_finish: appendPracticeProbeData,
};


/* create experiment definition array */
var directed_forgetting_experiment = [];

directed_forgetting_experiment.push(instruction_node);
directed_forgetting_experiment.push(test_img_block1);
directed_forgetting_experiment.push(test_img_block2);
// show one practice trial
directed_forgetting_experiment.push(start_fixation_block);
directed_forgetting_experiment.push(training_block);
directed_forgetting_experiment.push(cue_block);
directed_forgetting_experiment.push(fixation_block);
directed_forgetting_experiment.push(practice_probe_block);
directed_forgetting_experiment.push(ITI_fixation_block);
// start practice
directed_forgetting_experiment.push(start_practice_block);
for (i = 0; i < (practice_length-1); i++) {
	directed_forgetting_experiment.push(start_fixation_block);
	directed_forgetting_experiment.push(training_block);
	directed_forgetting_experiment.push(cue_block);
	directed_forgetting_experiment.push(fixation_block);
	directed_forgetting_experiment.push(practice_probe_block);
	directed_forgetting_experiment.push(ITI_fixation_block);
}

directed_forgetting_experiment.push(end_block);