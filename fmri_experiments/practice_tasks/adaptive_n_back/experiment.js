/* ************************************ */
/* Define helper functions */
/* ************************************ */
var getInstructFeedback = function() {
	return '<div class = centerbox><p class = center-block-text>' + feedback_instruct_text +
		'</p></div>'
}

var randomDraw = function(lst) {
	var index = Math.floor(Math.random() * (lst.length))
	return lst[index]
};


var getStim = function() {
	var trial_type = target_trials.shift()
	var targets = letters.filter(function(x) { return x.toLowerCase() == target.toLowerCase()})
	var non_targets = letters.filter(function(x) { return x.toLowerCase() != target.toLowerCase()})
	if (trial_type === 'target') {
		curr_stim = randomDraw(targets)
	} else {
		curr_stim = randomDraw(non_targets)
	}
	stims.push(curr_stim)
	return '<div class = "centerbox"><div class = "center-text">' + curr_stim + '</div></div>'
}

var getData = function() {
	return {
		trial_id: "stim",
		exp_stage: "adaptive",
		load: delay,
		target: target,
		block_num: current_block
	}
}

var getText = function() {
	return '<div class = "centerbox"><p class = "block-text">In these next blocks, you should press the left arrow key when the current letter matches the letter that appeared ' +
	delay +
		' trials before. Otherwise press the down arrow key</p><p class = center-block-text>Press <strong>enter</strong> to begin.</p></div>'
}

/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// generic task variables
var run_attention_checks = false
var attention_check_thresh = 0.65
var sumInstructTime = 0 //ms
var instructTimeThresh = 0 ///in seconds
var credit_var = true //default to true

// task specific variables
var letters = 'bBdDgGtTvV'.split("")
var num_blocks = 20 // number of adaptive blocks
var base_num_trials = 20 // total num_trials = base + load 
var control_num_trials = 42
var control_before = Math.round(Math.random()) //0 control comes before test, 1, after
var block_acc = 0 // record block accuracy to determine next blocks delay
var delay = 2 // starting delay
var trials_left = 0 // counter used by adaptive_test_node
var target_trials = [] // array defining whether each trial in a block is a target trial
var current_trial = 0
var current_block = 0  
var block_trial = 0
var target = ""
var curr_stim = ''
var stims = [] //hold stims per block

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
// Set up attention check node

/* define static blocks */
var feedback_instruct_text =
	'Welcome to the practice phase of this experiment. Press <strong>enter</strong> to begin.'
var feedback_instruct_block = {
	type: 'poldrack-text',
	cont_key: [13],
	data: {
		trial_id: 'instruction'
	},
	text: getInstructFeedback,
	timing_post_trial: 0,
	timing_response: 180000
};
/// This ensures that the subject does not read through the instructions too quickly.  If they do it too quickly, then we will go over the loop again.
var instructions_block = {
	type: 'poldrack-instructions',
	pages: [
		'<div class = "centerbox"><p class = "block-text">In this task you will see a sequence of letters presented one at a time. Your job is to respond by pressing the <strong>left arrow key</strong> when the letter matches the same letter that occured some number of trials before (the number of trials is called the "delay"), otherwise you should press the <strong>down arrow key</strong>. The letters will be both lower and upper case. You should ignore the case (so "t" matches "T").</p><p class = block-text>The specific delay you should pay attention to will differ between blocks of trials, and you will be told the delay before starting a block.</p><p class = block-text>For instance, if the delay is 2, you are supposed to press the left arrow key when the current letter matches the letter that occurred 2 trials ago. If you saw the sequence: g...G...v...T...b...t...b, you would press the left arrow key on the last "t" and the last "b" and the down arrow key for every other letter.</p></div>'
	],
	data: {
		trial_id: 'instruction'
	},
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

var end_block = {
	type: 'poldrack-text',
	text: '<div class = "centerbox"><p class = "center-block-text">Thanks for completing this practice!</p><p class = center-block-text>Press <strong>enter</strong> to begin.</p></div>',
	cont_key: [13],
	data: {
		trial_id: "end",
    	exp_id: 'adaptive_n_back'
	},
	timing_response: 180000,
	timing_post_trial: 0,
};

var start_practice_block = {
	type: 'poldrack-text',
	text: '<div class = centerbox><p class = block-text>For practice, you should press the left arrow key when the current letter matches the letter that appeared <strong>1 trial before</strong>. Otherwise press the down arrow key</p><p class = center-block-text>You will receive feedback about whether you were correct or not during practice. There will be no feedback during the main experiment. Press <strong>enter</strong> to begin practice.</p></div>',
	cont_key: [13],
	data: {
		trial_id: "instruction"
	},
	timing_response: 180000,
	timing_post_trial: 1000
};



//Setup 1-back practice
practice_trials = []
for (var i = 0; i < (base_num_trials + 1); i++) {
	var stim = randomDraw(letters)
	stims.push(stim)
	if (i >= 1) {
		target = stims[i - 1]
	}
	if (stim.toLowerCase() == target.toLowerCase()) { 
		correct_response = 37
	} else {
		correct_response = 40
	}
	var practice_block = {
		type: 'poldrack-categorize',
		is_html: true,
		stimulus: '<div class = centerbox><div class = center-text>' + stim + '</div></div>',
		key_answer: correct_response,
		data: {
			trial_id: "stim",
			exp_stage: "practice",
			stim: stim,
			target: target
		},
		correct_text: '<div class = centerbox><div style="color:green;font-size:60px"; class = center-text>Correct!</div></div>',
		incorrect_text: '<div class = centerbox><div style="color:red;font-size:60px"; class = center-text>Incorrect</div></div>',
		timeout_message: '<div class = centerbox><div style="font-size:60px" class = center-text>Respond Faster!</div></div>',
		timing_feedback_duration: 500,
		show_stim_with_feedback: false,
		choices: [37,40],
		timing_stim: 500,
		timing_response: 2000,
		timing_post_trial: 500
	};
	practice_trials.push(practice_block)
}

	
//Set up experiment
var adaptive_n_back_experiment = []
adaptive_n_back_experiment.push(instruction_node);
adaptive_n_back_experiment.push(start_practice_block)
adaptive_n_back_experiment = adaptive_n_back_experiment.concat(practice_trials)
adaptive_n_back_experiment.push(end_block)