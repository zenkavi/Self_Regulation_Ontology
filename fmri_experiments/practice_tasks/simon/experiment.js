/* ************************************ */
/* Define helper functions */
/* ************************************ */

var post_trial_gap = function() {
  gap = Math.floor(Math.random() * 500) + 500
  return gap;
}

/* Append gap and current trial to data and then recalculate for next trial*/
var appendData = function() {
  jsPsych.data.addDataToLastTrial({
    trial_num: current_trial
  })
  current_trial = current_trial + 1
}

var appendTestData = function(data) {
  correct = false
  if (data.key_press == data.correct_response) {
  	correct = true
  }
  jsPsych.data.addDataToLastTrial({
  	correct: correct,
    trial_num: current_trial
  })
  current_trial = current_trial + 1
}

var getInstructFeedback = function() {
    return '<div class = centerbox><p class = center-block-text>' + feedback_instruct_text +
      '</p></div>'
  }
  /* ************************************ */
  /* Define experimental variables */
  /* ************************************ */
  // generic task variables
var run_attention_checks = false
var attention_check_thresh = 0.45
var sumInstructTime = 0 //ms
var instructTimeThresh = 0 ///in seconds
var credit_var = true

// task specific variables
var correct_responses = jsPsych.randomization.shuffle([["left arrow", 37],["right arrow", 39]])
var choices = [37, 39]
var current_trial = 0
var gap = Math.floor(Math.random() * 2000) + 1000
var test_stimuli = [{
  stimulus: '<div class = centerbox><div class = simon_left id = stim1></div></div>',
  data: {
    correct_response: correct_responses[0][1],
    stim_side: 'left',
    stim_color: 'red', 
    condition: correct_responses[0][1] == 37 ? 'congruent' : 'incongruent'
  },
  key_answer: correct_responses[0][1]
}, {
  stimulus: '<div class = centerbox><div class = simon_right id = stim1></div></div>',
  data: {
    correct_response: correct_responses[0][1],
    stim_side: 'right',
    stim_color: 'red', 
    condition: correct_responses[0][1] == 37 ? 'incongruent' : 'congruent'
  },
  key_answer: correct_responses[0][1]
}, {
  stimulus: '<div class = centerbox><div class = simon_left id = stim2></div></div>',
  data: {
    correct_response: correct_responses[1][1],
    stim_side: 'left',
    stim_color: 'blue', 
    condition: correct_responses[0][1] == 37 ? 'incongruent' : 'congruent'
  },
  key_answer: correct_responses[1][1]
}, {
  stimulus: '<div class = centerbox><div class = simon_right id = stim2></div></div>',
  data: {
    correct_response: correct_responses[1][1],
    stim_side: 'right',
    stim_color: 'blue', 
    condition: correct_responses[0][1] == 37 ? 'congruent' : 'incongruent'
  },
  key_answer: correct_responses[1][1]
}];

var practice_trials = jsPsych.randomization.repeat(test_stimuli, 5);


/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
/* define static blocks */
var feedback_instruct_text =
  'Welcome to the practice phase of this experiment. Press <strong>enter</strong> to begin.'
var feedback_instruct_block = {
  type: 'poldrack-text',
  data: {
    trial_id: "instruction"
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
    trial_id: "instruction"
  },
  pages: [
    '<div class = centerbox><p class = block-text>On each trial of this task, a red or blue box will appear. If you see a <font color="red">red</font> box, press the ' +
    correct_responses[0][0] + '. If you see a <font color="blue">blue</font> box, press the ' + correct_responses[1][0] + '.</p><p class = block-text>During practice you will get feedback about whether you responded correctly. We will begin practice after you end the instructions.</p></div>',
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
      feedback_instruct_text =
        'Done with instructions. Press <strong>enter</strong> to continue.'
      return false
    }
  }
}

var end_block = {
  type: 'poldrack-text',
  data: {
    trial_id: "end",
    exp_id: 'simon'
  },
  timing_response: 180000,
  text: '<div class = centerbox><p class = center-block-text>Thanks for completing this practice!</p><p class = center-block-text>Press <strong>enter</strong> to continue.</p></div>',
  cont_key: [13],
  timing_post_trial: 0,
};



/* define practice block */
var practice_block = {
  type: 'poldrack-categorize',
  timeline: practice_trials,
  is_html: true,
  data: {
    trial_id: "stim",
    exp_stage: "practice"
  },
  correct_text: '<div class = centerbox><div style="color:green"; class = center-text>Correct!</div></div>',
  incorrect_text: '<div class = centerbox><div style="color:red"; class = center-text>Incorrect</div></div>',
  timeout_message: '<div class = centerbox><div class = center-text>Respond faster!</div></div>',
  choices: choices,
  timing_response: 2000,
  timing_stim: 2000,
  timing_feedback_duration: 1000,
  show_stim_with_feedback: false,
  timing_post_trial: post_trial_gap,
  on_finish: appendData
}


/* create experiment definition array */
var simon_experiment = [];
simon_experiment.push(instruction_node);
simon_experiment.push(practice_block);
simon_experiment.push(end_block)