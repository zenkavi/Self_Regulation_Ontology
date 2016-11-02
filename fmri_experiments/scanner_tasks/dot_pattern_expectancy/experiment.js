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

var randomDraw = function(lst) {
  var index = Math.floor(Math.random() * (lst.length))
  return lst[index]
}

var getInvalidCue = function() {
  return prefix + path + randomDraw(cues) + postfix
}

var getInvalidProbe = function() {
  return prefix + path + randomDraw(probes) + postfix
}

var getFeedback = function() {
  var curr_trial = jsPsych.progress().current_trial_global
  var curr_data = jsPsych.data.getData()[curr_trial - 1]
  var condition = curr_data.condition
  var response = curr_data.key_press
  var feedback_text = ''
  var correct = false
  var correct_response = choices[1]
  if (condition == "AX") {
    correct_response = choices[0]
  }
  if (response == correct_response) {
    correct = true
    feedback_text =  '<div class = centerbox><div style="color:green"; class = center-text>Correct!</div></div>'
  } else if (response == -1) {
    feedback_text =  '<div class = centerbox><div class = center-text>Respond Faster!</p></div>'
  } else {
    feedback_text = '<div class = centerbox><div style="color:red"; class = center-text>Incorrect</div></div>'
  }
  jsPsych.data.addDataToLastTrial({'correct': correct, 'correct_response': correct_response})
  return feedback_text
}

/* ************************************ */
/* Define experimental variables */
/* ************************************ */

// task specific variables
var current_trial = 0
var choices = [89, 71]
var correct_responses = [
  ["index finger", 89],
  ["middle finger", 71]
]
var exp_stage = 'tests'
var path = '/static/experiments/dot_pattern_expectancy/images/'
var prefix = '<div class = centerbox><div class = img-container><img src = "'
var postfix = '"</img></div></div>'
var cues = jsPsych.randomization.shuffle(['cue1.png', 'cue2.png', 'cue3.png', 'cue4.png',
  'cue5.png', 'cue6.png'
])
var probes = jsPsych.randomization.shuffle(['probe1.png', 'probe2.png', 'probe3.png', 'probe4.png',
  'probe5.png', 'probe6.png'
])
var images = []
for (var i = 0; i < cues.length; i++) {
  images.push(path + cues[i])
  images.push(path + probes[i])
}
//preload images
jsPsych.pluginAPI.preloadImages(images)

var valid_cue = cues.pop()
var valid_probe = probes.pop()

var trial_proportions = ["AX", "AX", "AX", "AX", "AX", "AX", "AX", "AX", "AX", "AX", "AX", "BX",
  "BX", "AY", "AY", "BY"
]
var block1_list = jsPsych.randomization.repeat(trial_proportions, 2)
var block2_list = jsPsych.randomization.repeat(trial_proportions, 2)
var block3_list = jsPsych.randomization.repeat(trial_proportions, 2)
var blocks = [block1_list, block2_list, block3_list]

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
 var test_intro_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text>Get ready!</p></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 1500, 
  timing_response: 1500,
  data: {
    trial_id: "rest block"
  },
  timing_post_trial: 500
};

/* define static blocks */
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

/// This ensures that the subject does not read through the instructions too quickly.  If they do it too quickly, then we will go over the loop again.
var instructions_block = {
  type: 'poldrack-text',
  data: {
    trial_id: "instruction"
  },
  text: '<div class = centerbox><p style = "font-size:40px" class = center-block-text>Target Pair (press index finger):</p><p class = center-block-text><img src = "/static/experiments/dot_pattern_expectancy/images/' +
    valid_cue +
    '" ></img>&nbsp&nbsp&nbsp...followed by...&nbsp&nbsp&nbsp<img src = "/static/experiments/dot_pattern_expectancy/images/' +
    valid_probe + '" ></img><br></br></p><p style = "font-size:40px" class = center-block-text>Otherwise press your middle finger</div>',
  cont_key: [32],
  timing_post_trial: 1000
};

var rest_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text>Take a break!<br>Next run will start in a moment</div></div>',
  is_html: true,
  choices: 'none',
  timing_response: 10000,
  data: {
    trial_id: "rest_block"
  },
  timing_post_trial: 1000
};

var feedback_block = {
  type: 'poldrack-single-stim',
  stimulus: getFeedback,
  is_html: true,
  choices: 'none',
  data: {
    trial_id: "feedback",
  },
  timing_post_trial: get_ITI,
  timing_stim: 1000,
  timing_response: 1000,
  on_finish: function(data) {
    console.log('Trial Num: ', current_trial)
    var correct = false
    if (data.stimulus.indexOf('Correct!') != -1) {
      correct = true
    }
    console.log('Correct Response? ', correct)
    jsPsych.data.addDataToLastTrial({
    	exp_stage: exp_stage,
    	trial_num: current_trial
    })
    current_trial += 1
  }
}

var fixation_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = fixation>+</div></div>',
  is_html: true,
  choices: 'none',
  data: {
    trial_id: "fixation",
  },
  timing_post_trial: 0,
  timing_stim: 2000,
  timing_response: 2000,
  on_finish: function() {
    jsPsych.data.addDataToLastTrial({exp_stage: exp_stage})
  }
}


/* define test block cues and probes*/
var A_cue = {
  type: 'poldrack-single-stim',
  stimulus: prefix + path + valid_cue + postfix,
  is_html: true,
  choices: 'none',
  data: {
    trial_id: "cue",
  },
  timing_stim: 500,
  timing_response: 500,
  timing_post_trial: 0,
  on_finish: function() {
    jsPsych.data.addDataToLastTrial({
    	exp_stage: exp_stage,
    	trial_num: current_trial
    })
  }
};

var other_cue = {
  type: 'poldrack-single-stim',
  stimulus: getInvalidCue,
  is_html: true,
  choices: 'none',
  data: {
    trial_id: "cue",
    exp_stage: "test"
  },
  timing_stim: 500,
  timing_response: 500,
  timing_post_trial: 0,
  on_finish: function() {
    jsPsych.data.addDataToLastTrial({
    	exp_stage: exp_stage,
    	trial_num: current_trial
    })
  }
};

var X_probe = {
  type: 'poldrack-single-stim',
  stimulus: prefix + path + valid_probe + postfix,
  is_html: true,
  choices: choices,
  data: {
    trial_id: "probe",
    exp_stage: "test"
  },
  timing_stim: 500,
  timing_response: 1500,
  timing_post_trial: 0,
  on_finish: function() {
    jsPsych.data.addDataToLastTrial({
    	exp_stage: exp_stage,
    	trial_num: current_trial
	})
  }
};

var other_probe = {
  type: 'poldrack-single-stim',
  stimulus: getInvalidProbe,
  is_html: true,
  choices: choices,
  data: {
    trial_id: "probe",
    exp_stage: "test"
  },
  timing_stim: 500,
  timing_response: 1500,
  timing_post_trial: 0,
  on_finish: function() {
    jsPsych.data.addDataToLastTrial({
    	exp_stage: exp_stage,
    	trial_num: current_trial
    })
  }
};

/* ************************************ */
/* Set up experiment */
/* ************************************ */

var dot_pattern_expectancy_experiment = []
dot_pattern_expectancy_experiment.push(instructions_block);
setup_fmri_intro(dot_pattern_expectancy_experiment, choices)

dot_pattern_expectancy_experiment.push(test_intro_block);
for (b = 0; b < blocks.length; b++) {
  var block = blocks[b]
  for (i = 0; i < block.length; i++) {
    switch (block[i]) {
      case "AX":
        cue = jQuery.extend(true, {}, A_cue)
        probe = jQuery.extend(true, {}, X_probe)
        cue.data.condition = "AX"
        probe.data.condition = "AX"
        break;
      case "BX":
        cue = jQuery.extend(true, {}, other_cue)
        probe = jQuery.extend(true, {}, X_probe)
        cue.data.condition = "BX"
        probe.data.condition = "BX"
        break;
      case "AY":
        cue = jQuery.extend(true, {}, A_cue)
        probe = jQuery.extend(true, {}, other_probe)
        cue.data.condition = "AY"
        probe.data.condition = "AY"
        break;
      case "BY":
        cue = jQuery.extend(true, {}, other_cue)
        probe = jQuery.extend(true, {}, other_probe)
        cue.data.condition = "BY"
        probe.data.condition = "BY"
        break;
    }
    dot_pattern_expectancy_experiment.push(cue)
    dot_pattern_expectancy_experiment.push(fixation_block)
    dot_pattern_expectancy_experiment.push(probe)
    dot_pattern_expectancy_experiment.push(feedback_block)
  }
  dot_pattern_expectancy_experiment.push(rest_block)
}
dot_pattern_expectancy_experiment.push(end_block)