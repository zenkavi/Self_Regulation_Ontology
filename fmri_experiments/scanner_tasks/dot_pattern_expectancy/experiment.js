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
  gap = randomExponential(1/2)*400
  if (gap > 10000) {
    gap = 10000
  } else if (gap < 0) {
    gap = 0
  } else {
    gap = Math.round(gap/1000)*1000
  }
  return 1500 + gap //500 (stim time) + 1000 (minimum ITI)
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

/* ************************************ */
/* Define experimental variables */
/* ************************************ */

// task specific variables
var current_trial = 0
var choices = [89, 71]
var correct_responses = [
  ["left button", 89],
  ["right button",71]
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
var valid_cue = cues.pop()
var valid_probe = probes.pop()

//preload images
var images = []
for (var i = 0; i < cues.length; i++) {
  images.push(path + cues[i])
  images.push(path + probes[i])
}
jsPsych.pluginAPI.preloadImages(images)

// set up blocks
var num_blocks = 4
var block_length = 40

var trial_proportions = ["AX", "AX", "AX", "AX", "AX", "AX", "AX", "AX", "AX", "AX", "AX", "BX",
  "BX", "BX", "AY", "AY", "AY", "BY", "BY", "BY"]
var blocks = []
for (b = 0; b < num_blocks; b++) {
  blocks.push(jsPsych.randomization.repeat(trial_proportions, block_length/trial_proportions.length))
}


/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
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

/* define static blocks */
 var end_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text><i>Fin</i></div></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 3000, 
  timing_response: 3000,
  data: {
    trial_id: "end",
    exp_id: 'dot_pattern_expectancy'
  },
  timing_post_trial: 0
};


 var instructions_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><p style = "font-size:40px" class = center-block-text>Target Pair (press left):</p><p class = center-block-text><img src = "/static/experiments/dot_pattern_expectancy/images/' +
    valid_cue +
    '" ></img>&nbsp&nbsp&nbsp...followed by...&nbsp&nbsp&nbsp<img src = "/static/experiments/dot_pattern_expectancy/images/' +
    valid_probe + '" ></img><br></br></p><p style = "font-size:40px" class = center-block-text>Otherwise press right</div>',
  is_html: true,
  choices: 'none',
  timing_stim: 20000, 
  timing_response: 20000,
  data: {
    trial_id: "instructions",
  },
  timing_post_trial: 0
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
  timing_response: get_ITI,
  timing_post_trial: 0,
  on_finish: function(data) {
    var correct_response = choices[1]
    if (data.condition == "AX") {
      correct_response = choices[0]
    }
    var correct = false
    if (data.key_press == correct_response) {
      correct = true
    }
    jsPsych.data.addDataToLastTrial({
      correct_response: correct_response,
      correct: correct,
    	exp_stage: exp_stage,
    	trial_num: current_trial
	   })
     console.log('Trial: ' + current_trial +
              '\nCorrect Response? ' + correct + '\n')
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
  timing_response: get_ITI,
  timing_post_trial: 0,
  on_finish: function(data) {
    var correct_response = choices[1]
    if (data.condition == "AX") {
      correct_response = choices[0]
    }
    var correct = false
    if (data.key_press == correct_response) {
      correct = true
    }
    jsPsych.data.addDataToLastTrial({
      correct_response: correct_response,
      correct: correct,
      exp_stage: exp_stage,
      trial_num: current_trial
     })
     console.log('Trial: ' + current_trial +
              '\nCorrect Response? ' + correct + '\n')
  }
};

/* ************************************ */
/* Set up experiment */
/* ************************************ */

var dot_pattern_expectancy_experiment = []
dot_pattern_expectancy_experiment.push(instructions_block);
setup_fmri_intro(dot_pattern_expectancy_experiment, choices)

for (b = 0; b < num_blocks; b++) {
  dot_pattern_expectancy_experiment.push(start_test_block);
  var block = blocks[b]
  for (i = 0; i < block_length; i++) {
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
  }
  if ((b+1)<num_blocks) {
    dot_pattern_expectancy_experiment.push(rest_block)
  }
}
dot_pattern_expectancy_experiment.push(end_block)