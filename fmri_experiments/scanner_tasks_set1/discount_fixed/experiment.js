/* ************************************ */
/* Define helper functions */
/* ************************************ */

ITIs = [0.816,0.068,0.136,0.408,0.0,0.476,0.408,0.068,0.0,1.7,0.204,0.544,0.544,0.68,0.476,0.272,0.272,0.68,0.68,0.748,0.068,0.0,0.408,0.068,1.7,0.408,0.748,0.544,0.136,0.204,0.136,0.068,0.068,0.408,0.34,0.748,0.272,1.496,1.088,0.068,0.612,1.156,0.068,0.272,0.544,0.34,0.68,0.544,0.476,0.476,0.816,0.0,0.544,0.204,0.34,2.244,0.0,0.068,0.204,0.408,0.068,0.272,0.136,0.068,0.952,0.884,0.068,1.768,1.02,0.068,0.136,0.136,0.612,0.816,0.068,0.068,0.068,0.34,0.0,1.156,0.408,0.136,0.0,0.0,0.612,1.632,0.34,0.068,0.204,1.36,0.136,0.816,0.476,0.068,1.02,1.088,1.292,0.068,0.612,0.0,0.34,0.068,0.884,0.544,0.204,0.408,0.408,0.34,0.612,0.612,0.204,0.34,0.34,2.38,0.068,0.272,2.516,0.204,0.68,0.544]

var get_ITI = function() {
  return 5500 + ITIs.shift()*1000 // 500 minimum ITI
 }

/* ************************************ */
/* Define experimental variables */
/* ************************************ */

// task specific variables
var choices = [89, 71]
var bonus_list = [] //keeps track of choices for bonus
//hard coded options 
var options = {
  small_amt: [20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20],
  large_amt: [77,50,43,71,75,36,36,50,73,33,69,63,39,66,23,41,45,39,44,80,30,23,24,40,25,57,49,78,67,25,27,44,54,28,45,71,32,31,43,78,67,79,66,22,54,77,24,80,77,23,55,28,83,29,58,32,33,23,55,22,27,58,28,65,25,74,74,56,41,65,62,71,23,48,64,81,58,60,66,25,35,29,32,45,30,27,76,23,67,56,58,72,34,28,43,41,34,65,82,52,39,24,61,31,54,24,67,45,29,50,41,33,70,42,25,85,58,24,75,49],
  later_del: [161, 59, 38,176, 60,114,140, 41, 44, 78,172, 37, 95, 58, 61,129,157,161,143,145, 36,134, 87,159,141, 89,140,108, 74,108, 91, 92, 23, 59,107, 88,111, 76,120,129, 41, 92,145, 21, 22,142,139, 39, 24,110,163,132,105, 44, 72, 57,103, 42,107, 27,142, 36,163,156,176, 52,121,172, 55,137,180, 76,176,128, 89, 58, 55,124, 71, 70,171, 75, 74, 72, 23,130,161, 77,103, 93, 53, 95,176,123, 19,174, 43,159, 74,139,104, 38,141, 74,126,125, 23, 26, 58,129, 27,163,108,145, 74,129, 73, 89, 41,172]
}

var stim_html = []

//loop through each option to create html
for (var i = 0; i < options.small_amt.length; i++) {
  stim_html[i] =
      '<div class = dd-stim><div class = amtbox style = "color:white">$'+options.large_amt[i]+'</div><br><br>'+
      '<div class = delbox style = "color:white">'+ options.later_del[i]+' days</div></div>'
}

data_prop = []

for (var i = 0; i < options.small_amt.length; i++) {
  data_prop.push({
    small_amount: options.small_amt[i],
    large_amount: options.large_amt[i],
    later_delay: options.later_del[i]
  });
}

trials = []

//used new features to include the stimulus properties in recorded data
for (var i = 0; i < stim_html.length; i++) { 
  trials.push({
    stimulus: stim_html[i],
    data: data_prop[i]
  });
}

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */

var instructions_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text>Choose between <strong>$20 today</strong> or the presented option.<br><br><strong>Index:</strong> Accept option on screen (reject $20 today). <br><br><strong>Middle:</strong> Reject option on screen (accept $20 today)<br><br>We will start with a practice trial.</div></div>',
  is_html: true,
  timing_stim: -1, 
  timing_response: -1,
  response_ends_trial: true,
  choices: [32],
  data: {
    trial_id: "instructions",
  },
  timing_post_trial: 500
};

var practice_block = {
  type: 'poldrack-single-stim',
  data: {
    trial_id: "stim",
    exp_stage: "practice"
  },
  stimulus: '<div class = dd-stim><div class = amtbox style = "color:white">$53.75</div><br><br><div class = delbox style = "color:white">34 days</div></div>',
  is_html: true,
  choices: choices,
  response_ends_trial: true, 
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

var fixation_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = black-centerbox><div class = fixation style = "color:white">+</div></div>',
  is_html: true,
  choices: 'none',
  data: {
    trial_id: "fixation"
  },
  timing_post_trial: 500,
  timing_stim: 500,
  timing_response: 500,
  on_finish: function() {
    jsPsych.data.addDataToLastTrial({'exp_stage': 'test'})
  },
}

var end_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text><i>Fin</i></div></div>',
  is_html: true,
  choices: [32],
  timing_response: -1,
  response_ends_trial: true,
  data: {
    trial_id: "end",
    exp_id: 'discount_fixed'
  },
  timing_post_trial: 0
};

//Set up experiment
var discount_fixed_experiment = []
test_keys(discount_fixed_experiment,choices)
discount_fixed_experiment.push(instructions_block);
discount_fixed_experiment.push(practice_block);
setup_fmri_intro(discount_fixed_experiment)
discount_fixed_experiment.push(start_test_block);
for (i = 0; i < options.small_amt.length; i++) {
  discount_fixed_experiment.push(fixation_block)
  var test_block = {
  type: 'poldrack-single-stim',
  data: {
    trial_id: "stim",
    exp_stage: "test"
  },
  stimulus:trials[i].stimulus,
  timing_stim: 5000,
  timing_response: get_ITI,  
  data: trials[i].data,
  is_html: true,
  choices: choices,
  response_ends_trial: false,
  timing_post_trial: 0,
  on_finish: function(data) {
    var choice = false;
    if (data.key_press == 89) {
      choice = 'larger_later';
      bonus_list.push({'amount': data.large_amount, 'delay': data.later_delay})
    } else if (data.key_press == 71) {
      choice = 'smaller_sooner';
      bonus_list.push({'amount': data.small_amount, 'delay': 0})
    }
    jsPsych.data.addDataToLastTrial({
      choice: choice
    });
  }
};

  discount_fixed_experiment.push(test_block)
}
discount_fixed_experiment.push(end_block);