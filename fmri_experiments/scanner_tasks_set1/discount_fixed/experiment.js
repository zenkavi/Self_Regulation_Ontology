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
  large_amt: [23.25,20.25,21.75,23.75,25,25.75,22.75,21.5,22,22.25,23,26,27.25,22.75,23,23.25,21.75,21,22.5,21.25,26.5,26,28,21.5,28.25,23,22.5,25,22.5,25.75,30,20.75,29.75,27.5,27.75,22.25,24.75,37,23,23.5,22.5,21.5,23.25,35,23.25,34.75,39.25,40.25,28.25,43,42.25,38.5,38.5,51.5,50.5,47.75,53.25,28.5,23.25,33.5,36.25,21.25,40.25,49.25,35.25,50.5,34.25,31,24.25,71,59.75,49.5,63.75,75,54.75,71.5,88.25,82.75,88.25,78.75,28.25,59,34.75,93,55.5,56,65.75,21.75,94.5,80.25,123.25,129,121.5,119.75,88.5,122.25,104.25,103.75,108,55,27,188.25,63.25,71,194.25,101.25,131.75,32.5,141.75,218.5,156.5,223,255.75,22.25,291.25,142.75,108.25,172.25,183,216],
  later_del: [80,4,38,84,105,118,53,27,35,40,51,100,116,42,43,46,23,11,31,13,79,71,91,16,87,30,24,47,22,50,86,6,77,57,58,15,32,113,19,21,14,8,18,81,17,74,93,95,37,101,94,75,73,120,112,98,114,28,10,41,48,3,56,78,39,76,34,25,9,110,83,59,85,103,63,90,115,102,107,89,12,55,20,96,45,44,54,2,82,64,106,108,97,92,61,88,70,67,68,26,5,117,29,33,109,49,65,7,66,104,69,99,111,1,119,52,36,60,62,72]
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
setup_fmri_intro(discount_fixed_experiment, choices)
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