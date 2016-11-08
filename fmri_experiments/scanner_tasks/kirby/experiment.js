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
  gap = randomExponential(1/2)*1000
  if (gap > 5000) {
    gap = 5000
  } else if (gap < 500) {
    gap = 500
  }
  return gap
}

/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// task specific variables
var choices = [71,66]
var bonus_list = [] //keeps track of choices for bonus
//hard coded options in the amounts and order specified in Kirby and Marakovic (1996)
var options = {
  small_amt: [54, 55, 19, 31, 14, 47, 15, 25, 78, 40, 11, 67, 34, 27, 69, 49, 80, 24, 33, 28, 34,
    25, 41, 54, 54, 22, 20
  ],
  large_amt: [55, 75, 25, 85, 25, 50, 35, 60, 80, 55, 30, 75, 35, 50, 85, 60, 85, 35, 80, 30, 50,
    30, 75, 60, 80, 25, 55
  ],
  later_del: [117, 61, 53, 7, 19, 160, 13, 14, 192, 62, 7, 119, 186, 21, 91, 89, 157, 29, 14, 179,
    30, 80, 20, 111, 30, 136, 7
  ]
}

var stim_html = []

//loop through each option to create html
for (var i = 0; i < options.small_amt.length; i++) {
  stim_html[i] =
    "<div class = centerbox id='container'><p class = center-block-text>Please select the option that you would prefer pressing the <strong>left</strong> or <strong>right</strong> button:</p><div class='table'><div class='row'><div id = 'option'><center><font color='green'>$" +
    options.small_amt[i] +
    "<br>today</font></center></div><div id = 'option'><center><font color='green'>$" + options.large_amt[
      i] + "<br>" + options.later_del[i] + " days</font></center></div></div></div></div>"
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
  type: 'poldrack-text',
  data: {
    trial_id: "instruction"
  },
  text: '<div class = centerbox><div class = center-text style="font-size:40px">Choose your preferred choice using the left or right button.</div></div>',
    cont_key: [32],
  timing_post_trial: 1000
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
    exp_id: 'stroop'
  },
  timing_post_trial: 0
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
  timing_post_trial: 500
};

var test_block = {
  type: 'poldrack-single-stim',
  data: {
    trial_id: "stim",
    exp_stage: "test"
  },
  timeline: trials,
  is_html: true,
  choices: choices,
  response_ends_trial: true,
  timing_post_trial: get_ITI,
  //used new feature to include choice info in recorded data
  on_finish: function(data) {
    var choice = false;
    if (data.key_press == choices[1]) {
      choice = 'larger_later';
      bonus_list.push({'amount': data.large_amount, 'delay': data.later_delay})
    } else if (data.key_press == choices[0]) {
      choice = 'smaller_sooner';
      bonus_list.push({'amount': data.small_amount, 'delay': 0})
    }
    jsPsych.data.addDataToLastTrial({
      choice: choice
    });
  }
};



//Set up experiment
var kirby_experiment = []
kirby_experiment.push(instructions_block);
setup_fmri_intro(kirby_experiment, choices)
kirby_experiment.push(start_test_block);
kirby_experiment.push(test_block);
kirby_experiment.push(end_block);