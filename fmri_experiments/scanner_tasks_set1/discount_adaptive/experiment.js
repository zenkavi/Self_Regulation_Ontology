/* ************************************ */
/* Define helper functions */
/* ************************************ */

var makeTrials = function(k) {

  var sampleAmt = function() {
    return Math.round(((Math.floor(Math.random() * (35 - 10)) + 10)+Math.random())*100)/100;
  }

  var times = function (n, iterator) {
    var accum = Array(Math.max(0, n));
    for (var i = 0; i < n; i++) accum[i] = iterator.call();
    return accum;
  }

  var make_adaptive_stims = function(k, trials_per_con = 36){
  
    //sample sooner amount

    var hard_patient_small_amt = times(trials_per_con/2,sampleAmt)
    var hard_impatient_small_amt = times(trials_per_con/2,sampleAmt)
    var easy_patient_small_amt = times(trials_per_con/2,sampleAmt)
    var easy_impatient_small_amt = times(trials_per_con/2,sampleAmt)
    //sample sooner delay (0 or 14)
    var hard_patient_sooner_del = Array(9).fill(0).concat(Array(9).fill(14))
    var hard_impatient_sooner_del = Array(9).fill(0).concat(Array(9).fill(14))
    var easy_patient_sooner_del = Array(9).fill(0).concat(Array(9).fill(14))
    var easy_impatient_sooner_del = Array(9).fill(0).concat(Array(9).fill(14))
    // sample later delay
    var hard_patient_later_del = Array(5).fill(14).concat(Array(9).fill(28)).concat(Array(4).fill(42))
    var hard_impatient_later_del = Array(5).fill(14).concat(Array(9).fill(28)).concat(Array(4).fill(42))
    var easy_patient_later_del = Array(5).fill(14).concat(Array(9).fill(28)).concat(Array(4).fill(42))
    var easy_impatient_later_del = Array(5).fill(14).concat(Array(9).fill(28)).concat(Array(4).fill(42))
    // calculate implied_k close/far to given k
    var hard_patient_k = Math.pow(10,(Math.log10(k)*0.9))
    var hard_impatient_k = Math.pow(10,(Math.log10(k)*1.1))
    var easy_patient_k = Math.pow(10,(Math.log10(k)*0.5))
    var easy_impatient_k = Math.pow(10,(Math.log10(k)*1.5))
    // solve for larger amount
    var hard_patient_large_amt = hard_patient_later_del.map(function(x){ return x*hard_patient_k + 1}).map(function (num, idx) {
  return num * hard_patient_small_amt[idx]}).map(function(x){return Math.round(x*100)/100})
    var hard_impatient_large_amt = hard_impatient_later_del.map(function(x){ return x*hard_impatient_k + 1}).map(function (num, idx) {
  return num * hard_impatient_small_amt[idx]}).map(function(x){return Math.round(x*100)/100})
    var easy_patient_large_amt = easy_patient_later_del.map(function(x){ return x*easy_patient_k + 1}).map(function (num, idx) {
  return num * easy_patient_small_amt[idx]}).map(function(x){return Math.round(x*100)/100})
    var easy_impatient_large_amt = easy_impatient_later_del.map(function(x){ return x*easy_impatient_k + 1}).map(function (num, idx) {
  return num * easy_impatient_small_amt[idx]}).map(function(x){return Math.round(x*100)/100})
    
    // output
    var options = {small_amt: hard_patient_small_amt.concat(hard_impatient_small_amt).concat(easy_patient_small_amt).concat(easy_impatient_small_amt),
              sooner_del: hard_patient_sooner_del.concat(hard_impatient_sooner_del).concat(easy_patient_sooner_del).concat(easy_impatient_sooner_del),
              large_amt: hard_patient_large_amt.concat(hard_impatient_large_amt).concat(easy_patient_large_amt).concat(easy_impatient_large_amt),
              later_del: hard_patient_later_del.concat(hard_impatient_later_del).concat(easy_patient_later_del).concat(easy_impatient_later_del),
              implied_k: Array(18).fill(hard_patient_k).concat(Array(18).fill(hard_impatient_k)).concat(Array(18).fill(easy_patient_k)).concat(Array(18).fill(easy_impatient_k)),
              trial_type: Array(18).fill('hard_patient').concat(Array(18).fill('hard_impatient')).concat(Array(18).fill('easy_patient')).concat(Array(18).fill('easy_impatient')),
              now_trial: Array(9).fill(1).concat(Array(9).fill(0)).concat(Array(9).fill(1).concat(Array(9).fill(0))).concat(Array(9).fill(1).concat(Array(9).fill(0)).concat(Array(9).fill(1).concat(Array(9).fill(0))))
            }
    
    return options
  }

  var options = make_adaptive_stims(k)


  var stim_html = []

  //loop through each option to create html
  for (var i = 0; i < options.small_amt.length; i++) {
    stim_html[i] =
        '<div class = dd-stim><div class = amtbox1 style = "color:white">$'+options.small_amt[i]+'</div><br><br>'+
        '<div class = delbox1 style = "color:white">'+ options.sooner_del[i]+' days</div><div class = amtbox2 style = "color:white">$'+options.large_amt[i]+'</div><br><br>'+
        '<div class = delbox2 style = "color:white">'+ options.later_del[i]+' days</div></div>'
  }

  data_prop = []

  for (var i = 0; i < options.small_amt.length; i++) {
    data_prop.push({
      small_amount: options.small_amt[i],
      large_amount: options.large_amt[i],
      later_delay: options.later_del[i], 
      implied_k: options.implied_k[i],
      trial_type: options.trial_type[i],
      now_trial: options.now_trial[i]
    });
  }

  trials = []

  for (var i = 0; i < stim_html.length; i++) { 
    trials.push({
      stimulus: stim_html[i],
      data: data_prop[i]
    });
  }
    
    return trials
 
 }

/* ************************************ */
/* Define experimental variables */
/* ************************************ */

// task specific variables
var choices = [89, 71]
var bonus_list = [] //keeps track of choices for bonus

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */

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
    k = parseInt(data.responses.slice(7, 10))
    trials = makeTrials(k)
  }
}

var instructions_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-block-text>Choose between <strong>$20 today</strong> or the presented option.<br><br><strong>Index:</strong> Accept option on screen (reject $20 today). <br><br><strong>Middle:</strong> Reject option on screen (accept $20 today)<br><br>We will start with a practice trial.</div></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 3500, 
  timing_response: 3500,
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
  stimulus: '<div class = centerbox><div class = center-block-text>Beginning task. Remember:<br><br><strong>Index:</strong> Accept option on screen (reject $20 today). <br><br><strong>Middle:</strong> Reject option on screen (accept $20 today).</div></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 2500, 
  timing_response: 2500,
  data: {
    trial_id: "instructions",
  },
  timing_post_trial: 500
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
    exp_id: 'discount_adaptive'
  },
  timing_post_trial: 0
};

//Set up experiment
var discount_adaptive_experiment = []
discount_adaptive_experiment.push(task_setup_block);
discount_adaptive_experiment.push(instructions_block);
discount_adaptive_experiment.push(practice_block);
discount_adaptive_experiment.push(start_test_block);
for (i = 0; i < options.small_amt.length; i++) {
  discount_adaptive_experiment.push(fixation_block)
  var test_block = {
  type: 'poldrack-single-stim',
  data: {
    trial_id: "stim",
    exp_stage: "test"
  },
  stimulus:trials[i].stimulus,
  data: trials[i].data,
  is_html: true,
  choices: choices,
  response_ends_trial: true,
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

  discount_adaptive_experiment.push(test_block)
}
discount_adaptive_experiment.push(end_block);