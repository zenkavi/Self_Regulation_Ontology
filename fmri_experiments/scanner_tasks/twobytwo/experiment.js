/* ************************************ */
/* Define helper functions */
/* ************************************ */
var randomDraw = function(lst) {
  var index = Math.floor(Math.random() * (lst.length))
  return lst[index]
}

var getInstructFeedback = function() {
  return '<div class = centerbox><p class = center-block-text>' + feedback_instruct_text +
    '</p></div>'
}

// Task Specific Functions
var getKeys = function(obj) {
  var keys = [];
  for (var key in obj) {
    keys.push(key);
  }
  return keys
}

var genStims = function(n) {
  stims = []
  for (var i = 0; i < n; i++) {
    var number = randomDraw('12346789')
    var color = randomDraw(['orange', 'blue'])
    var stim = {
      number: parseInt(number),
      color: color
    }
    stims.push(stim)
  }
  return stims
}

var getCTI = function() {
  return CTI
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
  gap = randomExponential(1/2)*250
  if (gap > 10000) {
    gap = 10000
  } else if (gap < 0) {
    gap = 0
  } else {
    gap = Math.round(gap/1000)*1000
  }
  return 2000 + gap //1000 (stim time) + 1000 (minimum ITI)
 }



/* Index into task_switches using the global var current_trial. Using the task_switch and cue_switch
change the task. If "stay", keep the same task but change the cue based on "cue switch". 
If "switch", switch to the other task and randomly draw a cue_i
*/
var setStims = function() {
  var tmp;
  switch (task_switches[current_trial].task_switch) {
    case "stay":
      if (task_switches[current_trial].cue_switch == "switch") {
        cue_i = 1 - cue_i
      }
      break
    case "switch":
      cue_i = randomDraw([0, 1])
      if (curr_task == "color") {
        curr_task = "magnitude"
      } else {
        curr_task = "color"
      }
      break
  }
  curr_cue = tasks[curr_task].cues[cue_i]
  curr_stim = stims[current_trial]
  CTI = task_switches[current_trial].CTI
}

var getCue = function() {
  var cue_html = '<div class = upperbox><div class = "center-text" >' + curr_cue +
    '</div></div><div class = lowerbox><div class = fixation>+</div></div>'
  return cue_html
}

var getStim = function() {
  var stim_html = '<div class = upperbox><div class = "center-text" >' + curr_cue +
    '</div></div><div class = lowerbox><div class = "center-text" style=color:' + curr_stim.color +
    ';>' + curr_stim.number + '</div>'
  return stim_html
}

//Returns the key corresponding to the correct response for the current
// task and stim
var getResponse = function() {
  switch (curr_task) {
    case 'color':
      if (curr_stim.color == 'orange') {
        return response_keys.key[0]
      } else {
        return response_keys.key[1]
      }
      break;
    case 'magnitude':
      if (curr_stim.number > 5) {
        return response_keys.key[0]
      } else {
        return response_keys.key[1]
      }
      break;
  }
}


/* Append gap and current trial to data and then recalculate for next trial*/
var appendData = function() {
  var trial_num = current_trial //current_trial has already been updated with setStims, so subtract one to record data
  var task_switch = task_switches[trial_num]
  jsPsych.data.addDataToLastTrial({
    cue: curr_cue,
    stim_color: curr_stim.color,
    stim_number: curr_stim.number,
    task: curr_task,
    task_switch: task_switch.task_switch,
    cue_switch: task_switch.cue_switch,
    trial_num: trial_num
  })
}

/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// task specific variables
var response_keys = jsPsych.randomization.repeat([{
  key: 89,
  key_name: 'Index Finger'
}, {
  key: 71,
  key_name: 'Middle finger'
}], 1, true)
var choices = response_keys.key
var num_blocks = 3
var block_length = 80
var test_length = num_blocks * test_length

//set up block stim. correct_responses indexed by [block][stim][type]
var tasks = {
  color: {
    task: 'color',
    cues: ['Color', 'Orange-Blue']
  },
  magnitude: {
    task: 'magnitude',
    cues: ['Magnitude', 'High-Low']
  }
}

var task_switch_types = ["stay", "switch"]
var cue_switch_types = ["stay", "switch"]
var CTIs = [100, 900]
var task_switches = []
for (var t = 0; t < task_switch_types.length; t++) {
  for (var c = 0; c < cue_switch_types.length; c++) {
    for (var j = 0; j < CTIs.length; j++) {
      task_switches.push({
        task_switch: task_switch_types[t],
        cue_switch: cue_switch_types[c],
        CTI: CTIs[j]
      })
    }
  }
}
var task_switches = jsPsych.randomization.repeat(task_switches, test_length / 8)
var testStims = genStims(test_length)
var stims = testStims
var curr_task = randomDraw(getKeys(tasks))
var cue_i = randomDraw([0, 1]) //index for one of two cues of the current task
var curr_cue = tasks[curr_task].cues[cue_i] //object that holds the current cue, set by setStims()
var curr_stim = 'na' //object that holds the current stim, set by setStims()
var current_trial = 0
var CTI = 0 //cue-target-interval
var exp_stage = 'test' // defines the exp_stage, switched by start_test_block






/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
var prompt_task_list = '<strong>Color</strong> or <strong>Orange-Blue</strong>: ' +
  response_keys.key_name[0] + ' if orange and ' + response_keys.key_name[1] + ' if blue.' +
  '<br><br><strong>Parity</strong> or <strong>Odd-Even</strong>: ' + response_keys.key_name[0] +
  ' if even and ' + response_keys.key_name[1] + ' if odd.' +
  '<br><br><strong>Magnitude</strong> or <strong>High-Low</strong>: ' + response_keys.key_name[0] +
  ' if >5 and ' + response_keys.key_name[1] + ' if <5.'

var instructions_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text style="font-size:40px">' + prompt_task_list + '</div></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 20000, 
  timing_response: 20000,
  data: {
    trial_id: "instructions",
  },
  timing_post_trial: 0
};

 var end_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text><i>Fin</i></div></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 3000, 
  timing_response: 3000,
  data: {
    trial_id: "end",
    exp_id: 'twobytwo'
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

/* define test blocks */
var setStims_block = {
  type: 'call-function',
  data: {
    trial_id: "set_stims"
  },
  func: setStims,
  timing_post_trial: 0
}

var fixation_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = upperbox><div class = fixation>+</div></div><div class = lowerbox><div class = fixation>+</div></div>',
  is_html: true,
  choices: 'none',
  data: {
    trial_id: "fixation"
  },
  timing_post_trial: 0,
  timing_response: 500,
  prompt: '<div class = promptbox>' + prompt_task_list + '</div>',
  on_finish: function() {
    jsPsych.data.addDataToLastTrial({
      exp_stage: exp_stage
    })
  }
}

var cue_block = {
  type: 'poldrack-single-stim',
  stimulus: getCue,
  is_html: true,
  choices: 'none',
  data: {
    trial_id: 'cue'
  },
  timing_response: getCTI,
  timing_stim: getCTI,
  timing_post_trial: 0,
  prompt: '<div class = promptbox>' + prompt_task_list + '</div>',
  on_finish: function() {
    jsPsych.data.addDataToLastTrial({
      exp_stage: exp_stage
    })
    appendData()
  }
};

var test_block = {
  type: 'poldrack-single-stim',
  stimulus: getStim,
  is_html: true,
  key_answer: getResponse,
  choices: choices,
  data: {
    trial_id: 'stim',
    exp_stage: 'test'
  },
  timing_post_trial: 0,
  timing_response: get_ITI,
  timing_stim: 1000,
  prompt: '<div class = upperbox><div class = fixation>+</div></div><div class = lowerbox><div class = fixation>+</div></div>',
  prompt: '<div class = promptbox>' + prompt_task_list + '</div>',
  on_finish: function(data) {
    appendData()
    correct_response = getResponse()
    correct = false
    if (data.key_press === correct_response) {
      correct = true
    }
    jsPsych.data.addDataToLastTrial({
      'correct_response': correct_response,
      'correct': correct
    })
    console.log('Trial: ' + current_trial +
              '\nCorrect Response? ' + correct + '\n')
    current_trial += 1
  }
}


/* create experiment definition array */
var threebytwo_experiment = [];
threebytwo_experiment.push(instructions_block);
setup_fmri_intro(threebytwo_experiment, choices)
for (var b = 0; b < num_blocks; b++) {
	threebytwo_experiment.push(start_test_block)
  threebytwo_experiment.push(fixation_block)
	for (var i = 0; i < block_length; i++) {
	  threebytwo_experiment.push(setStims_block)
	  threebytwo_experiment.push(cue_block);
	  threebytwo_experiment.push(test_block);
	}
	if ((b+1)<num_blocks) {
		threebytwo_experiment.push(rest_block)
	}
}
threebytwo_experiment.push(end_block)
