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
  gap = randomExponential(1/2)*900
  if (gap > 10000) {
    gap = 10000
  } else if (gap < 0) {
    gap = 0
  } else {
    gap = Math.round(gap/1000)*1000
  }
  return 2000 + gap //1000 (feedback time) + 1000 (minimum ITI)
 }

var getStim = function() {
  var ref_board = makeBoard('your_board', curr_placement, 'ref')
  var goal_state_board = makeBoard('peg_board', problems[problem_i].goal_state.problem)
  var canvas = '<div class = watt_canvas><div class="watt_vertical_line"></div></div>'
  var hold_box;
  if (held_ball !== 0) {
    ball = colors[held_ball - 1]
    hold_box = '<div class = watt_hand_box><div class = "watt_hand_ball watt_' + ball +
      '"><div class = watt_ball_label>' + ball[0] +
      '</div></div></div><div class = watt_hand_label><strong>Ball in Hand</strong></div>'
  } else {
    hold_box =
      '<div class = watt_hand_box></div><div class = watt_hand_label><strong>Ball in Hand</strong></div>'
  }
  return canvas + ref_board + goal_state_board + hold_box
}

var getFB = function() {
  var data = jsPsych.data.getLastTrialData()
  var goal_state = data.goal_state
  var isequal = true
  correct = false
  for (var i = 0; i < goal_state.length; i++) {
    isequal = arraysEqual(goal_state[i], data.current_position[i])
    if (isequal === false) {
      break;
    }
  }
  var feedback;
  if (isequal === true) {
    feedback = "Complete"
    correct = true
  } else {
    feedback = "Didn't get that one."
  }
  var ref_board = makeBoard('your_board', curr_placement)
  var goal_state_board = makeBoard('peg_board', goal_state)
  var canvas = '<div class = watt_canvas><div class="watt_vertical_line"></div></div>'
  var feedback_box = '<div class = watt_feedbackbox><p class = center-text>' + feedback +
    '</p></div>'
  return canvas + ref_board + goal_state_board + feedback_box
}

var pegClick = function(choice) {
  var peg = curr_placement[choice]
  var ball_location = 0
  if (held_ball === 0) {
    for (var i = peg.length - 1; i >= 0; i--) {
      if (peg[i] !== 0) {
        held_ball = peg[i]
        peg[i] = 0
        num_moves += 1
        break;
      }
    }
  } else {
    var open_spot = peg.indexOf(0)
    if (open_spot != -1) {
      peg[open_spot] = held_ball
      held_ball = 0
    }
  }
}

var makeBoard = function(container, ball_placement, board_type) {
  var board = '<div class = watt_' + container + '><div class = watt_base></div>'
  if (container == 'your_board') {
    board += '<div class = watt_board_label><strong>Your Board</strong></div>'
  } else {
    board += '<div class = watt_board_label><strong>Target Board</strong></div>'
  }
  for (var p = 0; p < 3; p++) {
    board += '<div id = watt_peg_' + (p + 1) + '><div class = watt_peg></div></div>' //place peg
      //place balls
    if (board_type == 'ref') {
      if (ball_placement[p][0] === 0 & held_ball === 0) {
        board += '<div id = watt_peg_' + (p + 1) + ' onclick = "pegClick(this.id)">'
      } else if (ball_placement[p].slice(-1)[0] !== 0 & held_ball !== 0) {
        board += '<div id = watt_peg_' + (p + 1) + ' onclick = "pegClick(this.id)">'
      } else {
        board += '<div class = special id = watt_peg_' + (p + 1) + ' onclick = "pegClick(this.id)">'
      }
    } else {
      board += '<div id = watt_peg_' + (p + 1) + ' >'
    }
    var peg = ball_placement[p]
    for (var b = 0; b < peg.length; b++) {
      if (peg[b] !== 0) {
        ball = colors[peg[b] - 1]
        board += '<div class = "watt_ball watt_' + ball + '"><div class = watt_ball_label>' + ball[0] +
          '</div></div>'
      }
    }
    board += '</div>'
  }
  board += '</div>'
  return board
}

var arraysEqual = function(arr1, arr2) {
  if (arr1.length !== arr2.length)
    return false;
  for (var i = arr1.length; i--;) {
    if (arr1[i] !== arr2[i])
      return false;
  }
  return true;
}

var reset_problem = function() {
    colors = jsPsych.randomization.shuffle(['Green', 'Red', 'Blue'])
    held_ball = 0
    problem_start_time = new Date()
    num_moves = 0;
    if (problem_i < problems.length) {
      curr_placement = jQuery.extend(true, [], problems[problem_i].start_state)
    }
}

var get_hand_choices = function() {
  var trial_choices = []
  var sum = 0
  for (var i=0; i<choices.length; i++) {
    sum = curr_placement[i].reduce(function(a,b) {return a+b;}, 0)
    if (sum > 0) {
      trial_choices.push(choices[i])
    }
  }
  return trial_choices
}

var get_board_choices = function() {
  var trial_choices = []
  for (var i=0; i<choices.length; i++) {
    if (curr_placement[i].indexOf(0) != -1) {
      trial_choices.push(choices[i])
    }
  }
  return trial_choices
}

/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// task specific variables
var choices = [89, 71, 82]
var correct = false
var colors = ['Green', 'Red', 'Blue']
var problem_i = 0
var num_moves = 0 //tracks number of moves for a problem
var held_ball = 0
var ref_board = ''
// timing variables
var problem_start_time = 0
var start_time = new Date()
var task_limit = 600000

var problems = []
var test_problems = []
var practice_problems = []
// Problems can be either partially ambiguous or unambiguous (goal hierarchy)
// and either with and intermediate step or without (search depth)
// In this implementation all practice problems are unambiguous (UA) and
// all test are partially ambiguous (PA)
// set up practice problems
practice_problems = [
  {'start_state': [
      [1, 0, 0],
      [2, 3, 0],
      [0, 0, 0]
  ], 
  'goal_state': {
    'condition': 'UA_with_intermeidate',
    'problem': [
      [1, 2, 3],
      [0, 0, 0],
      [0, 0, 0]
    ]
  }},
  {'start_state': [
      [1, 2, 0],
      [3, 0, 0],
      [0, 0, 0]
  ], 
  'goal_state': {
    'condition': 'UA_with_intermeidate',
    'problem': [
      [1, 3, 2],
      [0, 0, 0],
      [0, 0, 0]
    ]
  }},
  {'start_state': [
      [0, 0, 0],
      [1, 2, 0],
      [3, 0, 0]
  ], 
  'goal_state': {
    'condition': 'UA_without_intermeidate',
    'problem': [
      [3, 2, 1],
      [0, 0, 0],
      [0, 0, 0]
    ]
  }},
  {'start_state': [
      [0, 0, 0],
      [1, 2, 0],
      [3, 0, 0]
  ], 
  'goal_state': {
    'condition': 'UA_without_intermeidate',
    'problem': [
      [2, 3, 1],
      [0, 0, 0],
      [0, 0, 0]
    ]
  }},
]

// set up test_problems
var base_start_state = [
    [1, 2, 3],
    [0, 0, 0],
    [0, 0, 0]
  ]
var base_goal_states = [
  {'condition': 'PA_with_intermediate',
  'problem': [
    [1, 0, 0],
    [2, 3, 0],
    [0, 0, 0]
  ]},
  {'condition': 'PA_with_intermediate',
  'problem': [
    [1, 3, 0],
    [2, 0, 0],
    [0, 0, 0]
  ]},
  {'condition': 'PA_without_intermediate',
  'problem': [
    [0, 0, 0],
    [3, 2, 0],
    [1, 0, 0]
  ]},
  {'condition': 'PA_without_intermediate',
  'problem': [
    [0, 0, 0],
    [3, 1, 0],
    [2, 0, 0]
  ]},
]

//permute start and goal states
var start_permutations = [[0,1,2],[1,0,2],[1,2,0]]
//second permutations used for flipping the non-tower peg
var goal_permutations = [[0,2,1],[2,0,1],[2,1,0]]

for (s=0; s<start_permutations.length; s++) {
  var start_permute = start_permutations[s]
  var goal_permute = goal_permutations[s]
  var start_state = []
  for (peg=0; peg<start_permutations.length; peg++){
    start_state.push(base_start_state[start_permute[peg]])
  }
  //permute goal states
  for (gs=0; gs<base_goal_states.length; gs++) {
    var goal_state = []
    for (peg=0; peg<start_permutations.length; peg++){
      goal_state.push(base_goal_states[gs]['problem'][start_permute[peg]])
    }
    test_problems.push(
      {'start_state': start_state, 
      'goal_state': {'problem': goal_state, 'condition': base_goal_states[gs]['condition']}}
      )
    // flip pegs that don't start with a tower
    var goal_state = []
    for (peg=0; peg<start_permutations.length; peg++){
      goal_state.push(base_goal_states[gs]['problem'][goal_permute[peg]])
    }
    test_problems.push(
      {'start_state': start_state, 
      'goal_state': {'problem': goal_state, 'condition': base_goal_states[gs]['condition']}}
      )
  }
}
var test_problems = jsPsych.randomization.shuffle(test_problems).concat(jsPsych.randomization.shuffle(test_problems))

// set up practice
var exp_stage = 'practice'
var problems = practice_problems
// setup blocks
var num_blocks = 3
var block_length = test_problems.length/num_blocks
var curr_placement = jQuery.extend(true, [], problems[problem_i].start_state)

/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
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
    exp_id: 'ward_and_allport'
  },
  timing_post_trial: 0
};

 var reminder_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text>Plan ahead first!<br></br><br></br>Work carefully but swiftly!</div></div>',
  is_html: true,
  choices: 'none',
  timing_response: 5000,
  response_ends_trial: true,
  data: {trial_id: "reminder"},
  timing_post_trial: 0
};

var reminder_node = {
    timeline: [reminder_block],
    conditional_function: function(){
        var data = jsPsych.data.getLastTrialData();
        if(problem_i%15 == 0 && problem_i > 0){
            return true;
        } else {
            return false;
        }
    }
}


var instructions_block = {
  type: 'poldrack-single-stim',
  stimulus: "<div class = centerbox><div class = center-text>Solve the Tower Tasks!<br>Plan ahead first and work swiftly!<br>We'll start with some practice.</div></div>",
  is_html: true,
  choices: 'none',
  timing_stim: 5000, 
  timing_response: 5000,
  data: {
    trial_id: "instructions",
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
  timing_post_trial: 500,
  on_finish: function() {
    exp_stage = 'test'
    problem_i = 0
    problems = test_problems
    reset_problem()
  }
};

var start_practice_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text>About to start practice</p></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 1500, 
  timing_response: 1500,
  data: {
    trial_id: "practice_start_block"
  },
  timing_post_trial: 500,
  on_finish: function() {
    reset_problem()
  }
};

var tohand_block = {
  type: 'poldrack-single-stim',
  stimulus: getStim,
  choices: get_hand_choices,
  is_html: true,
  data: {
    trial_id: "to_hand",
    exp_stage: exp_stage
  },
  timing_stim: -1,
  timing_response: -1,
  response_ends_trial: true,
  timing_post_trial: 0,
  on_finish: function(data) {
    pegClick(choices.indexOf(data.key_press))
    jsPsych.data.addDataToLastTrial({
      'current_position': jQuery.extend(true, [], curr_placement),
      'num_moves_made': num_moves,
      'min_moves': 3,
      'start_state': problems[problem_i].start_state,
      'goal_state': problems[problem_i].goal_state.problem,
      'condition': problems[problem_i].goal_state.condition,
      'problem_id': problem_i
    })
  }
}

var toboard_block = {
  type: 'poldrack-single-stim',
  stimulus: getStim,
  choices: get_board_choices,
  is_html: true,
  data: {
    trial_id: "to_board",
    exp_stage: exp_stage
  },
  timing_stim: -1,
  timing_response: -1,
  response_ends_trial: true,
  timing_post_trial: 0,
  on_finish: function(data) {
    pegClick(choices.indexOf(data.key_press))
    jsPsych.data.addDataToLastTrial({
      'current_position': jQuery.extend(true, [], curr_placement),
      'num_moves_made': num_moves,
      'min_moves': 3,
      'start_state': problems[problem_i].start_state,
      'goal_state': problems[problem_i].goal_state.problem,
      'condition': problems[problem_i].goal_state.condition,
      'problem_id': problem_i
    })
  }
}

var feedback_block = {
  type: 'poldrack-single-stim',
  stimulus: getFB,
  choices: 'none',
  is_html: true,
  data: {
    trial_id: 'feedback'
  },
  timing_stim: 1000,
  timing_response: get_ITI,
  timing_post_trial: 0,
  on_finish: function() {
    var time_elapsed = new Date() - problem_start_time
    jsPsych.data.addDataToLastTrial({
      'exp_stage': exp_stage,
      'problem_time': time_elapsed,
      'num_moves_made': num_moves,
      'min_moves': 3,
      'correct': correct
    })
    //advance round
    problem_i += 1
    reset_problem()
  },
}

var problem_node = {
  timeline: [tohand_block, toboard_block],
  loop_function: function(data) {
    data = data[1]
    var goal_state = data.goal_state
    var isequal = true
    for (var i = 0; i < goal_state.length; i++) {
      isequal = arraysEqual(goal_state[i], data.current_position[i])
      if (isequal === false) {
        break;
      }
    }
    return !isequal
  }
}

var task_node = {
  timeline: [problem_node, feedback_block, reminder_node],
  loop_function: function(data) {
    var time_elapsed = new Date() - start_time
    if (time_elapsed < task_limit && problem_i < problems.length) {
      return true
    } else {
      return false
    }
  },
  timing_post_trial: 1000
}


/* create experiment definition array */
var ward_and_allport_experiment = [];

ward_and_allport_experiment.push(instructions_block);
setup_fmri_intro(ward_and_allport_experiment, choices)
ward_and_allport_experiment.push(start_practice_block);
ward_and_allport_experiment.push(task_node)
ward_and_allport_experiment.push(start_test_block);
ward_and_allport_experiment.push(task_node)
ward_and_allport_experiment.push(end_block);