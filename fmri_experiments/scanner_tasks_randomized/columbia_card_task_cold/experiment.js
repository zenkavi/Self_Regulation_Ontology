/* ************************************ */
/* Define helper functions */
/* ************************************ */
var appendTestData = function() {
	jsPsych.data.addDataToLastTrial({
		num_cards_chosen: currID,
		num_loss_cards: numLossCards,
		gain_amount: gainAmt,
		loss_amount: lossAmt,
		round_points: roundPointsArray.slice(-1),
		whichRound: whichRound
	})
}

var getButtons = function(buttonType) {
	var buttons = ""
	buttons = "<div class = allbuttons>"
	for (i = 0; i < 33; i++) {
		buttons += "<button type = 'button' class = 'CCT-btn chooseButton' id = " + i +
			" onclick = chooseButton(this.id)>" + i + "</button>"
	}
	return buttons
}

var getBoard = function(board_type) {
	var board = ''
	if (board_type == 2) {
		board = "<div class = cardbox>"
		for (i = 1; i < 33; i++) {
		board += "<div class = square><input type='image' class = card_image id = c" + i +
			" src='/static/experiments/columbia_card_task_cold/images/beforeChosen.png'></div>"
		}
		
	} else {
		board = "<div class = cardbox2>"
		for (i = 1; i < 33; i++) {
		board += "<div class = square><input class = card_image type='image' id = c" + i +
			" src='/static/experiments/columbia_card_task_cold/images/beforeChosen.png'></div>"
		}
	}
	board += "</div>"
	return board
}

var getText = function() {
	return '<div class = centerbox><p class = block-text>Overall, you earned ' + totalPoints + ' points. These are the points used for your bonus from three randomly picked trials:  ' +
		'<ul list-text><li>' + prize1 + '</li><li>' + prize2 + '</li><li>' + prize3 + '</li></ul>' +
		'</p><p class = block-text>Press <strong>enter</strong> to continue.</p></div>'
}

var turnOneCard = function(whichCard, win) {
	if (win === 'loss') {
		document.getElementById("c" + whichCard + "").src =
			'/static/experiments/columbia_card_task_cold/images/loss.png';
	} else {
		document.getElementById("c" + whichCard + "").src =
			'/static/experiments/columbia_card_task_cold/images/chosen.png';
	}
}

function doSetTimeout(card_i, delay, points, win) {
	CCT_timeouts.push(setTimeout(function() {
		turnOneCard(card_i, win);
		document.getElementById("current_round").innerHTML = 'Current Round Points: ' + points
	}, delay));
}

function clearTimers() {
	for (var i = 0; i < CCT_timeouts.length; i++) {
		clearTimeout(CCT_timeouts[i]);
	}
}


var appendPayoutData = function(){
	jsPsych.data.addDataToLastTrial({reward: [prize1, prize2, prize3]})
}

var chooseButton = function(clicked_id) {
	$('#nextButton').prop('disabled', false)
	$('.chooseButton').prop('disabled', true)
	currID = parseInt(clicked_id)
	var roundPoints = 0
	var cards_to_turn = jsPsych.randomization.repeat(cardArray, 1).slice(0, currID)
	for (var i = 0; i < cards_to_turn.length; i++) {
		var card_i = cards_to_turn[i]
		if (whichLossCards.indexOf(card_i) == -1) {
			roundPoints += gainAmt
		} else {
			roundPoints -= lossAmt
			break
		}
	}
	roundPointsArray.push(roundPoints)
	if ($('#feedback').length) {
		document.getElementById("feedback").innerHTML =
			'<strong>You chose ' + clicked_id +
			' card(s)</strong>. When you click on the "Next" button, the next round starts. Please note that the loss amount, the gain amount, and the number of loss cards might have changed.'
	}
}

var instructButton = function(clicked_id) {
	currID = parseInt(clicked_id)
	document.getElementById(clicked_id).src =
		'/static/experiments/columbia_card_task_cold/images/chosen.png';
}

// appends text to be presented in the game
function appendTextAfter(input, search_term, new_text) {
	var index = input.indexOf(search_term) + search_term.length
	return input.slice(0, index) + new_text + input.slice(index)
}



// this function sets up the round params (loss amount, gain amount, which ones are loss cards, initializes the array for cards to be clicked, )
var getRound = function() {
	var currID = 0
	cardArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
		24, 25, 26, 27, 28, 29, 30, 31, 32
	]
	shuffledCardArray = jsPsych.randomization.repeat(cardArray, 1)
	whichRound = whichRound + 1
	randomChosenCards = []
	roundParams = shuffledParamsArray.pop()
	numLossCards = roundParams[0]
	gainAmt = roundParams[1]
	lossAmt = roundParams[2]
	whichLossCards = []
	for (i = 0; i < numLossCards; i++) {
		whichLossCards.push(shuffledCardArray.pop())
	}
	gameState = gameSetup
	gameState = appendTextAfter(gameState, 'Game Round: ', whichRound)
	gameState = appendTextAfter(gameState, 'Loss Amount: ', lossAmt)
	gameState = appendTextAfter(gameState, 'Number of Loss Cards: ', numLossCards)
	gameState = appendTextAfter(gameState, 'Gain Amount: ', gainAmt)
	return gameState
}




/* ************************************ */
/* Define experimental variables */
/* ************************************ */
// task specific variables
var choices = [82]
var currID = 0
var numLossCards = 1
var gainAmt = ""
var lossAmt = ""
var points = []
var whichLossCards = [17]
var CCT_timeouts = []
var numRounds = 24
var whichRound = 0
var totalPoints = 0
var roundOver = 0
var roundPointsArray = []
var prize1 = 0
var prize2 = 0
var prize3 = 0


	
// this params array is organized such that the 0 index = the number of loss cards in round, the 1 index = the gain amount of each happy card, and the 2nd index = the loss amount when you turn over a sad face
var paramsArray = [
	[1, 10, 250],
	[1, 10, 750],
	[1, 30, 250],
	[1, 30, 750],
	[3, 10, 250],
	[3, 10, 750],
	[3, 30, 250],
	[3, 30, 750]
]

var cardArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	24, 25, 26, 27, 28, 29, 30, 31, 32
]

var shuffledCardArray = jsPsych.randomization.repeat(cardArray, 1)
var shuffledParamsArray = jsPsych.randomization.repeat(paramsArray, numRounds/8)


var gameSetup = 
	"<div class = practiceText><div class = block-text2 id = feedback></div></div>" +
	"<div class = cct-box2>"+
	"<div class = titleBigBox>   <div class = titleboxLeft><div class = game-text id = game_round>Game Round: </div></div>   <div class = titleboxLeft1><div class = game-text id = loss_amount>Loss Amount: </div></div>    <div class = titleboxMiddle1><div class = game-text id = gain_amount>Gain Amount: </div></div>    <div class = titlebox><div class = game-text>How many cards do you want to take? </div></div>     <div class = titleboxRight1><div class = game-text id = num_loss_cards>Number of Loss Cards: </div></div>" +
	"<div class = buttonbox><button type='button' id = nextButton class = 'CCT-btn select-button' onclick = clearTimers() disabled>Next Round</button></div>"+
	getButtons()+
	"</div>"+
	getBoard()



/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */
/* define static blocks */
var instructions_block = {
  type: 'poldrack-single-stim',
  stimulus: '<div class = centerbox><div class = center-text>Try to get as many points as possible</div></div>',
  is_html: true,
  choices: 'none',
  timing_stim: 9500, 
  timing_response: 9500,
  data: {
    trial_id: "instructions",
  },
  timing_post_trial: 500
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


var test_block = {
	type: 'single-stim-button',
	button_class: 'select-button',
	stimulus: getRound,
	data: {
		trial_id: 'stim',
		exp_stage: 'test'
	},
	timing_post_trial: 0,
	on_finish: appendTestData,
	response_ends_trial: true,
};

var payout_text = {
	type: 'poldrack-text',
	text: getText,
	data: {
		trial_id: 'reward'
	},
	cont_key: [13],
	timing_post_trial: 1000,
	on_finish: appendPayoutData,
};

var payoutTrial = {
	type: 'call-function',
	data: {
		trial_id: 'calculate reward'
	},
	func: function() {
		totalPoints = math.sum(roundPointsArray)
		randomRoundPointsArray = jsPsych.randomization.repeat(roundPointsArray, 1)
		prize1 = randomRoundPointsArray.pop()
		prize2 = randomRoundPointsArray.pop()
		prize3 = randomRoundPointsArray.pop()
		performance_var = prize1 + prize2 + prize3
	}
};



/* create experiment definition array */
var columbia_card_task_cold_experiment = [];
columbia_card_task_cold_experiment.push(instructions_block);
setup_fmri_intro(columbia_card_task_cold_experiment, choices)
columbia_card_task_cold_experiment.push(start_test_block);
for (b = 0; b < numRounds; b++) {
	columbia_card_task_cold_experiment.push(test_block);
}
columbia_card_task_cold_experiment.push(payoutTrial);
columbia_card_task_cold_experiment.push(payout_text);
columbia_card_task_cold_experiment.push(end_block);
