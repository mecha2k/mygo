var BOARD_SIZE = 19
var jrecord = new JGO.Record(BOARD_SIZE, BOARD_SIZE)
var jboard = jrecord.jboard
var jsetup = new JGO.Setup(jboard, JGO.BOARD.largeWalnut)
var player = JGO.BLACK // next player
var ko = false
var lastMove = false // ko coordinate and last move coordinate
var lastHover = false
var lastX = -1
var lastY = -1 // hover helper vars
var record = []
var colnames = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T"
]
var waitingForBot = false

function resetGame(ev) {
  jrecord.jboard.clear()
  jrecord.root = jrecord.current = null
  jrecord.info = {}
  record = []
  waitingForBot = false
  ev.preventDefault()
}

function coordsToString(point) {
  var row = BOARD_SIZE - 1 - point.j
  var col = point.i
  return colnames[col] + (row + 1).toString()
}

function stringToCoords(move_string) {
  var colStr = move_string.substring(0, 1)
  var rowStr = move_string.substring(1)
  var col = colnames.indexOf(colStr)
  var row = BOARD_SIZE - parseInt(rowStr, 10)
  return new JGO.Coordinate(col, row)
}

function applyMove(player, coord) {
  var play = jboard.playMove(coord, player, ko)

  if (play.success) {
    record.push(coordsToString(coord))
    var node = jrecord.createNode(true)
    node.info.captures[player] += play.captures.length // tally captures
    node.setType(coord, player) // play stone
    node.setType(play.captures, JGO.CLEAR) // clear opponent's stones

    if (lastMove) {
      node.setMark(lastMove, JGO.MARK.NONE) // clear previous mark
    }
    if (ko) {
      node.setMark(ko, JGO.MARK.NONE) // clear previous ko mark
    }
    node.setMark(coord, JGO.MARK.CIRCLE) // mark move
    lastMove = coord

    if (play.ko) node.setMark(play.ko, JGO.MARK.CIRCLE) // mark ko, too
    ko = play.ko
  } else alert("Illegal move: " + play.errorMsg)
}

function waitForBot() {
  console.log("Waiting for bot...")
  document.getElementById("status").style.display = "none"
  document.getElementById("spinner").style.display = "block"
  waitingForBot = true
}

function stopWaiting(botmove) {
  var text = "Bot plays " + botmove
  if (botmove === "pass") {
    text = "Bot passes"
  } else if (botmove === "resign") {
    text = "Bot resigns"
  }
  document.getElementById("status").innerHTML = text
  document.getElementById("status").style.display = "block"
  document.getElementById("spinner").style.display = "none"
  waitingForBot = false
}

jsetup.setOptions({ stars: { points: 5 } })

jsetup.create("board", function (canvas) {
  canvas.addListener("click", function (coord, ev) {
    if (waitingForBot) {
      return
    }
    var opponent = player === JGO.BLACK ? JGO.WHITE : JGO.BLACK

    if (ev.shiftKey) {
      // on shift do edit
      if (jboard.getMark(coord) === JGO.MARK.NONE) jboard.setMark(coord, JGO.MARK.SELECTED)
      else jboard.setMark(coord, JGO.MARK.NONE)

      return
    }

    // clear hover away - it'll be replaced or then it will be an illegal move
    // in any case so no need to worry about putting it back afterwards
    if (lastHover) jboard.setType(new JGO.Coordinate(lastX, lastY), JGO.CLEAR)

    lastHover = false

    console.log("Human", coordsToString(coord))
    applyMove(JGO.BLACK, coord)

    waitForBot()
    fetch("/select-move/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ board_size: BOARD_SIZE, moves: record })
    })
      .then(function (response) {
        if (!waitingForBot) {
          console.log("Got response but not waiting for one")
          return
        }
        response.json().then(function (data) {
          if (data["bot_move"] === "pass" || data["bot_move"] === "resign") {
            record.push(data["bot_move"])
          } else {
            var botCoord = stringToCoords(data["bot_move"])
            applyMove(JGO.WHITE, botCoord)
          }
          stopWaiting(data["bot_move"])
        })
      })
      .catch(function (error) {
        console.log(error)
        stopWaiting(data["bot_move"])
      })
  })

  canvas.addListener("mousemove", function (coord) {
    if (coord.i === -1 || coord.j === -1 || (coord.i === lastX && coord.j === lastY)) return

    if (lastHover)
      // clear previous hover if there was one
      jboard.setType(new JGO.Coordinate(lastX, lastY), JGO.CLEAR)

    lastX = coord.i
    lastY = coord.j

    if (jboard.getType(coord) === JGO.CLEAR && jboard.getMark(coord) === JGO.MARK.NONE) {
      jboard.setType(coord, player === JGO.WHITE ? JGO.DIM_WHITE : JGO.DIM_BLACK)
      lastHover = true
    } else lastHover = false
  })

  canvas.addListener("mouseout", function () {
    if (lastHover) jboard.setType(new JGO.Coordinate(lastX, lastY), JGO.CLEAR)

    lastHover = false
  })
})
