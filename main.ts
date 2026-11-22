// Sokoban solver entry for make.fun sokoban-solver-01 (Node 22, node /bot/main.ts).
// Levels are independent rooms off a hallway: split into rooms, solve each push-optimally
// with an A* over pushes (minimum-cost box-goal matching over exact one-box push distances
// as the estimate; dead squares and frozen 2x2 blocks prune). Budgets are state counts,
// never the clock, and ties break by a fixed order, so executions are byte-identical.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];
const ROOM_STATES = 2_000_000;
const LEVEL_STATES = 4_000_000;

function parse(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box alone can be pushed onto some goal (reverse pull over open floor). */
function liveSquares(w, wall, goals) {
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Exact pushes a lone box needs from every square to goal g (reverse pull BFS); -1 is never. */
function pushDistTo(w, wall, g) {
  const dist = new Int32Array(wall.length).fill(-1);
  dist[g] = 0;
  const queue = [g];
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && dist[y] < 0) {
        dist[y] = dist[x] + 1;
        queue.push(y);
      }
    }
  }
  return dist;
}

/** The player's reachable region around boxes: its lowest square, marking `seen` with `stamp`. */
function reach(w, wall, occupied, from, seen, stamp, stack) {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const q1 = p - w, q2 = p + w, q3 = p - 1, q4 = p + 1;
    if (!wall[q1] && !occupied[q1] && seen[q1] !== stamp) { seen[q1] = stamp; stack[n++] = q1; }
    if (!wall[q2] && !occupied[q2] && seen[q2] !== stamp) { seen[q2] = stamp; stack[n++] = q2; }
    if (!wall[q3] && !occupied[q3] && seen[q3] !== stamp) { seen[q3] = stamp; stack[n++] = q3; }
    if (!wall[q4] && !occupied[q4] && seen[q4] !== stamp) { seen[q4] = stamp; stack[n++] = q4; }
  }
  return low;
}

/** Whether the box just pushed to t froze in a 2x2 block of walls and boxes with a box off goal. */
function frozenBlock(w, wall, occupied, goal, t) {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1, c = a + w, d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** Shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(w, wall, occupied, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/**
 * Push-optimal solution of one (sub)level: {pushes, moves} or null past maxStates.
 * A* over pushes; a state is the sorted boxes plus the player's reachable region.
 */
function solvePushOptimal(w, wall, goals, startBoxes, startPlayer, maxStates) {
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const dists = goals.map((g) => pushDistTo(w, wall, g));
  const n = startBoxes.length;
  // Minimum-cost matching of boxes to goals over the one-box push distances (admissible).
  const estimate = (boxes) => {
    if (n === 1) return dists[0][boxes[0]] < 0 ? Infinity : dists[0][boxes[0]];
    if (n > 6) {
      // Too many boxes to match exactly (only the whole-level fallback sees this):
      // the sum of each box's own nearest-goal distance still never overestimates.
      let sum = 0;
      for (const b of boxes) {
        let m = Infinity;
        for (const dt of dists) if (dt[b] >= 0 && dt[b] < m) m = dt[b];
        if (m === Infinity) return Infinity;
        sum += m;
      }
      return sum;
    }
    let best = Infinity;
    const perm = goals.map((_, i) => i);
    const walk = (k) => {
      if (k === n) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const d = dists[perm[i]][boxes[i]];
          if (d < 0) return;
          sum += d;
        }
        if (sum < best) best = sum;
        return;
      }
      for (let i = k; i < n; i++) {
        [perm[k], perm[i]] = [perm[i], perm[k]];
        walk(k + 1);
        [perm[k], perm[i]] = [perm[i], perm[k]];
      }
    };
    walk(0);
    return best;
  };
  if (estimate(startBoxes) === Infinity) return null;
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const keyOf = (boxes, low) => low + '|' + boxes.join(',');
  const statesBoxes = [startBoxes];
  const statesPlayer = [startPlayer];
  const parent = [-1];
  const gScore = [0];
  const best = new Map();
  for (const b of startBoxes) occupied[b] = 1;
  best.set(keyOf(startBoxes, reach(w, wall, occupied, startPlayer, seen, ++stamp, stack)), 0);
  for (const b of startBoxes) occupied[b] = 0;
  // Binary heap on (f, then deeper g, then insertion order): fully deterministic.
  const heap = [];
  let seq = 0;
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hpush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const hpop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  hpush([estimate(startBoxes), 0, seq++, 0]);
  let found = startBoxes.every((b) => goal[b]) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const [, g, , s] = hpop();
    if (g !== gScore[s]) continue;
    const boxes = statesBoxes[s];
    for (const b of boxes) occupied[b] = 1;
    reach(w, wall, occupied, statesPlayer[s], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t]) continue;
        if (dists.every((dt) => dt[t] < 0)) continue; // dead square: a lone box there reaches no goal
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let key = null;
        let next = null;
        let est = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          est = estimate(next);
          if (est !== Infinity) key = keyOf(next, reach(w, wall, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (key === null) continue;
        const ng = g + 1;
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= ng) continue;
        if (statesBoxes.length >= maxStates) return null;
        best.set(key, ng);
        const id = statesBoxes.length;
        statesBoxes.push(next);
        statesPlayer.push(b);
        parent.push(s);
        gScore.push(ng);
        if (next.every((x) => goal[x])) { found = id; break; }
        hpush([ng + est, ng, seq++, id]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;
  const path = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s];
    const from = boxes.find((b) => !nextBoxes.includes(b));
    const to = nextBoxes.find((b) => !boxes.includes(b));
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d);
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** The level's rooms, cut along the dead squares of the player's row; null if not a set of rooms. */
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

function solve(text) {
  const level = parse(text);
  const { w, wall, goals, boxes, player } = level;
  if (boxes.every((b) => goals.includes(b))) return '';
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solvePushOptimal(w, wall, goals.slice().sort((a, b) => a - b), boxes, player, LEVEL_STATES);
    return whole === null ? '' : whole.moves;
  }
  let at = player;
  let moves = '';
  for (const room of rooms) {
    const subWall = wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) subWall[b] = 1;
    }
    const solved = solvePushOptimal(
      w, subWall,
      room.goals.slice().sort((a, b) => a - b),
      room.boxes.slice().sort((a, b) => a - b),
      at, ROOM_STATES,
    );
    if (solved === null) return '';
    moves += solved.moves;
    at = endOf(w, at, solved.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
