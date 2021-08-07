#![allow(dead_code)]
use rand::prelude::*;
use std::{cmp::Reverse, collections::BinaryHeap};

use text_scanner::scan;

#[derive(Clone, Copy)]
pub struct Pos {
    pub r: usize,
    pub c: usize,
}

impl Pos {
    fn new(r: usize, c: usize) -> Pos {
        Pos { r, c }
    }
}

pub struct Problem {
    n: usize,
    start_sq: usize,
    num_square: usize,
    num_road: usize,
    square_to_roads: Vec<Vec<usize>>,
    road_to_squares: Vec<Vec<usize>>,
    square_distances: Vec<Vec<usize>>,
    move_operations: Vec<Vec<String>>,
    is_initial_road: Vec<bool>,
    // adj_squares: Vec<Vec<usize>>,
    // pos_to_square: Vec<Vec<Option<usize>>>,
    // square_to_pos: Vec<Pos>,
    // pos_to_cost: Vec<Vec<Option<usize>>>,
}

pub fn read_problem() -> Problem {
    let n: usize = scan();
    let sr: usize = scan();
    let sc: usize = scan();
    let grid: Vec<Vec<char>> = (0..n).map(|_| scan::<String>().chars().collect()).collect();
    let mut pos_to_square: Vec<Vec<Option<usize>>> = vec![vec![None; n]; n];
    let mut pos_to_cost: Vec<Vec<Option<usize>>> = vec![vec![None; n]; n];
    let mut square_to_pos = Vec::new();
    let mut num_square = 0;
    for r in 0..n {
        for c in 0..n {
            if grid[r][c] != '#' {
                pos_to_square[r][c] = Some(num_square);
                pos_to_cost[r][c] = Some((grid[r][c] as u8 - b'0') as usize);
                square_to_pos.push(Pos::new(r, c));
                num_square += 1;
            }
        }
    }
    let mut num_road = 0;
    let mut road_to_squares = Vec::new();
    let mut square_to_roads = vec![Vec::new(); num_square];
    let mut used = vec![vec![false; n]; n];
    for r in 0..n {
        for c in 0..n {
            if !used[r][c] && pos_to_square[r][c].is_some() {
                let mut squares = Vec::new();
                let mut i = c;
                while i < n && pos_to_square[r][i].is_some() {
                    let sid = pos_to_square[r][i].unwrap();
                    used[r][i] = true;
                    squares.push(sid);
                    i += 1;
                }
                if squares.len() > 1 {
                    for &sid in &squares {
                        square_to_roads[sid].push(num_road);
                    }
                    road_to_squares.push(squares);
                    num_road += 1;
                }
            }
        }
    }
    let mut used = vec![vec![false; n]; n];
    for r in 0..n {
        for c in 0..n {
            if !used[r][c] && pos_to_square[r][c].is_some() {
                let mut squares = Vec::new();
                let mut i = r;
                while i < n && pos_to_square[i][c].is_some() {
                    let sid = pos_to_square[i][c].unwrap();
                    used[i][c] = true;
                    squares.push(sid);
                    i += 1;
                }
                if squares.len() > 1 {
                    for &sid in &squares {
                        square_to_roads[sid].push(num_road);
                    }
                    road_to_squares.push(squares);
                    num_road += 1;
                }
            }
        }
    }

    let mut adj_squares = Vec::new();
    for from_sq in 0..num_square {
        let pos = square_to_pos[from_sq];
        let mut new_pos = Vec::new();
        if pos.r >= 1 {
            new_pos.push((Pos::new(pos.r - 1, pos.c), 'U'));
        }
        if pos.r + 1 < n {
            new_pos.push((Pos::new(pos.r + 1, pos.c), 'D'));
        }
        if pos.c >= 1 {
            new_pos.push((Pos::new(pos.r, pos.c - 1), 'L'));
        }
        if pos.c + 1 < n {
            new_pos.push((Pos::new(pos.r, pos.c + 1), 'R'));
        }
        let mut this_adj = Vec::new();
        for (np, m) in new_pos {
            if let Some(new_sq) = pos_to_square[np.r][np.c] {
                this_adj.push((new_sq, m));
            }
        }
        adj_squares.push(this_adj);
    }

    #[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
    struct State {
        cost: usize,
        square: usize,
    }

    let mut square_distances = Vec::new();
    let mut move_operations = Vec::new();
    for start_sq in 0..num_square {
        let mut dist = vec![usize::max_value(); num_square];
        let mut moves = vec![String::new(); num_square];
        let mut queue = BinaryHeap::new();
        dist[start_sq] = 0;
        moves[start_sq] = String::new();
        queue.push(Reverse(State {
            cost: 0,
            square: start_sq,
        }));
        while let Some(Reverse(State { cost, square })) = queue.pop() {
            if cost > dist[square] {
                continue;
            }
            for &(new_square, m) in &adj_squares[square] {
                let pos = square_to_pos[new_square];
                let new_cost = cost + pos_to_cost[pos.r][pos.c].unwrap();
                if dist[new_square] > new_cost {
                    dist[new_square] = new_cost;
                    moves[new_square] = format!("{}{}", moves[square], m);
                    queue.push(Reverse(State {
                        cost: new_cost,
                        square: new_square,
                    }));
                }
            }
        }
        square_distances.push(dist);
        move_operations.push(moves);
    }

    let start_sq = pos_to_square[sr][sc].unwrap();
    let is_initial_road = (0..num_road)
        .map(|r| square_to_roads[start_sq].contains(&r))
        .collect();

    Problem {
        n,
        start_sq,
        num_road,
        num_square,
        road_to_squares,
        square_to_roads,
        square_distances,
        is_initial_road,
        move_operations,
        // square_to_pos,
        // pos_to_square,
        // pos_to_cost,
        // adj_squares,
    }
}

#[derive(Clone)]
struct OptimizeState<'a> {
    problem: &'a Problem,
    road_to_square: Vec<Option<usize>>,
    road_visit_order: Vec<usize>,
}

struct StateChange {
    old_total_time: usize,
    new_total_time: usize,
}

impl OptimizeState<'_> {
    fn square_visits<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        std::iter::once(self.problem.start_sq)
            .chain(
                self.road_visit_order
                    .iter()
                    .map(move |&r| self.road_to_square[r].unwrap()),
            )
            .chain(std::iter::once(self.problem.start_sq))
    }

    fn moves(&self) -> String {
        let mut moves = String::new();
        for (from, to) in self.square_visits().zip(self.square_visits().skip(1)) {
            moves.push_str(&self.problem.move_operations[from][to]);
        }
        moves
    }

    fn total_time(&self) -> usize {
        let mut total_time = 0;
        for (from, to) in self.square_visits().zip(self.square_visits().skip(1)) {
            total_time += self.problem.square_distances[from][to];
        }
        total_time
    }

    fn create_random<'a, R: Rng>(p: &'a Problem, rng: &mut R) -> OptimizeState<'a> {
        let mut road_visit_order: Vec<usize> =
            (0..p.num_road).filter(|&r| !p.is_initial_road[r]).collect();
        road_visit_order.shuffle(rng);

        let mut road_to_square = vec![None; p.num_road];
        for &r in &road_visit_order {
            road_to_square[r] = Some(*p.road_to_squares[r].choose(rng).unwrap());
        }

        OptimizeState {
            problem: p,
            road_to_square,
            road_visit_order,
        }
    }

    fn try_choose_square<F, R>(&mut self, accept: F, rng: &mut R) -> Option<StateChange>
    where
        F: Fn(i64) -> bool,
        R: Rng,
    {
        let r: usize = rng.gen_range(0, self.road_to_square.len());
        if self.road_to_square[r].is_none() {
            return None;
        }
        let old_total_time = self.total_time();
        let old_square = self.road_to_square[r];
        self.road_to_square[r] = Some(*self.problem.road_to_squares[r].choose(rng).unwrap());
        let new_total_time = self.total_time();
        let diff = new_total_time as i64 - old_total_time as i64;
        if accept(diff) {
            Some(StateChange {
                old_total_time,
                new_total_time,
            })
        } else {
            self.road_to_square[r] = old_square;
            None
        }
    }

    fn try_2_opt<F, R>(&mut self, accept: F, rng: &mut R) -> Option<StateChange>
    where
        F: Fn(i64) -> bool,
        R: Rng,
    {
        let old_total_time = self.total_time();
        let visits_num = self.road_visit_order.len();
        let i: usize = rng.gen_range(0, visits_num - 1);
        let j: usize = rng.gen_range(i + 1, visits_num);
        self.road_visit_order[i..j + 1].reverse();
        let new_total_time = self.total_time();
        let diff = new_total_time as i64 - old_total_time as i64;
        if accept(diff) {
            Some(StateChange {
                old_total_time,
                new_total_time,
            })
        } else {
            self.road_visit_order[i..j + 1].reverse();
            None
        }
    }
}

fn climbing() -> impl Fn(i64) -> bool {
    |diff| diff < 0
}

fn kick(iter_count: usize, count_limit: usize) -> impl Fn(i64) -> bool {
    move |diff| diff < 0 || iter_count >= count_limit
}

fn annealing<'a, R: Rng>(
    time_ratio: f64,
    start_temp: f64,
    end_temp: f64,
    rng: &mut R,
) -> impl Fn(i64) -> bool {
    let value = rng.gen::<f64>();
    move |diff| {
        let temp = start_temp + (end_temp - start_temp) * time_ratio;
        value < ((-diff as f64) / temp).exp()
    }
}

fn main() {
    let start = std::time::Instant::now();
    let time_limit = std::time::Duration::from_millis(2600);

    let kick_count = 5000;

    let problem = read_problem();

    let mut rng = SmallRng::seed_from_u64(58);
    let mut state = OptimizeState::create_random(&problem, &mut rng);

    let mut best_state = state.clone();
    let mut best_total_time = best_state.total_time();

    let mut iter_count = 0;

    while start.elapsed() <= time_limit {
        if let Some(changed) = state.try_2_opt(kick(iter_count, kick_count), &mut rng) {
            if changed.new_total_time < best_total_time {
                best_state = state.clone();
                best_total_time = changed.new_total_time;
                // eprintln!(
                //     "{:04}ms: {} -> {}",
                //     start.elapsed().as_millis(),
                //     changed.old_total_time,
                //     changed.new_total_time
                // );
            }
            iter_count = 0;
        }
        if let Some(changed) = state.try_choose_square(climbing(), &mut rng) {
            if changed.new_total_time < best_total_time {
                best_state = state.clone();
                best_total_time = changed.new_total_time;
                // eprintln!(
                //     "{:04}ms: {} -> {}",
                //     start.elapsed().as_millis(),
                //     changed.old_total_time,
                //     changed.new_total_time
                // );
            }
            iter_count = 0;
        }
        iter_count += 1;
    }

    eprintln!("total_time: {}", best_total_time);
    assert!(best_total_time == best_state.total_time());
    println!("{}", best_state.moves());
}

mod text_scanner;
