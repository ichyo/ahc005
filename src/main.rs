#![allow(dead_code)]
use judge::compute_score_detail;
use rand::prelude::*;
use std::{cmp::Reverse, collections::BinaryHeap, time::Duration};

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
impl Problem {
    pub fn from_params(n: usize, sr: usize, sc: usize, grid: Vec<Vec<char>>) -> Problem {
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
    pub fn from_judge(input: &judge::Input) -> Problem {
        Problem::from_params(input.N, input.s.0, input.s.1, input.c.clone())
    }
    pub fn from_stdin() -> Problem {
        let n: usize = scan();
        let sr: usize = scan();
        let sc: usize = scan();
        let grid: Vec<Vec<char>> = (0..n).map(|_| scan::<String>().chars().collect()).collect();
        Problem::from_params(n, sr, sc, grid)
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

    fn try_choose_square<F, R>(
        &mut self,
        accept: F,
        rng: &mut R,
        old_total_time: usize,
    ) -> Option<StateChange>
    where
        F: Fn(i64) -> bool,
        R: Rng,
    {
        let visits_num = self.road_visit_order.len();
        let index: usize = rng.gen_range(0, visits_num);

        let r = self.road_visit_order[index];
        let old_square = self.road_to_square[r].unwrap();
        let new_square = *self.problem.road_to_squares[r].choose(rng).unwrap();

        if old_square == new_square {
            return None;
        }

        self.road_to_square[r] = Some(new_square);

        let x = if index == 0 {
            self.problem.start_sq
        } else {
            self.road_to_square[self.road_visit_order[index - 1]].unwrap()
        };
        let y = if index + 1 == visits_num {
            self.problem.start_sq
        } else {
            self.road_to_square[self.road_visit_order[index + 1]].unwrap()
        };

        let mut new_total_time = old_total_time;
        new_total_time += self.problem.square_distances[x][new_square];
        new_total_time += self.problem.square_distances[new_square][y];
        new_total_time -= self.problem.square_distances[x][old_square];
        new_total_time -= self.problem.square_distances[old_square][y];

        let diff = new_total_time as i64 - old_total_time as i64;
        if accept(diff) {
            Some(StateChange {
                old_total_time,
                new_total_time,
            })
        } else {
            self.road_to_square[r] = Some(old_square);
            None
        }
    }

    fn try_2_opt<F, R>(
        &mut self,
        accept: F,
        rng: &mut R,
        old_total_time: usize,
    ) -> Option<StateChange>
    where
        F: Fn(i64) -> bool,
        R: Rng,
    {
        let visits_num = self.road_visit_order.len();
        let i: usize = rng.gen_range(0, visits_num - 1);
        let j: usize = rng.gen_range(i + 1, visits_num);
        let mut new_total_time = old_total_time;
        let x = if i == 0 {
            self.problem.start_sq
        } else {
            self.road_to_square[self.road_visit_order[i - 1]].unwrap()
        };
        let y = if j + 1 == visits_num {
            self.problem.start_sq
        } else {
            self.road_to_square[self.road_visit_order[j + 1]].unwrap()
        };
        let p = self.road_to_square[self.road_visit_order[i]].unwrap();
        let q = self.road_to_square[self.road_visit_order[j]].unwrap();
        // x -> p, q -> y
        // x -> q, p -> x
        new_total_time += self.problem.square_distances[x][q];
        new_total_time += self.problem.square_distances[y][p];
        new_total_time -= self.problem.square_distances[x][p];
        new_total_time -= self.problem.square_distances[y][q];

        let diff = new_total_time as i64 - old_total_time as i64;
        if accept(diff) {
            self.road_visit_order[i..j + 1].reverse();
            Some(StateChange {
                old_total_time,
                new_total_time,
            })
        } else {
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

fn solve<'a>(problem: &'a Problem, time_limit: Duration) -> OptimizeState<'a> {
    let start = std::time::Instant::now();

    let kick_count = 5000;

    let mut rng = SmallRng::seed_from_u64(58);
    let mut state = OptimizeState::create_random(&problem, &mut rng);
    let mut old_total_time = state.total_time();

    let mut best_state = state.clone();
    let mut best_total_time = best_state.total_time();

    let mut iter_count = 0;

    loop {
        let time_ratio = start.elapsed().as_secs_f64() / time_limit.as_secs_f64();
        if time_ratio >= 1.0 {
            break;
        }
        // if iter_count >= kick_count {
        //     for _ in 0..3 {
        //         state.try_2_opt(|_| true, &mut rng);
        //     }
        //     iter_count = 0;
        // }
        let start_temp = 10.0;
        let end_temp = 1.0;
        if let Some(changed) = state.try_2_opt(
            annealing(time_ratio, start_temp, end_temp, &mut rng),
            &mut rng,
            old_total_time,
        ) {
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
            old_total_time = changed.new_total_time;
        }

        if let Some(changed) = state.try_choose_square(
            annealing(time_ratio, start_temp, end_temp, &mut rng),
            &mut rng,
            old_total_time,
        ) {
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
            old_total_time = changed.new_total_time;
        }
        iter_count += 1;
    }
    best_state
}

fn local_test() {
    let mut scores = Vec::new();
    let mut sum = 0;
    for seed in 0..100 {
        let start = std::time::Instant::now();
        let time_limit = std::time::Duration::from_millis(1300);
        let input = judge::gen(seed);
        let problem = Problem::from_judge(&input);
        let best_state = solve(&problem, time_limit - start.elapsed());
        let score = compute_score_detail(&input, &best_state.moves());
        assert!(score.1.is_empty());
        sum += score.0;
        scores.push(score.0);

        eprintln!(
            "seed {:3}: {} (avg: {})",
            seed,
            score.0,
            sum / scores.len() as i64
        );
    }
    let sum = scores.iter().sum::<i64>();
    println!("{}", sum);
}

fn atcoder() {
    let start = std::time::Instant::now();
    let problem = Problem::from_stdin();
    let time_limit = std::time::Duration::from_millis(2600);
    let best_state = solve(&problem, time_limit - start.elapsed());

    eprintln!("total_time: {}", best_state.total_time());
    println!("{}", best_state.moves());
}

fn main() {
    if std::env::var("LOCAL_TEST").is_ok() {
        local_test();
    } else {
        atcoder();
    }
}

mod judge;
mod text_scanner;
