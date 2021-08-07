use rand::prelude::*;

pub trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

pub const DIJ: [(usize, usize); 4] = [(!0, 0), (0, !0), (0, 1), (1, 0)];
pub const DIR: [char; 4] = ['U', 'L', 'R', 'D'];

pub type Output = String;

pub struct Input {
    pub N: usize,
    pub s: (usize, usize),
    pub c: Vec<Vec<char>>,
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {} {}", self.N, self.s.0, self.s.1)?;
        for i in 0..self.N {
            writeln!(f, "{}", self.c[i].iter().collect::<String>())?;
        }
        Ok(())
    }
}

pub fn get_visited(
    input: &Input,
    out: &Output,
) -> (Vec<Vec<bool>>, i64, Vec<(usize, usize)>, String) {
    let mut visited = mat![false; input.N; input.N];
    let (mut pi, mut pj) = input.s;
    let mut length = 0;
    let mut ps = vec![(pi, pj)];
    let mut err = String::new();
    for c in out.chars() {
        if let Some(d) = DIR.iter().position(|&d| d == c) {
            pi += DIJ[d].0;
            pj += DIJ[d].1;
            if pi >= input.N || pj >= input.N || input.c[pi][pj] == '#' {
                err = "Visiting an obstacle".to_owned();
                break;
            }
        } else {
            err = format!("Illegal output: {}", c);
            break;
        }
        length += (input.c[pi][pj] as u8 - b'0') as i64;
        ps.push((pi, pj));
    }
    for &(pi, pj) in &ps {
        for d in 0..4 {
            for k in 0.. {
                let i = pi + DIJ[d].0 * k;
                let j = pj + DIJ[d].1 * k;
                if i < input.N && j < input.N && input.c[i][j] != '#' {
                    visited[i][j] = true;
                } else {
                    break;
                }
            }
        }
    }
    (visited, length, ps, err)
}

pub fn get_at_t(input: &Input, out: &Output, t: i64) -> Output {
    if t == i64::max_value() {
        return out.clone();
    }
    let (mut pi, mut pj) = input.s;
    let mut length = 0;
    let mut out2 = vec![];
    for c in out.chars() {
        out2.push(c);
        if let Some(d) = DIR.iter().position(|&d| d == c) {
            pi += DIJ[d].0;
            pj += DIJ[d].1;
            if pi >= input.N || pj >= input.N || input.c[pi][pj] == '#' {
                return out2.into_iter().collect::<String>();
            }
        } else {
            return out2.into_iter().collect::<String>();
        }
        length += (input.c[pi][pj] as u8 - b'0') as i64;
        if length > t {
            break;
        }
    }
    out2.into_iter().collect::<String>()
}

pub fn compute_score_detail(input: &Input, out: &Output) -> (i64, String) {
    let (visited, length, ps, err) = get_visited(input, out);
    if err.len() > 0 {
        return (0, err);
    }
    let mut num = 0;
    let mut den = 0;
    for i in 0..input.N {
        for j in 0..input.N {
            if input.c[i][j] != '#' {
                den += 1;
                if visited[i][j] {
                    num += 1;
                }
            }
        }
    }
    if *ps.last().unwrap() != input.s {
        return (0, "You have to go back to the starting point".to_owned());
    }
    let mut score = 1e4 * num as f64 / den as f64;
    if num == den {
        score += 1e7 * input.N as f64 / length as f64;
    }
    (score.round() as i64, String::new())
}

pub fn gen(seed: u64) -> Input {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed ^ 10);
    let N = (rng.gen_range(25, 36) * 2 - 1) as usize;
    let mut c = mat!['#'; N; N];
    let K = rng.gen_range(N as u64 * 2, N as u64 * 4 + 1);
    for _ in 0..K {
        let dir = rng.gen_range(0, 2);
        let i = (rng.gen_range(0, (N as i32 + 1) / 2) * 2) as usize;
        let j = rng.gen_range(0, N as i32);
        let h = rng.gen_range(3, 11);
        let w = (b'0' + rng.gen_range(5, 10) as u8) as char;
        for k in (j - h).max(0)..=(j + h).min(N as i32 - 1) {
            if dir == 0 {
                c[i][k as usize] = w;
            } else {
                c[k as usize][i] = w;
            }
        }
    }
    let mut visited = mat![false; N; N];
    let mut c_max = mat!['#'; N; N];
    let mut max_size = 0;
    for i in 0..N {
        for j in 0..N {
            if visited[i][j] || c[i][j] == '#' {
                continue;
            }
            let mut c2 = mat!['#'; N; N];
            let mut count = 0;
            let mut stack = vec![(i, j)];
            visited[i][j] = true;
            while let Some((pi, pj)) = stack.pop() {
                count += 1;
                c2[pi][pj] = c[pi][pj];
                for &(di, dj) in &DIJ {
                    let qi = pi + di;
                    let qj = pj + dj;
                    if qi < N && qj < N && !visited[qi][qj] && c[qi][qj] != '#' {
                        visited[qi][qj] = true;
                        stack.push((qi, qj));
                    }
                }
            }
            if max_size.setmax(count) {
                c_max = c2;
            }
        }
    }
    let mut s;
    loop {
        s = (
            rng.gen_range(0, N as i32) as usize,
            rng.gen_range(0, N as i32) as usize,
        );
        if c_max[s.0][s.1] != '#' {
            break;
        }
    }
    Input { N, s, c: c_max }
}
