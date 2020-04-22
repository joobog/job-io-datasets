mod phase_combis_2;


use phase_combis_2::PhaseNum;
use phase_combis_2::PhaseLength;

pub type CodingType = u16;
pub type SimType = f32;

fn fmax<T: PartialOrd>(a: T, b: T) -> std::cmp::Ordering {
    if a < b {
        std::cmp::Ordering::Less
    }
    else if a > b {
        std::cmp::Ordering::Greater
    }
    else {
        std::cmp::Ordering::Equal
    }
}

/// Sum up quotients (low value / high value)
pub fn sum_quotients(coding_1:&[CodingType], coding_2:&[CodingType]) -> SimType {
    assert!(coding_1.len() == coding_2.len());
    let l = coding_1.len();
    let mut quotients: Vec<SimType> = vec![0.0; l];

    for i in 0..l {
        quotients[i] = if coding_1[i] < coding_2[i] {
            coding_1[i] as SimType / coding_2[i] as SimType
        }
        else {
            coding_2[i] as SimType / coding_1[i] as SimType
        };
    }
    let sum: SimType = quotients.iter().sum();
    sum
}


/// Get max quotient and max length of two non-equal sized phases
/// given 222282 2293
/// Test all possibilites 
/// 229300 222282
/// 022930 222282
/// 002293 222282
/// return (similarity, length)
fn find_max_phase_similarity(coding_1: &Vec<CodingType>, coding_2: &Vec<CodingType>) -> (SimType, PhaseLength) {
    let c1: &Vec<CodingType>;    
    let c2: &Vec<CodingType>;
    if coding_1.len() > coding_2.len() {
        c1 = coding_2;
        c2 = coding_1;
    }
    else {
        c1 = coding_1;
        c2 = coding_2;
    }

    let l1 = c1.len();
    let l2 = c2.len();
    let max_len = l2;

    let mut sims: Vec<SimType> = Vec::new();
    for shift in 0..l2-l1+1 {
        sims.push(sum_quotients(&c1[..], &c2[shift..(shift+l1)]));
    }

    let max_value = sims.iter().max_by(|x, y| fmax(**x, **y));

    match max_value {
       Some(&m) => (m / (max_len as SimType), max_len as PhaseNum),
       None => (0.0, 0),
    }
}


/// (sim1*len1 + sim2*len2 + sim3*len4) / (len1 + len2 + len3)
fn weighted_mean(data: &Vec<(SimType, PhaseLength)>) -> SimType {
    let mut len_sum: PhaseLength = 0;
    let mut weighted_sim_sum: SimType = 0.0;
    for (sim, len) in data {
        len_sum = len_sum + *len;
        weighted_sim_sum = weighted_sim_sum + sim * (*len as SimType);
    }
    weighted_sim_sum / (len_sum as SimType)
}


/// Helper function for find max value in a f32 vector
fn max_f32(data: &Vec<SimType>) -> SimType {
    let mut max: SimType = 0.0;
    for d in data {
        if *d > max {
            max = *d;
        }
    }
    max
}

pub fn job_similarity(job_coding_1: &Vec<Vec<CodingType>>, job_coding_2: &Vec<Vec<CodingType>>) -> SimType {
    let c1: &Vec<Vec<CodingType>>;    
    let c2: &Vec<Vec<CodingType>>;
    if job_coding_1.len() > job_coding_2.len() {
        c1 = job_coding_2;
        c2 = job_coding_1;
    }
    else {
        c1 = job_coding_1;
        c2 = job_coding_2;
    }

    let l1 = c1.len();
    let l2 = c2.len();

    let combis = phase_combis_2::PhaseCombinations::new(l1 as PhaseNum, l2 as PhaseNum);
    let mut job_sims: Vec<SimType> = Vec::new();

    // go through all combinations
    for combi in combis {
       let mut sims: Vec<(SimType, PhaseLength)> = Vec::new();
       for (idx_c1, idx_c2) in combi.to_indices_attached().iter().enumerate() {
           let (sim, phase_len) = find_max_phase_similarity(&c1[idx_c1], &c2[*idx_c2 as usize]);
           sims.push((sim, phase_len));
       }
       for idx in combi.to_indices_not_attached().iter() {
           let phase_len = c2[*idx as usize].len();
           let sim = 0.0;
           sims.push((sim, phase_len as PhaseNum));
       }
       job_sims.push(weighted_mean(&sims));
    }
    let max_sim = max_f32(&job_sims);
    max_sim
}



pub fn detect_phases(data: &Vec<CodingType>) -> Vec<Vec<CodingType>> {
	let data_length = data.len();
	let mut phases: Vec<(usize, usize)> = Vec::new();
	let mut idx = 0;
	while idx < data_length {
		while (idx < data_length) && (data[idx] == 0) {
			idx = idx + 1;
		}
		let start = idx;
		while (idx < data_length) && (data[idx] != 0) {
			idx = idx + 1;
		}
		let stop = idx;
		phases.push((start, stop));
	}
	if phases[phases.len()-1].0 == data_length {
		phases.pop();
	}
    let mut ps: Vec<Vec<CodingType>> = Vec::new();
    for phase in phases {
        let mut p:Vec<CodingType> = Vec::new();
        for i in phase.0..(phase.1) {
            p.push(data[i]);
        }
        ps.push(p);
    }
    ps
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use super::*;

    //#[test]
    //fn test_phase_combinations() {
    //    let res = get_phase_combinations(3, 5);
    //    println!("{:?}", res);
    //    assert_eq!(res[0], Combination{used_idxs: vec![0, 1], unused_idxs: vec![2]});
    //    assert_eq!(res[1], Combination{used_idxs: vec![1, 2], unused_idxs: vec![0]});
    //    assert_eq!(res[2], Combination{used_idxs: vec![0, 2], unused_idxs: vec![1]});
    //}

    #[test]
    fn test_print() {
        let coding_1 = vec![1, 1, 1, 1];
        let coding_2 = vec![2, 1, 1, 1];
        println!("{:?} {:?} {}", coding_1, coding_2, sum_quotients(&coding_1, &coding_2));

        let job_coding_1: Vec<Vec<CodingType>> = vec![vec![2, 2, 9, 3], vec![9, 1, 1]];
        let job_coding_2: Vec<Vec<CodingType>> = vec![vec![2, 2, 2, 2, 8, 2], vec![1], vec![8, 1, 1]];
        println!("job similarity: {}", job_similarity(&job_coding_1, &job_coding_2));
        assert!(true);
    }


    #[test]
    fn test_sum_quotients() {
        // 0.5 + 1 + 1 + 1 = 3.5
        let coding_1 = vec![1, 1, 1, 1];
        let coding_2 = vec![2, 1, 1, 1];
        assert_eq!(sum_quotients(&coding_1, &coding_2), 3.5);
    }

    #[test]
    fn test_sliding_similarity() {
        //let coding_1 = vec![2, 2, 9, 3];
        //let coding_2 = vec![2, 2, 2, 2, 8, 2];

        let coding_1 = vec![2, 2, 9, 3];
        let coding_2 = vec![2, 2, 2, 2];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 2.889, 0.001);
        let coding_1 = vec![2, 2, 9, 3];
        let coding_2 = vec![2, 2, 2, 8];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 2.597, 0.001);
        let coding_1 = vec![2, 2, 9, 3];
        let coding_2 = vec![2, 2, 8, 2];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 3.556, 0.001);

        let coding_1 = vec![2, 2, 9, 3];
        let coding_2 = vec![2, 2, 2, 2, 8, 2];
        let (sim, len) = find_max_phase_similarity(&coding_1, &coding_2);
        assert_approx_eq!(sim, 3.556/(len as SimType), 0.001);

        let coding_1 = vec![8];
        let coding_2 = vec![8];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 1.0, 0.001);

        let coding_1 = vec![32,175,128,128];
        let coding_2 = vec![32,175,128,128];
        assert_approx_eq!(sum_quotients(&coding_1, &coding_2), 4.0, 0.001);

        let coding_1 = vec![8];
        let coding_2 = vec![8];
        let (sim, len) = find_max_phase_similarity(&coding_1, &coding_2);
        assert_approx_eq!(sim, 1.0/(len as SimType), 0.001);

        let coding_1 = vec![32,175,128,128];
        let coding_2 = vec![32,175,128,128];
        let (sim, len) = find_max_phase_similarity(&coding_1, &coding_2);
        assert_approx_eq!(sim, 4.0/(len as SimType), 0.001);

        let job_coding_1: Vec<Vec<CodingType>> = vec![vec![8],vec![8],vec![32,175,128,128]];
        let job_coding_2: Vec<Vec<CodingType>> = vec![vec![8],vec![8],vec![32,175,128,128]];
        assert_eq!(job_similarity(&job_coding_1, &job_coding_2), 1.0);
    }


    #[test]
    /// (sim1*len1 + sim2*len2 + sim3*len4) / (len1 + len2 + len3)
    fn test_weighted_mean() {
        let mut data: Vec<(SimType, PhaseLength)> = Vec::new();
        data.push((1.0, 1));
        data.push((1.0, 1));
        data.push((1.0, 4));
        let sim = weighted_mean(&data);
        println!("sim {}", sim);
        assert_approx_eq!(sim, 1.0, 0.001);
    }

    #[test]
    fn test_job_similarity() {
        //let job_coding_1: Vec<Vec<CodingType>> = vec![vec![2, 2, 9, 3], vec![9, 1, 1]];
        //let job_coding_2: Vec<Vec<CodingType>> = vec![vec![2, 2, 2, 2, 8, 2], vec![1], vec![8, 1, 1]];
        //assert_eq!(job_similarity(&job_coding_1, &job_coding_2), 0.8095239);
        let job_coding_1: Vec<Vec<CodingType>> = vec![vec![8],vec![8],vec![32,175,128,128]];
        let job_coding_2: Vec<Vec<CodingType>> = vec![vec![8],vec![8],vec![32,175,128,128]];
        assert_eq!(job_similarity(&job_coding_1, &job_coding_2), 1.0);
    }

    #[test]
    fn test_detect_phases() {
        let coding: Vec<CodingType> = vec![2, 2, 2, 2, 8, 2, 0, 0, 0, 1, 0, 8, 1, 1, 0];
        let res_phases = detect_phases(&coding);
        let expected_phases: Vec<Vec<CodingType>> = vec![vec![2, 2, 2, 2, 8, 2], vec![1], vec![8, 1, 1]];
        assert_eq!(res_phases, expected_phases);
    }
}

