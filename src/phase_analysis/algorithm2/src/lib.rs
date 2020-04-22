
extern crate generic_levenshtein;



//use generic_levenshtein;

pub type CodingType = u16;
pub type SimType = f32;


pub fn compute_similarity_1d(coding_1: &Vec<CodingType>, coding_2: &Vec<CodingType>) -> SimType {
    let d = generic_levenshtein::distance(coding_1, coding_2);
    let s: SimType = (1 as SimType) - (d as SimType) / (std::cmp::max(coding_1.len(), coding_2.len()) as SimType);
    s
}


pub fn compute_similarity_2d(coding_1: &Vec<Vec<CodingType>>, coding_2: &Vec<Vec<CodingType>>) -> SimType {
    let sim_sum = coding_1.iter().zip(coding_2.iter())
        .map(|(mc_1, mc_2)| { compute_similarity_1d(&mc_1, &mc_2) })
        .sum::<SimType>();
    let n_metrics = coding_1.len(); 
    sim_sum /(n_metrics as SimType)
}


#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use super::*;

    #[test]
    fn test_levenshtein() {
        let c1 = vec![1, 2, 3, 4, 5];
        let c2 = vec![1, 2, 3, 4, 5];
        assert_eq!(generic_levenshtein::distance(&c1, &c2), 0);
    }

    #[test]
    fn test_compute_similarity_1d() {
        let c1 = vec![1, 2, 3, 4];
        let c2 = vec![1, 2, 3, 4, 5];
        assert_approx_eq!(compute_similarity_1d(&c1, &c2), 0.8, 0.001);
    }

    #[test]
    fn test_compute_similarity_2d() {
        // job1 
        // metric1: vec![1, 2, 3, 4]
        // metric2: vec![0, 1, 0, 0]
        //
        // job2
        // metric1: vec![1, 2, 3, 4, 5]
        // metric2: vec![0, 0, 0, 0, 1]
        //
        // Similarities:
        // metric1: (1 - 1/5) = 0.8 
        // metric2: (1 - 2/5) = 0.6
        // mean: (0.8 + 0.6) / 2 = 0.7
        
        let c1 = vec![vec![1, 2, 3, 4], vec![0, 1, 0, 0]];
        let c2 = vec![vec![1, 2, 3, 4, 5], vec![0, 0, 0, 0, 1]];
        assert_approx_eq!(compute_similarity_2d(&c1, &c2), 0.7, 0.001);
    }
}




//def compute_similarity(probe_sec:pd.Series, group_pri:pd.DataFrame, eps:float, dist_type:str) -> pd.DataFrame:
//    if dist_type == 'levenshtein':
//        sim: pd.Series = group_pri.apply(compute_dist_levenshtein, args=(probe_sec,), axis=1)
//    elif dist_type == 'hex':
//        sim: pd.Series = group_pri.apply(compute_dist_hex, args=(probe_sec,), axis=1)
//    else:
//        raise SystemExit('Distance type %s is not supported.' % dist_type)
//    #sim.index=['sim']
//    return sim
