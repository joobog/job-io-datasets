extern crate csv;
extern crate serde;
extern crate chrono;
extern crate threadpool;
extern crate ordered_float;

use algorithm;
//use algorithm2;
use std::sync::mpsc::channel;
//use std::sync::mpsc::sync_channel;
use std::sync::Arc;
//use std::sync::Mutex;
use std::error::Error;
use std::fs::File;
use serde::Deserialize;
use serde::Serialize;
use threadpool::ThreadPool;
use std::collections::HashMap;
use ordered_float::OrderedFloat;
//use std::thread;

pub struct Config {
    pub dataset_fn: String,
    pub output_fn: String,
    pub nrows: usize,
    pub n_workers: usize,
}


#[derive(Debug, Deserialize)]
pub struct Record {
    jobid: u32,
    md_file_create: String,
    md_file_delete: String,
    md_mod: String,
    md_other: String,
    md_read: String,
    read_bytes: String,
    read_calls: String,
    write_bytes: String,
    write_calls: String,
    coding_abs: String,
    coding_abs_aggzeros: String,
}

pub struct Entity {
    pub jobid: u32,
    pub sim: f32,
}


struct Cluster {
    pub centroid: u32,
    pub entities: Vec<Entity>,
}


#[derive(Clone)]
pub struct Profile<T> {
    pub name: String,
    pub id: u32,
    pub dataset: HashMap<Jobid,T>,
    pub func: fn(&T, &T) -> f32,
}

type Jobid = u32;
type HexCoding = Vec<Vec<algorithm2::CodingType>>;
type AbsCoding = Vec<algorithm2::CodingType>;
type AbsAggzerosCoding = Vec<algorithm2::CodingType>;
type PhasesCoding= Vec<Vec<Vec<algorithm::CodingType>>>;

//#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize)]
#[derive(Clone)]
pub enum ALG {
    Phases(Profile<PhasesCoding>),
    Abs(Profile<AbsCoding>),
    AbsAggzeros(Profile<AbsAggzerosCoding>),
    Hex(Profile<HexCoding>),
}




//#[derive(Debug, Clone)]
//struct IsolatedPhases {
//    jobid: u32,
//    coding_abs: Vec<u32>, 
//    coding_abs_aggzeros: Vec<u32>, 
//    coding_hex: Vec<Vec<u32>>,
//    phases: Vec<Vec<Vec<u32>>>,
//    len: u8,
//}


//#[derive()]
//pub struct JobComparison {
//    pub jobid_1: u32,
//    pub jobid_2: u32,
//    pub alg_type: ALG,
//    pub sim: f32,
//    pub threshold_sim: OrderedFloat<f32>,
//}


#[derive(Debug, Serialize)]
pub struct OutputRow {
    pub jobid: u32,
    pub cluster: u32,
    pub alg_type: u32,
    pub sim: f32,
    pub threshold_sim: f32,
}



pub fn convert_to_coding(coding: String) -> Vec<u32> {
    let split = coding.split(":");
    let vec: Vec<u32> = split
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap()) 
        .collect();
    vec
}



fn cluster<V> (cfg: &Config, min_sim: OrderedFloat<f32>, codings: &HashMap<Jobid,V>, cluster_func: fn(&V, &V) -> f32) -> Vec<Cluster>
{
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut avail_codings: Vec<(u32, &V)> = codings.iter().take(cfg.nrows).map(|(k, v)| (*k, v.clone())).collect();
    //println!("Grouping {:?}, Alg {:?}", min_sim, alg);
    while let Some((jobid, coding)) = avail_codings.pop() {
       let mut cluster_sims = Vec::new();
       for cluster in clusters.iter() {
           let coding_1 = codings.get(&cluster.centroid).unwrap(); // get centroid
           //cluster_sims.push(OrderedFloat(algorithm2::compute_similarity_1d(&coding_1, &coding)));
           cluster_sims.push(OrderedFloat(cluster_func(&coding_1, &coding)));
       }

       if let Some(max) = cluster_sims.iter().max() {
           if *max > min_sim {
               if let Some(max_position) = cluster_sims.iter().position(|x| x == max) {
                   clusters[max_position].entities.push(Entity{jobid: jobid, sim: max.into_inner()});
                   continue;
               }
           }
       }

       let new_cluster = Cluster{centroid: jobid, entities: vec![Entity{jobid: jobid, sim: 1.0}]};
       clusters.push(new_cluster);
    }
    clusters
}



pub fn run(cfg: Config) -> Result<(), Box<dyn Error>> {
    let file = File::open(&cfg.dataset_fn).expect("Unable to open dataset.");
    let mut rdr = csv::Reader::from_reader(file);

    let mut hex_codings: HashMap<Jobid, HexCoding> = HashMap::new();
    let mut abs_codings: HashMap<Jobid, AbsCoding> = HashMap::new();
    let mut abs_aggzeros_codings: HashMap<Jobid, AbsAggzerosCoding> = HashMap::new();
    let mut phases_codings: HashMap<Jobid, PhasesCoding> = HashMap::new();

    for result in rdr.deserialize() {
        let record: Record = result.expect("bla bla");
        let hex_coding = vec![
			convert_to_coding(record.md_file_create),
			convert_to_coding(record.md_file_delete),
			convert_to_coding(record.md_mod),
			convert_to_coding(record.md_other),
			convert_to_coding(record.md_read),
			convert_to_coding(record.read_bytes),
			convert_to_coding(record.read_calls),
			convert_to_coding(record.write_bytes),
			convert_to_coding(record.write_calls),];

        if hex_coding[0].len() > 0 {
            abs_codings.insert(record.jobid, convert_to_coding(record.coding_abs));
            abs_aggzeros_codings.insert(record.jobid, convert_to_coding(record.coding_abs_aggzeros));
            phases_codings.insert(record.jobid, algorithm::detect_phases_2d(&hex_coding));
            hex_codings.insert(record.jobid, hex_coding);
        }
    }

    let min_sims: Vec<OrderedFloat<f32>> = vec![
        OrderedFloat(0.5), 
        OrderedFloat(0.7),
        OrderedFloat(0.9),
        OrderedFloat(0.95),
        OrderedFloat(0.99),
    ];

    let mut algs = Vec::new();
    algs.push(ALG::Abs(Profile{name: String::from("abs"), id:1, dataset: abs_codings, func: algorithm2::compute_similarity_1d,}));
    algs.push(ALG::AbsAggzeros(Profile{name: String::from("abs_aggzeros"), id:2, dataset: abs_aggzeros_codings, func: algorithm2::compute_similarity_1d,}));
    algs.push(ALG::Hex(Profile{name: String::from("hex"), id:3, dataset: hex_codings, func: algorithm2::compute_similarity_2d,}));
    algs.push(ALG::Phases(Profile{name: String::from("phases"), id:4, dataset: phases_codings, func: algorithm::job_similarity_2d,}));

    let cfg = Arc::new(cfg);

    let pool = ThreadPool::new(cfg.n_workers);
    //let channel_buf_size = 2000;
    //let (tx, rx) = sync_channel(channel_buf_size);
    let (tx, rx) = channel();

    for main_min_sim in min_sims.iter() {
        let min_sim = *main_min_sim;
        for main_alg in algs.iter() {
            match main_alg.clone() {
                ALG::Abs(p) | ALG::AbsAggzeros(p)  => {
                    let alg = main_alg.clone();
                    let cfg = cfg.clone();
                    let codings = p.dataset.clone();
                    let tx = tx.clone();
                    pool.execute( move || {
                        let clusters = cluster(&cfg, min_sim, &codings, p.func);
                        tx.send((alg, min_sim, clusters)).unwrap();
                    });
                }
                ALG::Hex(p) => {
                    let alg = main_alg.clone();
                    let cfg = cfg.clone();
                    let codings = p.dataset.clone();
                    let tx = tx.clone();
                    pool.execute( move || {
                        let clusters = cluster(&cfg, min_sim, &codings, p.func);
                        tx.send((alg, min_sim, clusters)).unwrap();
                    });
                }
                ALG::Phases(p) => {
                    let alg = main_alg.clone();
                    let cfg = cfg.clone();
                    let codings = p.dataset.clone();
                    let tx = tx.clone();
                    pool.execute( move || {
                        let clusters = cluster(&cfg, min_sim, &codings, p.func);
                        tx.send((alg, min_sim, clusters)).unwrap();
                    });
                }
            };
        }
    }


    let file = File::create(&cfg.output_fn).expect("Unable to open");
    let mut wtr = csv::Writer::from_writer(&file);
    let start = chrono::Utc::now();

    let mut rx_iter = rx.iter();

    for _ in min_sims.iter() {
        for _ in algs.iter() {
            let (alg, min_sim, clusters) = rx_iter.next().unwrap();
            let alg_n = match alg {
                ALG::Abs(p) => p.id,
                ALG::AbsAggzeros(p)  => p.id,
                ALG::Hex(p) => p.id,
                ALG::Phases(p) => p.id
            };

            for cluster in clusters.iter() {
                let cluster_id = cluster.centroid;
                for entity in cluster.entities.iter() {
                    let output_row = OutputRow {
                        jobid: entity.jobid,
                        cluster:  cluster_id,
                        alg_type: alg_n,
                        sim: entity.sim,
                        threshold_sim: min_sim.into_inner(),
                    };

                    wtr.serialize(output_row)?;
                }
            }
        }
    }

    println!("Flushing data.");
    wtr.flush()?;
    let stop = chrono::Utc::now();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_to_coding() {
        let coding = String::from("256:256:0:0:38");
        let c = convert_to_coding(coding);
        let expected_c: Vec<u32> = vec![256, 256, 0, 0, 38];
        assert_eq!(expected_c, c);
    }
}

