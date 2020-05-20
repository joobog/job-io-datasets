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
use std::sync::Mutex;
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
    pub progress_fn: String,
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

#[derive(Debug, Serialize)]
pub struct Progress {
    min_sim: f32,
    alg: u32,
    nclusters: usize,
    jobs_done: usize,
    jobs_total: usize,
    elapsed: f64,
    delta: f64,
}


#[derive(Debug, Clone, Copy)]
pub struct Entity {
    pub jobid: u32,
    pub sim: f32,
}

struct Cluster {
    pub centroid_jobid: u32,
    pub entities: Vec<Entity>,
}

struct ClusterCentroid<T> {
    pub centroid_jobid: u32,
    pub centroid_coding: T,
    pub entities: Vec<Entity>,
}


#[derive(Clone)]
pub struct Profile<T> {
    pub name: String,
    pub id: u32,
    pub dataset: HashMap<Jobid,T>,
    pub func: fn(&T, &T) -> f32,
}

impl<T> std::fmt::Debug for Profile<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("").finish()
    }
}

type Jobid = u32;
type HexCoding = Vec<Vec<algorithm2::CodingType>>;
type AbsCoding = Vec<algorithm2::CodingType>;
type AbsAggzerosCoding = Vec<algorithm2::CodingType>;
type PhasesCoding= Vec<Vec<Vec<algorithm::CodingType>>>;

//#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize)]
#[derive(Debug, Clone)]
pub enum ALG {
    Phases(Profile<PhasesCoding>),
    Abs(Profile<AbsCoding>),
    AbsAggzeros(Profile<AbsAggzerosCoding>),
    Hex(Profile<HexCoding>),
}


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


fn cluster<V, W> (cfg: &Config, wtr: Arc<Mutex<csv::Writer<W>>>,alg: &ALG, min_sim: OrderedFloat<f32>, codings: &HashMap<Jobid,V>, cluster_func: fn(&V, &V) -> f32) -> Vec<Cluster> 
where W: std::io::Write
{
    //println!("Grouping {:?}, ALG {:?}", min_sim, alg);
    let start = chrono::Utc::now();
    let mut clusters: Vec<ClusterCentroid<&V>> = Vec::new();
    let mut avail_codings: Vec<(u32, &V)> = codings.iter().take(cfg.nrows).map(|(k, v)| (*k, v)).collect();
    let mut found_cluster;
    let mut counter = 0;
    let mut start_loop = chrono::Utc::now();
    while let Some((jobid, coding)) = avail_codings.pop() {
        found_cluster = false;
        //let nclusters = clusters.len();
        for cluster in clusters.iter_mut() {
            let sim = cluster_func(&cluster.centroid_coding, &coding);
            if sim >= min_sim.into_inner() {
                // append to existing cluster
                cluster.entities.push(Entity{jobid: jobid, sim: sim});
                found_cluster = true;
                break;
            }
        }
        // create new cluster
        if !found_cluster {
            //println!("Add cluster jobid = {}, clusters = {}", jobid, clusters.len());
            clusters.push(ClusterCentroid{
                centroid_jobid: jobid, 
                centroid_coding: coding, 
                entities: vec![Entity{jobid: jobid, sim: 1.0}]
            });
        }
        if (counter % 10_000) == 0 {
            let stop_loop = chrono::Utc::now();
            println!("{:?}, ALG {:?}, nclusters {:?}, left = {:?}/{:?}, ({:.3} seconds)", 
                     min_sim,
                     alg,
                     clusters.len(),
                     avail_codings.len(), 
                     codings.len(),
                     ((stop_loop - start_loop).num_milliseconds() as f64) / (1000 as f64)
                    );

            let alg_n = match alg {
                ALG::Abs(p) => p.id,
                ALG::AbsAggzeros(p)  => p.id,
                ALG::Hex(p) => p.id,
                ALG::Phases(p) => p.id
            };

            let progress = Progress{
                min_sim: min_sim.into_inner(),
                alg: alg_n,
                nclusters: clusters.len(),
                jobs_done: codings.len() - avail_codings.len(),
                jobs_total: codings.len(),
                elapsed: ((stop_loop - start).num_milliseconds() as f64) / (1000 as f64),
                delta: ((stop_loop - start_loop).num_milliseconds() as f64) / (1000 as f64)
            };

            {
                let mut wtr = wtr.lock().unwrap();
                wtr.serialize(progress).unwrap();
                wtr.flush().unwrap();
            }

            start_loop = stop_loop;
        }
        counter += 1;
    }
    // reshaping to common representation
    let clusters: Vec<_> = clusters.iter().map(|x| Cluster{centroid_jobid: x.centroid_jobid, entities: x.entities.clone(),}).collect();
    let stop = chrono::Utc::now();
    println!("Finish grouping {:?}, ALG {:?}, nclusters {:?}, ({:.3} seconds)", 
             min_sim,
             alg,
             clusters.len(),
             ((stop - start).num_milliseconds() as f64) / (1000 as f64)
            );
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
        let abs_aggzeros_coding = convert_to_coding(record.coding_abs_aggzeros);
        let abs_coding = convert_to_coding(record.coding_abs);
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
        let phases_coding = algorithm::detect_phases_2d(&hex_coding);

        if abs_aggzeros_coding.iter().sum::<u32>() > 0 {
            abs_aggzeros_codings.insert(record.jobid, abs_aggzeros_coding);
        }
        if abs_coding.iter().sum::<u32>() > 0 {
            abs_codings.insert(record.jobid, abs_coding);
        }
        if phases_coding.iter().map(|x| x.len()).sum::<usize>() > 0 {
            phases_codings.insert(record.jobid, phases_coding);
        }
        if hex_coding.iter().map(|x| x.iter().sum::<u32>()).sum::<u32>() > 0 {
           hex_codings.insert(record.jobid, hex_coding);
        }
    }

    let min_sims: Vec<OrderedFloat<f32>> = vec![
        OrderedFloat(0.1), 
        OrderedFloat(0.3), 
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
    let file = File::create(&cfg.progress_fn).expect("Unable to open progress file");
    let wtr = csv::Writer::from_writer(file);
    let wtr = Arc::new(Mutex::new(wtr));

    for main_min_sim in min_sims.iter() {
        let min_sim = *main_min_sim;
        let wtr = wtr.clone();

        for main_alg in algs.iter() {
            match main_alg.clone() {
                ALG::Abs(p) | ALG::AbsAggzeros(p)  => {
                    let alg = main_alg.clone();
                    let cfg = cfg.clone();
                    let codings = p.dataset.clone();
                    let tx = tx.clone();
                    let wtr = wtr.clone();
                    pool.execute( move || {
                        let clusters = cluster(&cfg, wtr, &alg, min_sim, &codings, p.func);
                        tx.send((alg, min_sim, clusters)).unwrap();
                    });
                }
                ALG::Hex(p) => {
                    let alg = main_alg.clone();
                    let cfg = cfg.clone();
                    let codings = p.dataset.clone();
                    let tx = tx.clone();
                    let wtr = wtr.clone();
                    pool.execute( move || {
                        let clusters = cluster(&cfg, wtr, &alg, min_sim, &codings, p.func);
                        tx.send((alg, min_sim, clusters)).unwrap();
                    });
                }
                ALG::Phases(p) => {
                    let alg = main_alg.clone();
                    let cfg = cfg.clone();
                    let codings = p.dataset.clone();
                    let tx = tx.clone();
                    let wtr = wtr.clone();
                    pool.execute( move || {
                        let clusters = cluster(&cfg, wtr, &alg, min_sim, &codings, p.func);
                        tx.send((alg, min_sim, clusters)).unwrap();
                    });
                }
            };
        }
        wtr.lock().unwrap().flush().unwrap();
    }


    let file = File::create(&cfg.output_fn).expect("Unable to open");
    let mut wtr = csv::Writer::from_writer(&file);
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
                let cluster_id = cluster.centroid_jobid;
                for entity in cluster.entities.iter() {
                    let output_row = OutputRow {
                        jobid: entity.jobid,
                        cluster: cluster_id,
                        alg_type: alg_n,
                        sim: entity.sim,
                        threshold_sim: min_sim.into_inner(),
                    };

                    wtr.serialize(output_row)?;
                }
            }
            wtr.flush()?;
        }
    }
    Ok(())
}
