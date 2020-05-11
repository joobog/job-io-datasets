extern crate csv;
extern crate serde;
extern crate chrono;
extern crate threadpool;
extern crate ordered_float;

use algorithm;
//use algorithm2;
//use std::sync::mpsc::channel;
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::error::Error;
use std::fs::File;
use serde::Deserialize;
use serde::Serialize;
use threadpool::ThreadPool;
use std::collections::HashMap;
use ordered_float::OrderedFloat;

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


#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize)]
pub enum ALG {
    Phases,
    Abs,
    AbsAggzeros,
    Hex,
}


#[derive(Debug, Clone)]
struct IsolatedPhases {
    jobid: u32,
    coding_abs: Vec<u32>, 
    coding_abs_aggzeros: Vec<u32>, 
    coding_hex: Vec<Vec<u32>>,
    phases: Vec<Vec<Vec<u32>>>,
    len: u8,
}


#[derive(Debug)]
pub struct JobComparison {
    pub jobid_1: u32,
    pub jobid_2: u32,
    pub alg_type: ALG,
    pub sim: f32,
    pub threshold_sim: OrderedFloat<f32>,
}


#[derive(Debug, Serialize)]
pub struct OutputRow {
    pub jobid: u32,
    pub cluster: u32,
    pub alg_type: u32,
    pub sim: f32,
    pub threshold_sim: f32,
}


pub struct Summary {
    pub cluster: u32,
    pub sim: f32,
}


pub fn convert_to_coding(coding: String) -> Vec<u32> {
    let split = coding.split(":");
    let vec: Vec<u32> = split
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap()) 
        .collect();
    vec
}


pub fn convert_to_output_rows(input: HashMap<u32, HashMap<ALG, HashMap<OrderedFloat<f32>, Summary>>>, alg_map: &HashMap<ALG, u32>) -> Vec<OutputRow> {
    let mut output: Vec<OutputRow> = Vec::new();
    for (jobid, alg_hashmap) in input {
        for (alg, threshold_hashmap) in alg_hashmap {
            for (threshold, summary) in threshold_hashmap {
                output.push(
                    OutputRow {
                        jobid,
                        cluster: summary.cluster,
                        alg_type: alg_map[&alg],
                        sim: summary.sim,
                        threshold_sim: threshold.into_inner(),
                    });
            }
        }
    }
    output
}


pub fn run(cfg: Config) -> Result<(), Box<dyn Error>> {
    let file = File::open(&cfg.dataset_fn).expect("Unable to open dataset.");
    let mut rdr = csv::Reader::from_reader(file);

    let mut phases_set: Vec<IsolatedPhases> = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result.expect("bla bla");
        let coding_abs = convert_to_coding(record.coding_abs);
        let coding_abs_aggzeros = convert_to_coding(record.coding_abs_aggzeros);

		let coding_hex = vec![
			convert_to_coding(record.md_file_create),
			convert_to_coding(record.md_file_delete),
			convert_to_coding(record.md_mod),
			convert_to_coding(record.md_other),
			convert_to_coding(record.md_read),
			convert_to_coding(record.read_bytes),
			convert_to_coding(record.read_calls),
			convert_to_coding(record.write_bytes),
			convert_to_coding(record.write_calls),];
        let coding_length = coding_hex[0].len();

        let phases = algorithm::detect_phases_2d(&coding_hex);
        //if phases.len() > 0 {
        if coding_abs_aggzeros.iter().sum::<u32>() > 0 {
            phases_set.push(IsolatedPhases{
                jobid: record.jobid, 
                coding_abs: coding_abs,
                coding_abs_aggzeros: coding_abs_aggzeros,
                coding_hex: coding_hex,
                phases: phases,
                len: (coding_length as u8),
            });
        }
    }

    let mut phases_set1: Vec<Arc<IsolatedPhases>> = Vec::new();
    for item in phases_set.iter() {
        phases_set1.push(Arc::new(item.clone()));
    }
    let phases_set2: Arc<Vec<IsolatedPhases>> = Arc::new(phases_set.clone());

    let mut counter = 0;
    let pool = ThreadPool::new(cfg.n_workers);
    let channel_buf_size = 100;
    let (tx, rx) = sync_channel(channel_buf_size);
    let n_jobs = std::cmp::min(phases_set.len(), cfg.nrows);


    for p1 in phases_set1.iter().take(n_jobs) {
        let tx_clone = tx.clone();
        let p1_clone = p1.clone();
        let phases_set2_clone = phases_set2.clone();

        let threshold_sims: Vec<OrderedFloat<f32>> = vec![
            OrderedFloat(0.5), 
            OrderedFloat(0.7),
            OrderedFloat(0.9),
            OrderedFloat(0.95),
            OrderedFloat(0.99),
        ];

        pool.execute( move || {
            let mut rows: Vec<JobComparison> = Vec::new();
            for p2 in phases_set2_clone.iter().take(n_jobs).skip(counter) {
                let sim_abs = algorithm2::compute_similarity_1d(&p1_clone.coding_abs, &p2.coding_abs);
                let sim_abs_aggzeros = algorithm2::compute_similarity_1d(&p1_clone.coding_abs_aggzeros, &p2.coding_abs_aggzeros);
                let sim_hex = algorithm2::compute_similarity_2d(&p1_clone.coding_hex, &p2.coding_hex);
                let sim_phases = algorithm::job_similarity_2d(&p1_clone.phases, &p2.phases);
                //let sim_abs = 0.0;
                //let sim_abs_aggzeros = 0.0; 
                //let sim_hex = 0.0; 
                //let sim_phases = 0.0;

                for threshold_sim in threshold_sims.iter() {
                    if sim_abs > threshold_sim.into_inner() {
                        let row = JobComparison{
                            jobid_1: p1_clone.jobid,
                            jobid_2: p2.jobid,
                            alg_type: ALG::Abs,
                            sim: sim_abs,
                            threshold_sim: *threshold_sim,
                        };
                        rows.push(row);
                    }
                    if sim_hex > threshold_sim.into_inner() {
                        let row = JobComparison{
                            jobid_1: p1_clone.jobid,
                            jobid_2: p2.jobid,
                            alg_type: ALG::Hex,
                            sim: sim_hex,
                            threshold_sim: *threshold_sim,
                        };
                        rows.push(row);
                    }
                    if sim_abs_aggzeros > threshold_sim.into_inner() {
                        let row = JobComparison{
                            jobid_1: p1_clone.jobid,
                            jobid_2: p2.jobid,
                            alg_type: ALG::AbsAggzeros,
                            sim: sim_abs_aggzeros,
                            threshold_sim: *threshold_sim,
                        };
                        rows.push(row);
                    }
                    if sim_phases > threshold_sim.into_inner() {
                        let row = JobComparison{
                            jobid_1: p1_clone.jobid,
                            jobid_2: p2.jobid,
                            alg_type: ALG::Phases,
                            sim: sim_phases,
                            threshold_sim: *threshold_sim,
                        };
                        rows.push(row);
                    }
                }
            }
            tx_clone.send(rows).unwrap(); 
        });
        counter += 1;
    }

    let mut groups: HashMap<u32, HashMap<ALG, HashMap<OrderedFloat<f32>, Summary>>> = HashMap::new();
    let start_group = chrono::Utc::now();

    for phase in phases_set.iter() {
        groups.entry(phase.jobid).or_insert(HashMap::new()).entry(ALG::Abs).or_insert(HashMap::new());
        groups.entry(phase.jobid).or_insert(HashMap::new()).entry(ALG::AbsAggzeros).or_insert(HashMap::new());
        groups.entry(phase.jobid).or_insert(HashMap::new()).entry(ALG::Hex).or_insert(HashMap::new());
        groups.entry(phase.jobid).or_insert(HashMap::new()).entry(ALG::Phases).or_insert(HashMap::new());
    }

    let mut rx_iter = rx.iter();
    let mut start = chrono::Utc::now();
    for i in 0..n_jobs {
        let rows = rx_iter.next().unwrap();
        for row in rows {
            groups.get_mut(&row.jobid_2).unwrap().get_mut(&row.alg_type).unwrap().entry(row.threshold_sim).or_insert(
               Summary{
                   cluster: row.jobid_1,
                   sim: row.sim,
               });
        }
        if ((i + 1) % 1000) == 0 {
            let stop = chrono::Utc::now();
            println!("batch {}/{} ({:.3}%), ({:.3} seconds)", 
                     i + 1, 
                     n_jobs, 
                     ((100 * (i + 1)) as f32) / (n_jobs as f32),
                     ((stop - start).num_milliseconds() as f64) / (1000 as f64)
                    );
            start = stop;
        }
    }
    let stop_group = chrono::Utc::now();
    println!("Finished grouping {}", ((stop_group - start_group).num_milliseconds() as f64) / (1000 as f64));


    let mut alg_map: HashMap<ALG, u32> = HashMap::new();
    alg_map.insert(ALG::Abs, 1);
    alg_map.insert(ALG::AbsAggzeros, 2);
    alg_map.insert(ALG::Hex, 3);
    alg_map.insert(ALG::Phases, 4);

    let start = chrono::Utc::now();
    let output_rows = convert_to_output_rows(groups, &alg_map);
    let stop = chrono::Utc::now();
    println!("Conversion duration {}", ((stop - start).num_milliseconds() as f64) / (1000 as f64));


    let file = File::create(&cfg.output_fn).expect("Unable to open");
    let mut wtr = csv::Writer::from_writer(&file);
    let start = chrono::Utc::now();
    for output_row in output_rows {
       wtr.serialize(output_row)?;
    }

    println!("Flushing data.");
    wtr.flush()?;
    let stop = chrono::Utc::now();

    println!("Write duration {}", ((stop - start).num_milliseconds() as f64) / (1000 as f64));
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

