extern crate csv;
extern crate serde;
extern crate chrono;
extern crate threadpool;

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

pub struct Config {
    pub dataset_fn: String,
    pub output_fn: String,
    pub nrows: usize,
    //pub batch_size: usize,
    pub min_similarity: algorithm::SimType,
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


pub fn convert_to_coding(coding: String) -> Vec<u16> {
    let split = coding.split(":");
    let vec: Vec<u16> = split
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap()) 
        .collect();
    vec
}

#[derive(Debug, Serialize)]
pub struct OutputRow {
    pub jobid_1: u32,
    pub jobid_2: u32,
    //pub num_phases_1: u8,
    //pub num_phases_2: u8,
    pub len_1: u8,
    pub len_2: u8,
    pub sim_abs: algorithm::SimType,
    pub sim_abs_aggzeros: algorithm::SimType,
    pub sim_hex: algorithm::SimType,
    pub sim_phases: algorithm::SimType,
}

#[derive(Debug, Clone)]
struct IsolatedPhases {
    jobid: u32,
    coding_abs: Vec<u16>, 
    coding_abs_aggzeros: Vec<u16>, 
    coding_hex: Vec<Vec<u16>>,
    phases: Vec<Vec<Vec<u16>>>,
    len: u8,
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
        if coding_abs_aggzeros.iter().sum::<u16>() > 0 {
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
    let channel_buf_size = 1000;
    let (tx, rx) = sync_channel(channel_buf_size);


    //let file = File::create(&cfg.output_fn).expect("Unable to open");
    //let wtr = Arc::new(Mutex::new(csv::Writer::from_writer(file)));

    let n_jobs = std::cmp::min(phases_set.len(), cfg.nrows);


    for p1 in phases_set1.iter().take(n_jobs) {
       let tx_clone = tx.clone();
       //let nrows = cfg.nrows;
       let p1_clone = p1.clone();
       let phases_set2_clone = phases_set2.clone();
       let min_similarity = cfg.min_similarity;
       //let wtr = wtr.clone();

       pool.execute( move || {
           let mut rows: Vec<OutputRow> = Vec::new();
           for p2 in phases_set2_clone.iter().take(n_jobs).skip(counter) {

               let mut sim_abs = algorithm2::compute_similarity_1d(&p1_clone.coding_abs, &p2.coding_abs);
               let mut sim_abs_aggzeros = algorithm2::compute_similarity_1d(&p1_clone.coding_abs_aggzeros, &p2.coding_abs_aggzeros);
               let mut sim_hex = algorithm2::compute_similarity_2d(&p1_clone.coding_hex, &p2.coding_hex);
               let mut sim_phases=  algorithm::job_similarity_2d(&p1_clone.phases, &p2.phases);

               if sim_abs < min_similarity {sim_abs = std::f32::NAN;}
               if sim_abs_aggzeros < min_similarity {sim_abs_aggzeros = std::f32::NAN;};
               if sim_hex < min_similarity {sim_hex = std::f32::NAN;}
               if sim_phases < min_similarity {sim_phases = std::f32::NAN;}

               if (sim_abs >= min_similarity) | (sim_abs_aggzeros >= min_similarity) | (sim_hex >= min_similarity) | (sim_phases >= min_similarity) {
                   let row = OutputRow {
                       jobid_1: p1_clone.jobid, 
                       jobid_2: p2.jobid,  
                       len_1: p1_clone.len,
                       len_2: p2.len,
                       sim_abs: sim_abs,
                       sim_abs_aggzeros: sim_abs_aggzeros, 
                       sim_hex: sim_hex, 
                       sim_phases: sim_phases, 
                   };
                   rows.push(row);
               }
           }
           tx_clone.send(rows).unwrap(); 
       });
       counter += 1;
    }

   
    let file = File::create(&cfg.output_fn).expect("Unable to open");
    let mut wtr = csv::Writer::from_writer(&file);

    let start = chrono::Utc::now();
    //let mut batch_counter = 0;

    let mut rx_iter = rx.iter();
    for i in 0..n_jobs {
        let rows = rx_iter.next().unwrap();
        for row in rows {
            wtr.serialize(row)?;
        }
        println!("batch {}/{} ({:.3}%)", 
                 i, 
                 n_jobs, 
                 ((100 * i) as f32) / (n_jobs as f32)
                );
    }

    let stop = chrono::Utc::now();
    println!("Flushing data.");
    wtr.flush()?;

    println!("Duration {}", ((stop - start).num_milliseconds() as f64) / (1000 as f64));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_to_coding() {
        let coding = String::from("256:256:0:0:38");
        let c = convert_to_coding(coding);
        let expected_c: Vec<u16> = vec![256, 256, 0, 0, 38];
        assert_eq!(expected_c, c);
    }
}

