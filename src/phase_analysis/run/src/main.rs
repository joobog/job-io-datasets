extern crate num_cpus;
use std::process;


fn main() {
    let root = String::from("../..//datasets");
    let dset_fn = format!("{}/job_codings.csv", root);
    let output_fn = format!("{}/job_codings_similarity.csv", root);

    println!("Processing {}", dset_fn);

    let cfg = run::Config{
        dataset_fn: dset_fn,
        output_fn: output_fn,
        nrows: 1_010_000,
        batch_size: 1000,
        min_similarity: 0.7,
        n_workers: num_cpus::get(),
    };

    if let Err(e) = run::run(cfg) {
       eprintln!("Error occured in run: {}", e);
       process::exit(1);
    }
}
