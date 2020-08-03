extern crate num_cpus;
extern crate jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

use std::process;


fn main() {
    let label = String::from("hex_nzseg");

    let cfg = run::Config{
        dataset_fn: String::from("../../datasets/job_codings.csv"),
        //output_fn: String::from("../../evaluation/job_codings_clusters_hex.csv"),
        output_fn: format!("../../evaluation/job_codings_clusters_{}.csv", label),
        progress_fn: format!("../../evaluation/progress_{}.csv", label),
        nrows: 1_000_000,
        //nrows: 100_000,
        //nrows: 50_000,
        //nrows: 30_000,
        n_workers: num_cpus::get(),
    };

    if let Err(e) = run::run(cfg) {
       eprintln!("Error occured in run: {}", e);
       process::exit(1);
    }
}
