extern crate num_cpus;
extern crate jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

use std::process;

fn main() {
    let cfg = run::Config{
        dataset_fn: String::from("../../datasets/job_codings.csv"),
        output_fn: String::from("../../evaluation/job_codings_clusters_2.csv"),
        //nrows: 1_000_000,
        nrows: 100_000,
        //nrows: 30_000,
        n_workers: num_cpus::get(),
    };

    if let Err(e) = run::run(cfg) {
       eprintln!("Error occured in run: {}", e);
       process::exit(1);
    }
}
