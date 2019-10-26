use std::fs::File;
use std::io::BufWriter;

use clap::{App, Arg, ArgMatches};
use finalfusion::prelude::*;
use ndarray::Array2;
use rayon::ThreadPoolBuilder;
use stdinout::OrExit;

use crate::io::{read_quantized_embeddings, QuantizedEmbeddingFormat};
use crate::FinalfusionApp;

// Option constants
static N_THREADS: &str = "n_threads";
static INPUT_FORMAT: &str = "input_format";

// Argument constants
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

pub struct ReconstructApp {
    input_filename: String,
    input_format: QuantizedEmbeddingFormat,
    output_filename: String,
    n_threads: usize,
}

impl FinalfusionApp for ReconstructApp {
    fn app() -> App<'static, 'static> {
        App::new("reconstruct")
            .about("Reconstruct embedding matrices from quantized embeddings")
            .arg(
                Arg::with_name(INPUT)
                    .help("Finalfusion model")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(OUTPUT)
                    .help("Output finalfusion model")
                    .index(2)
                    .required(true),
            )
            .arg(
                Arg::with_name(N_THREADS)
                    .short("t")
                    .long("threads")
                    .value_name("N")
                    .help("Number of threads (default: logical_cpus /2)")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(INPUT_FORMAT)
                    .short("f")
                    .long("from")
                    .value_name("FORMAT")
                    .takes_value(true)
                    .possible_values(&["finalfusion", "finalfusion_mmap"])
                    .default_value("finalfusion"),
            )
    }

    fn parse(matches: &ArgMatches) -> Self {
        // Arguments
        let input_filename = matches.value_of(INPUT).unwrap().to_owned();
        let output_filename = matches.value_of(OUTPUT).unwrap().to_owned();

        // Options
        let n_threads = matches
            .value_of(N_THREADS)
            .map(|a| a.parse().or_exit("Cannot parse number of threads", 1))
            .unwrap_or(num_cpus::get() / 2);

        let input_format = matches
            .value_of(INPUT_FORMAT)
            .map(|v| QuantizedEmbeddingFormat::try_from(v).or_exit("Cannot parse input format", 1))
            .unwrap();

        ReconstructApp {
            input_filename,
            input_format,
            output_filename,
            n_threads,
        }
    }

    fn run(&self) {
        env_logger::init();

        ThreadPoolBuilder::new()
            .num_threads(self.n_threads)
            .build_global()
            .unwrap();

        let embeddings = read_quantized_embeddings(&self.input_filename, self.input_format)
            .or_exit("Cannot read embeddings", 1);
        let (metadata, vocab, storage, norms) = embeddings.into_parts();

        // Reconstruct storage
        let reconstructed_storage = reconstruct_storage(&storage);
        let reconstructed_embeddings =
            Embeddings::new(metadata, vocab, reconstructed_storage, norms.unwrap());
        write_embeddings(&reconstructed_embeddings, &self.output_filename);
    }
}

fn reconstruct_storage(storage: &StorageWrap) -> NdArray {
    let mut reconstructions = Array2::zeros(storage.shape());
    for (idx, mut reconstruction) in reconstructions.outer_iter_mut().enumerate() {
        reconstruction.assign(&storage.embedding(idx));
    }
    reconstructions.into()
}

fn write_embeddings(embeddings: &Embeddings<VocabWrap, NdArray>, filename: &str) {
    let f = File::create(filename).or_exit("Cannot create embeddings file", 1);
    let mut writer = BufWriter::new(f);
    embeddings
        .write_embeddings(&mut writer)
        .or_exit("Cannot write embeddings", 1)
}
