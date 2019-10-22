use std::fs::File;
use std::io::BufWriter;
use std::process;

use clap::{App, Arg, ArgMatches};
use finalfusion::embeddings::Quantize;
use finalfusion::io::WriteEmbeddings;
use finalfusion::prelude::*;
use finalfusion::storage::{QuantizedArray, Storage, StorageView};
use finalfusion::vocab::Vocab;
use ndarray::ArrayView1;
use rayon::ThreadPoolBuilder;
use reductive::pq::PQ;
#[cfg(feature = "opq")]
use reductive::pq::{GaussianOPQ, OPQ};
use stdinout::OrExit;

use crate::io::{read_embeddings_view, EmbeddingFormat};
use crate::FinalfusionApp;

// Option constants
static INPUT_FORMAT: &str = "input_format";
static N_ATTEMPTS: &str = "n_attempts";
static N_ITERATIONS: &str = "n_iterations";
static N_SUBQUANTIZERS: &str = "n_subquantizers";
static N_THREADS: &str = "n_threads";
static QUANTIZER: &str = "quantizer";
static QUANTIZER_BITS: &str = "quantizer_bits";

// Argument constants
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

pub struct QuantizeApp {
    input_filename: String,
    input_format: EmbeddingFormat,
    n_attempts: usize,
    n_iterations: usize,
    n_subquantizers: Option<usize>,
    n_threads: usize,
    output_filename: String,
    quantizer: String,
    quantizer_bits: u32,
}

impl FinalfusionApp for QuantizeApp {
    fn app() -> App<'static, 'static> {
        App::new("quantize")
            .about("Quantize embedding matrices")
            .arg(
                Arg::with_name(INPUT)
                    .help("Finalfrontier model")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(OUTPUT)
                    .help("Output file")
                    .index(2)
                    .required(true),
            )
            .arg(
                Arg::with_name(N_ATTEMPTS)
                    .short("a")
                    .long("attempts")
                    .value_name("N")
                    .help("Number of quantization attempts")
                    .takes_value(true)
                    .default_value("1"),
            )
            .arg(
                Arg::with_name(QUANTIZER_BITS)
                    .short("b")
                    .long("bits")
                    .value_name("N")
                    .help("Number of quantizer bits (max: 8)")
                    .takes_value(true)
                    .default_value("8"),
            )
            .arg(
                Arg::with_name(INPUT_FORMAT)
                    .short("f")
                    .long("from")
                    .value_name("FORMAT")
                    .takes_value(true)
                    .possible_values(&["fasttext", "finalfusion", "text", "textdims", "word2vec"])
                    .default_value("word2vec"),
            )
            .arg(
                Arg::with_name(N_ITERATIONS)
                    .short("i")
                    .long("iter")
                    .value_name("N")
                    .help("Number of iterations")
                    .takes_value(true)
                    .default_value("100"),
            )
            .arg(
                Arg::with_name(QUANTIZER)
                    .short("q")
                    .long("quantizer")
                    .value_name("QUANTIZER")
                    .takes_value(true)
                    .possible_values(&["gaussian_opq", "opq", "pq"])
                    .default_value("pq"),
            )
            .arg(
                Arg::with_name(N_SUBQUANTIZERS)
                    .short("s")
                    .long("subquantizers")
                    .value_name("N")
                    .help("Number of subquantizers (default: d/2)")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(N_THREADS)
                    .short("t")
                    .long("threads")
                    .value_name("N")
                    .help("Number of threads (default: logical_cpus /2)")
                    .takes_value(true),
            )
    }

    fn parse(matches: &ArgMatches) -> Self {
        // Arguments
        let input_filename = matches.value_of(INPUT).unwrap().to_owned();
        let output_filename = matches.value_of(OUTPUT).unwrap().to_owned();

        // Options
        let input_format = matches
            .value_of(INPUT_FORMAT)
            .map(|v| EmbeddingFormat::try_from(v).or_exit("Cannot parse input format", 1))
            .unwrap();
        let n_attempts = matches
            .value_of(N_ATTEMPTS)
            .map(|a| a.parse().or_exit("Cannot parse number of attempts", 1))
            .unwrap();
        let n_iterations = matches
            .value_of(N_ITERATIONS)
            .map(|a| a.parse().or_exit("Cannot parse number of iterations", 1))
            .unwrap();
        let n_subquantizers = matches
            .value_of(N_SUBQUANTIZERS)
            .map(|a| a.parse().or_exit("Cannot parse number of subquantizers", 1));
        let n_threads = matches
            .value_of(N_THREADS)
            .map(|a| a.parse().or_exit("Cannot parse number of threads", 1))
            .unwrap_or(num_cpus::get() / 2);
        let quantizer = matches.value_of(QUANTIZER).map(ToOwned::to_owned).unwrap();
        let quantizer_bits = matches
            .value_of(QUANTIZER_BITS)
            .map(|a| {
                a.parse()
                    .or_exit("Cannot parse number of quantizer_bits", 1)
            })
            .unwrap();
        if quantizer_bits > 8 {
            eprintln!(
                "Maximum number of quantizer bits: 8, was: {}",
                quantizer_bits
            );
            process::exit(1);
        }

        QuantizeApp {
            input_filename,
            input_format,
            n_attempts,
            n_iterations,
            n_subquantizers,
            n_threads,
            output_filename,
            quantizer,
            quantizer_bits,
        }
    }

    fn run(&self) {
        env_logger::init();

        ThreadPoolBuilder::new()
            .num_threads(self.n_threads)
            .build_global()
            .unwrap();

        let embeddings = read_embeddings_view(&self.input_filename, self.input_format)
            .or_exit("Cannot read embeddings", 1);

        // Quantize
        let quantized_embeddings = quantize_embeddings(&self, &embeddings);
        write_embeddings(&quantized_embeddings, &self.output_filename);

        print_loss(embeddings.storage(), quantized_embeddings.storage());
    }
}

fn cosine_similarity(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
    let u_norm = u.dot(&u).sqrt();
    let v_norm = v.dot(&v).sqrt();
    u.dot(&v) / (u_norm * v_norm)
}

fn euclidean_distance(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
    let dist_vec = &u - &v;
    dist_vec.dot(&dist_vec).sqrt()
}

fn print_loss(storage: &dyn StorageView, quantized_storage: &dyn Storage) {
    let mut cosine_similarity_sum = 0f32;
    let mut euclidean_distance_sum = 0f32;

    for (idx, embedding) in storage.view().outer_iter().enumerate() {
        let reconstruction = quantized_storage.embedding(idx);
        cosine_similarity_sum += cosine_similarity(embedding, reconstruction.view());
        euclidean_distance_sum += euclidean_distance(embedding, reconstruction.view());
    }

    eprintln!(
        "Average cosine similarity: {}",
        cosine_similarity_sum / storage.view().nrows() as f32
    );

    eprintln!(
        "Average euclidean distance: {}",
        euclidean_distance_sum / storage.view().nrows() as f32
    );
}

#[cfg(not(feature = "opq"))]
fn quantize_embeddings<V, S>(
    config: &QuantizeApp,
    embeddings: &Embeddings<V, S>,
) -> Embeddings<V, QuantizedArray>
where
    V: Vocab + Clone,
    S: StorageView,
{
    let n_subquantizers = config
        .n_subquantizers
        .unwrap_or(embeddings.storage().shape().1 / 2);

    match config.quantizer.as_str() {
        "pq" => embeddings.quantize::<PQ<f32>>(
            n_subquantizers,
            config.quantizer_bits,
            config.n_iterations,
            config.n_attempts,
            true,
        ),
        quantizer => {
            eprintln!("Unknown quantizer: {}", quantizer);
            process::exit(1);
        }
    }
}

#[cfg(feature = "opq")]
fn quantize_embeddings<V, S>(
    config: &QuantizeApp,
    embeddings: &Embeddings<V, S>,
) -> Embeddings<V, QuantizedArray>
where
    V: Vocab + Clone,
    S: StorageView,
{
    let n_subquantizers = config
        .n_subquantizers
        .unwrap_or(embeddings.storage().shape().1 / 2);

    match config.quantizer.as_str() {
        "pq" => embeddings.quantize::<PQ<f32>>(
            n_subquantizers,
            config.quantizer_bits,
            config.n_iterations,
            config.n_attempts,
            true,
        ),
        "opq" => embeddings.quantize::<OPQ>(
            n_subquantizers,
            config.quantizer_bits,
            config.n_iterations,
            config.n_attempts,
            true,
        ),
        "gaussian_opq" => embeddings.quantize::<GaussianOPQ>(
            n_subquantizers,
            config.quantizer_bits,
            config.n_iterations,
            config.n_attempts,
            true,
        ),
        quantizer => {
            eprintln!("Unknown quantizer: {}", quantizer);
            process::exit(1);
        }
    }
}

fn write_embeddings(embeddings: &Embeddings<VocabWrap, QuantizedArray>, filename: &str) {
    let f = File::create(filename).or_exit("Cannot create embeddings file", 1);
    let mut writer = BufWriter::new(f);
    embeddings
        .write_embeddings(&mut writer)
        .or_exit("Cannot write embeddings", 1)
}
