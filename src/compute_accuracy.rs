use std::collections::BTreeMap;
use std::io::BufRead;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::{App, AppSettings, Arg, ArgMatches};
use finalfusion::prelude::*;
use finalfusion::similarity::Analogy;
use finalfusion::vocab::Vocab;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use stdinout::Input;

use crate::io::{read_embeddings_view, EmbeddingFormat};
use crate::FinalfusionApp;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

// Option constants
static EMBEDDINGS: &str = "EMBEDDINGS";
static ANALOGIES: &str = "ANALOGIES";
static THREADS: &str = "threads";

pub struct ComputeAccuracyApp {
    analogies_filename: Option<String>,
    embeddings_filename: String,
    n_threads: usize,
}

impl FinalfusionApp for ComputeAccuracyApp {
    fn app() -> App<'static, 'static> {
        App::new("compute-accuracy")
            .about("Compute prediction accuracy on a set of analogies")
            .settings(DEFAULT_CLAP_SETTINGS)
            .arg(
                Arg::with_name(THREADS)
                    .long("threads")
                    .value_name("N")
                    .help("Number of threads (default: logical_cpus / 2)")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(EMBEDDINGS)
                    .help("Embedding file")
                    .index(1)
                    .required(true),
            )
            .arg(Arg::with_name(ANALOGIES).help("Analogy file").index(2))
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let embeddings_filename = matches.value_of(EMBEDDINGS).unwrap().to_owned();
        let analogies_filename = matches.value_of(ANALOGIES).map(ToOwned::to_owned);
        let n_threads = matches
            .value_of("threads")
            .map(|v| {
                v.parse()
                    .context(format!("Cannot parse number of threads: {}", v))
            })
            .transpose()?
            .unwrap_or(num_cpus::get() / 2);

        Ok(ComputeAccuracyApp {
            analogies_filename,
            embeddings_filename,
            n_threads,
        })
    }

    fn run(&self) -> Result<()> {
        ThreadPoolBuilder::new()
            .num_threads(self.n_threads)
            .build_global()
            .unwrap();

        let embeddings =
            read_embeddings_view(&self.embeddings_filename, EmbeddingFormat::FinalFusion)
                .context("Cannot read embeddings")?;

        let analogies_file = Input::from(self.analogies_filename.as_ref());
        let reader = analogies_file
            .buf_read()
            .context("Cannot open analogy file for reading")?;

        let instances = read_analogies(reader)?;
        process_analogies(&embeddings, &instances);

        Ok(())
    }
}

struct Counts {
    n_correct: usize,
    n_instances: usize,
    n_skipped: usize,
    sum_cos: f32,
}

impl Default for Counts {
    fn default() -> Self {
        Counts {
            n_correct: 0,
            n_instances: 0,
            n_skipped: 0,
            sum_cos: 0.,
        }
    }
}

#[derive(Clone)]
struct Eval<'a> {
    embeddings: &'a Embeddings<VocabWrap, StorageViewWrap>,
    section_counts: Arc<Mutex<BTreeMap<String, Counts>>>,
}

impl<'a> Eval<'a> {
    fn new(embeddings: &'a Embeddings<VocabWrap, StorageViewWrap>) -> Self {
        Eval {
            embeddings,
            section_counts: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Evaluate an analogy.
    fn eval_analogy(&self, instance: &Instance) {
        // Skip instances where the to-be-predicted word is not in the
        // vocab. This is a shortcoming of the vocab size and not of the
        // embedding model itself.
        if self
            .embeddings
            .vocab()
            .idx(&instance.answer)
            .and_then(|idx| idx.word())
            .is_none()
        {
            let mut section_counts = self.section_counts.lock().unwrap();
            let counts = section_counts.entry(instance.section.clone()).or_default();
            counts.n_skipped += 1;
            return;
        }

        // If the model is not able to provide a query result, it is counted
        // as an error.
        let (is_correct, cos) = self
            .embeddings
            .analogy([&instance.query.0, &instance.query.1, &instance.query.2], 1)
            .map(|r| {
                let result = r.first().unwrap();
                (result.word() == instance.answer, result.cosine_similarity())
            })
            .unwrap_or((false, 0.));

        let mut section_counts = self.section_counts.lock().unwrap();
        let counts = section_counts.entry(instance.section.clone()).or_default();
        counts.n_instances += 1;
        if is_correct {
            counts.n_correct += 1;
        }
        counts.sum_cos += cos;
    }

    /// Print the accuracy for a section.
    fn print_section_accuracy(&self, section: &str, counts: &Counts) {
        if counts.n_instances == 0 {
            eprintln!("{}: no evaluation instances", section);
            return;
        }

        println!(
            "{}: {}/{} correct, accuracy: {:.2}, avg cos: {:1.2}, skipped: {}",
            section,
            counts.n_correct,
            counts.n_instances,
            (counts.n_correct as f64 / counts.n_instances as f64) * 100.,
            (counts.sum_cos / counts.n_instances as f32),
            counts.n_skipped,
        );
    }
}

impl<'a> Drop for Eval<'a> {
    fn drop(&mut self) {
        let section_counts = self.section_counts.lock().unwrap();

        // Print out counts for all sections.
        for (section, counts) in section_counts.iter() {
            self.print_section_accuracy(section, counts);
        }

        let n_correct = section_counts.values().map(|c| c.n_correct).sum::<usize>();
        let n_instances = section_counts
            .values()
            .map(|c| c.n_instances)
            .sum::<usize>();
        let n_skipped = section_counts.values().map(|c| c.n_skipped).sum::<usize>();
        let n_instances_with_skipped = n_instances + n_skipped;
        let cos = section_counts.values().map(|c| c.sum_cos).sum::<f32>();

        // Print out overall counts.
        println!(
            "Total: {}/{} correct, accuracy: {:.2}, avg cos: {:1.2}",
            n_correct,
            n_instances,
            (n_correct as f64 / n_instances as f64) * 100.,
            (cos / n_instances as f32)
        );

        // Print skip counts.
        println!(
            "Skipped: {}/{} ({}%)",
            n_skipped,
            n_instances_with_skipped,
            (n_skipped as f64 / n_instances_with_skipped as f64) * 100.
        );
    }
}

struct Instance {
    section: String,
    query: (String, String, String),
    answer: String,
}

fn read_analogies(reader: impl BufRead) -> Result<Vec<Instance>> {
    let mut section = String::new();

    let mut instances = Vec::new();

    for line in reader.lines() {
        let line = line.context("Cannot read line")?;

        if line.starts_with(": ") {
            section = line.chars().skip(2).collect::<String>();
            continue;
        }

        let quadruple: Vec<_> = line.split_whitespace().collect();

        instances.push(Instance {
            section: section.clone(),
            query: (
                quadruple[0].to_owned(),
                quadruple[1].to_owned(),
                quadruple[2].to_owned(),
            ),
            answer: quadruple[3].to_owned(),
        });
    }

    Ok(instances)
}

fn process_analogies(embeddings: &Embeddings<VocabWrap, StorageViewWrap>, instances: &[Instance]) {
    let pb = ProgressBar::new(instances.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar().template("{bar:30} {percent}% {msg} ETA: {eta_precise}"),
    );
    let eval = Eval::new(embeddings);
    instances.par_iter().enumerate().for_each(|(i, instance)| {
        if i % 50 == 0 {
            pb.inc(50);
        }
        eval.eval_analogy(instance)
    });
    pb.finish();
}
