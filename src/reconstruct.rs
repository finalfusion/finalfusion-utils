use std::fs::File;
use std::io::{BufReader, BufWriter};

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use finalfusion::io::WriteEmbeddings;
use finalfusion::norms::NdNorms;
use finalfusion::prelude::*;
use finalfusion::storage::Reconstruct;
use finalfusion::storage::{NdArray, QuantizedArray};
use finalfusion::vocab::Vocab;
use ndarray::{s, Array2};

use crate::util::l2_normalize_array;
use crate::FinalfusionApp;

// Argument constants
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

pub struct ReconstructApp {
    input_filename: String,
    output_filename: String,
}

impl FinalfusionApp for ReconstructApp {
    fn app() -> App<'static, 'static> {
        App::new("reconstruct")
            .about("Reconstruct quantized embedding matrices")
            .arg(
                Arg::with_name(INPUT)
                    .help("quantized finalfusion embeddings")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(OUTPUT)
                    .help("reconstructed finalfusion embeddings")
                    .index(2)
                    .required(true),
            )
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        // Arguments
        let input_filename = matches.value_of(INPUT).unwrap().to_owned();
        let output_filename = matches.value_of(OUTPUT).unwrap().to_owned();

        Ok(ReconstructApp {
            input_filename,
            output_filename,
        })
    }

    fn run(&self) -> Result<()> {
        let f = File::open(&self.input_filename).context("Cannot open embeddings file")?;
        let mut reader = BufReader::new(f);
        let embeddings: Embeddings<VocabWrap, QuantizedArray> =
            Embeddings::read_embeddings(&mut reader)
                .context("Cannot read quantized embedding matrix")?;

        let (metadata, vocab, quantized_storage, norms) = embeddings.into_parts();

        let mut array: Array2<f32> = quantized_storage.reconstruct().into();

        let norms = match norms {
            Some(norms) => norms,
            None => NdNorms::new(l2_normalize_array(
                array.view_mut().slice_mut(s![0..vocab.words_len(), ..]),
            )),
        };

        let embeddings = Embeddings::new(metadata, vocab, array.into(), norms);

        write_embeddings(&embeddings, &self.output_filename)
    }
}

fn write_embeddings(embeddings: &Embeddings<VocabWrap, NdArray>, filename: &str) -> Result<()> {
    let f =
        File::create(filename).context(format!("Cannot create embeddings file: {}", filename))?;
    let mut writer = BufWriter::new(f);
    embeddings
        .write_embeddings(&mut writer)
        .context("Cannot write embeddings")
}
