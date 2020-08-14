use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use finalfusion::compat::fasttext::ReadFastText;
use finalfusion::io::WriteEmbeddings;
use finalfusion::prelude::*;

use crate::io::EmbeddingFormat;
use crate::FinalfusionApp;

// Argument constants
static FORMAT: &str = "FORMAT";
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

pub struct BucketToExplicitApp {
    input_filename: String,
    output_filename: String,
    format: EmbeddingFormat,
}

impl FinalfusionApp for BucketToExplicitApp {
    fn app() -> App<'static, 'static> {
        App::new("bucket-to-explicit")
            .about("Convert embeddings with bucket-vocab to explicit vocab.")
            .arg(
                Arg::with_name(INPUT)
                    .help("Input file")
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
                Arg::with_name(FORMAT)
                    .help("File format")
                    .short("f")
                    .long("format")
                    .possible_values(&["finalfusion", "fasttext"])
                    .default_value("finalfusion")
                    .takes_value(true)
                    .value_name("FORMAT"),
            )
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let input_filename = matches.value_of(INPUT).unwrap().to_owned();
        let output_filename = matches.value_of(OUTPUT).unwrap().to_owned();
        let format = matches
            .value_of(FORMAT)
            .map(|v| {
                EmbeddingFormat::try_from(v).context(format!("Cannot parse input format: {}", v))
            })
            .transpose()?
            .unwrap();

        Ok(BucketToExplicitApp {
            input_filename,
            output_filename,
            format,
        })
    }

    fn run(&self) -> Result<()> {
        let f = File::open(&self.input_filename)
            .context(format!("Cannot open input file: {}", self.input_filename))?;
        let mut reader = BufReader::new(f);
        let f = File::create(&self.output_filename).context(format!(
            "Cannot create embeddings file for writing: {}",
            self.output_filename
        ))?;
        let embeddings = match self.format {
            EmbeddingFormat::FinalFusion => {
                Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader).context(
                    "Cannot read input embeddings. \
                    Only finalfusion and fastText files can be converted.",
                )?
            }
            EmbeddingFormat::FastText => Embeddings::read_fasttext(&mut reader)
                .context(
                    "Cannot read input embeddings. \
                    Only finalfusion and fastText files can be converted.",
                )?
                .into(),
            _ => unreachable!(),
        };
        let conv = embeddings.try_to_explicit()?;

        let mut writer = BufWriter::new(f);
        conv.write_embeddings(&mut writer)
            .context("Cannot write embeddings")
    }
}
