use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use finalfusion::io::ReadMetadata;
use finalfusion::metadata::Metadata;
use stdinout::Output;
use toml::ser::to_string_pretty;

use crate::FinalfusionApp;

// Argument constants
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

pub struct MetadataApp {
    input_filename: String,
    output_filename: Option<String>,
}

impl FinalfusionApp for MetadataApp {
    fn app() -> App<'static, 'static> {
        App::new("metadata")
            .about("Extract metadata from finalfusion embeddings")
            .arg(
                Arg::with_name(INPUT)
                    .help("finalfusion model")
                    .index(1)
                    .required(true),
            )
            .arg(Arg::with_name(OUTPUT).help("Output file").index(2))
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let input_filename = matches.value_of(INPUT).unwrap().to_owned();
        let output_filename = matches.value_of(OUTPUT).map(ToOwned::to_owned);

        Ok(MetadataApp {
            input_filename,
            output_filename,
        })
    }

    fn run(&self) -> Result<()> {
        let output = Output::from(self.output_filename.as_ref());
        let mut writer = BufWriter::new(output.write().context("Cannot open output for writing")?);

        if let Some(metadata) = read_metadata(&self.input_filename)? {
            writer
                .write_all(
                    to_string_pretty(&*metadata)
                        .context("Cannot serialize metadata to TOML")?
                        .as_bytes(),
                )
                .context("Cannot write metadata")?;
        }

        Ok(())
    }
}

fn read_metadata(filename: &str) -> Result<Option<Metadata>> {
    let f = File::open(filename).context(format!("Cannot open embeddings file: {}", filename))?;
    let mut reader = BufReader::new(f);
    ReadMetadata::read_metadata(&mut reader)
        .context(format!("Cannot read metadata from {}", filename))
}
