use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use clap::{App, Arg, ArgMatches};
use finalfusion::io::ReadMetadata;
use finalfusion::metadata::Metadata;
use stdinout::{OrExit, Output};
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

    fn parse(matches: &ArgMatches) -> Self {
        let input_filename = matches.value_of(INPUT).unwrap().to_owned();
        let output_filename = matches.value_of(OUTPUT).map(ToOwned::to_owned);

        MetadataApp {
            input_filename,
            output_filename,
        }
    }

    fn run(&self) {
        let output = Output::from(self.output_filename.as_ref());
        let mut writer =
            BufWriter::new(output.write().or_exit("Cannot open output for writing", 1));

        if let Some(metadata) = read_metadata(&self.input_filename) {
            writer
                .write_all(
                    to_string_pretty(&*metadata)
                        .or_exit("Cannot serialize metadata to TOML", 1)
                        .as_bytes(),
                )
                .or_exit("Cannot write metadata", 1);
        }
    }
}

fn read_metadata(filename: &str) -> Option<Metadata> {
    let f = File::open(filename).or_exit("Cannot open embeddings file", 1);
    let mut reader = BufReader::new(f);
    ReadMetadata::read_metadata(&mut reader).or_exit("Cannot read metadata", 1)
}
