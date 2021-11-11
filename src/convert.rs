use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufReader, Read};

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use finalfusion::compat::floret::ReadFloretText;
use finalfusion::io::ReadEmbeddings;
use finalfusion::metadata::Metadata;
use finalfusion::prelude::*;
use toml::Value;

use crate::io::{write_embeddings, EmbeddingFormat};
use crate::FinalfusionApp;

// Option constants
static INPUT_FORMAT: &str = "input_format";
static LOSSY: &str = "lossy";
static METADATA_FILENAME: &str = "metadata_filename";
static OUTPUT_FORMAT: &str = "output_format";
static UNNORMALIZE: &str = "unnormalize";

// Argument constants
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

pub struct ConvertApp {
    input_filename: String,
    output_filename: String,
    metadata_filename: Option<String>,
    input_format: EmbeddingFormat,
    output_format: EmbeddingFormat,
    lossy: bool,
    unnormalize: bool,
}

impl FinalfusionApp for ConvertApp {
    fn app() -> App<'static, 'static> {
        App::new("convert")
            .about("Convert between embedding formats")
            .arg(
                Arg::with_name(INPUT)
                    .help("finalfusion model")
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
                Arg::with_name(INPUT_FORMAT)
                    .short("f")
                    .long("from")
                    .value_name("FORMAT")
                    .takes_value(true)
                    .possible_values(&["fasttext", "finalfusion", "text", "textdims", "word2vec"])
                    .default_value("word2vec"),
            )
            .arg(
                Arg::with_name(LOSSY)
                    .long("lossy")
                    .help("do not fail on malformed UTF-8 byte sequences")
                    .takes_value(false),
            )
            .arg(
                Arg::with_name(METADATA_FILENAME)
                    .short("m")
                    .long("metadata")
                    .value_name("FILENAME")
                    .help("TOML metadata add to the embeddings")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(OUTPUT_FORMAT)
                    .short("t")
                    .long("to")
                    .value_name("FORMAT")
                    .takes_value(true)
                    .possible_values(&["finalfusion", "text", "textdims", "word2vec"])
                    .default_value("finalfusion"),
            )
            .arg(
                Arg::with_name(UNNORMALIZE)
                    .short("u")
                    .long("unnormalize")
                    .help("unnormalize embeddings (does not affect finalfusion format)")
                    .takes_value(false),
            )
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let input_filename = matches.value_of(INPUT).unwrap().to_owned();
        let input_format = matches
            .value_of(INPUT_FORMAT)
            .map(|v| {
                EmbeddingFormat::try_from(v).context(format!("Cannot parse input format: {}", v))
            })
            .transpose()?
            .unwrap();
        let output_filename = matches.value_of(OUTPUT).unwrap().to_owned();
        let output_format = matches
            .value_of(OUTPUT_FORMAT)
            .map(|v| {
                EmbeddingFormat::try_from(v).context(format!("Cannot parse output format: {}", v))
            })
            .transpose()?
            .unwrap();

        let metadata_filename = matches.value_of(METADATA_FILENAME).map(ToOwned::to_owned);

        Ok(ConvertApp {
            input_filename,
            output_filename,
            input_format,
            output_format,
            metadata_filename,
            lossy: matches.is_present(LOSSY),
            unnormalize: matches.is_present(UNNORMALIZE),
        })
    }

    fn run(&self) -> Result<()> {
        let metadata = self
            .metadata_filename
            .as_ref()
            .map(read_metadata)
            .transpose()?
            .map(Metadata::new);

        let mut embeddings = read_embeddings(&self.input_filename, self.input_format, self.lossy)?;

        // Overwrite metadata if provided, otherwise retain existing metadata.
        if metadata.is_some() {
            embeddings.set_metadata(metadata);
        }

        write_embeddings(
            &embeddings,
            &self.output_filename,
            self.output_format,
            self.unnormalize,
        )
        .context("Cannot write embeddings")
    }
}

fn read_metadata(filename: impl AsRef<str>) -> Result<Value> {
    let f = File::open(filename.as_ref())
        .context(format!("Cannot open metadata file: {}", filename.as_ref()))?;
    let mut reader = BufReader::new(f);
    let mut buf = String::new();
    reader
        .read_to_string(&mut buf)
        .context(format!("Cannot read metadata from {}", filename.as_ref()))?;
    buf.parse::<Value>().context(format!(
        "Cannot parse metadata TOML from {}",
        filename.as_ref()
    ))
}

fn read_embeddings(
    filename: &str,
    embedding_format: EmbeddingFormat,
    lossy: bool,
) -> Result<Embeddings<VocabWrap, StorageWrap>> {
    let f = File::open(filename).context(format!("Cannot open embeddings file: {}", filename))?;
    let mut reader = BufReader::new(f);

    use self::EmbeddingFormat::*;
    match (embedding_format, lossy) {
        (FastText, true) => ReadFastText::read_fasttext_lossy(&mut reader).map(Embeddings::into),
        (FastText, false) => ReadFastText::read_fasttext(&mut reader).map(Embeddings::into),
        (FinalFusion, _) => ReadEmbeddings::read_embeddings(&mut reader),
        (FinalFusionMmap, _) => MmapEmbeddings::mmap_embeddings(&mut reader),
        (Floret, _) => ReadFloretText::read_floret_text(&mut reader).map(Embeddings::into),
        (Word2Vec, true) => {
            ReadWord2Vec::read_word2vec_binary_lossy(&mut reader).map(Embeddings::into)
        }
        (Word2Vec, false) => ReadWord2Vec::read_word2vec_binary(&mut reader).map(Embeddings::into),
        (Text, true) => ReadText::read_text_lossy(&mut reader).map(Embeddings::into),
        (Text, false) => ReadText::read_text(&mut reader).map(Embeddings::into),
        (TextDims, true) => ReadTextDims::read_text_dims_lossy(&mut reader).map(Embeddings::into),
        (TextDims, false) => ReadTextDims::read_text_dims(&mut reader).map(Embeddings::into),
    }
    .context(format!(
        "Cannot read {} embeddings from {}",
        embedding_format, filename
    ))
}
