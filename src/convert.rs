use std::fs::File;
use std::io::{BufReader, BufWriter, Read};

use clap::{App, Arg, ArgMatches};
use failure::{err_msg, Fallible};
use finalfusion::compat::text::{WriteText, WriteTextDims};
use finalfusion::compat::word2vec::WriteWord2Vec;
use finalfusion::io::{ReadEmbeddings, WriteEmbeddings};
use finalfusion::metadata::Metadata;
use finalfusion::prelude::*;
use stdinout::OrExit;
use toml::Value;

use crate::io::EmbeddingFormat;
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

    fn parse(matches: &ArgMatches) -> Self {
        let input_filename = matches.value_of(INPUT).unwrap().to_owned();
        let input_format = matches
            .value_of(INPUT_FORMAT)
            .map(|v| EmbeddingFormat::try_from(v).or_exit("Cannot parse input format", 1))
            .unwrap();
        let output_filename = matches.value_of(OUTPUT).unwrap().to_owned();
        let output_format = matches
            .value_of(OUTPUT_FORMAT)
            .map(|v| EmbeddingFormat::try_from(v).or_exit("Cannot parse output format", 1))
            .unwrap();

        let metadata_filename = matches.value_of(METADATA_FILENAME).map(ToOwned::to_owned);

        ConvertApp {
            input_filename,
            output_filename,
            input_format,
            output_format,
            metadata_filename,
            lossy: matches.is_present(LOSSY),
            unnormalize: matches.is_present(UNNORMALIZE),
        }
    }

    fn run(&self) {
        let metadata = self
            .metadata_filename
            .as_ref()
            .map(read_metadata)
            .map(Metadata::new);

        let mut embeddings = read_embeddings(&self.input_filename, self.input_format, self.lossy);

        // Overwrite metadata if provided, otherwise retain existing metadata.
        if metadata.is_some() {
            embeddings.set_metadata(metadata);
        }

        write_embeddings(&embeddings, &self).or_exit("Cannot write embeddings", 1)
    }
}

fn read_metadata(filename: impl AsRef<str>) -> Value {
    let f = File::open(filename.as_ref()).or_exit("Cannot open metadata file", 1);
    let mut reader = BufReader::new(f);
    let mut buf = String::new();
    reader
        .read_to_string(&mut buf)
        .or_exit("Cannot read metadata", 1);
    buf.parse::<Value>()
        .or_exit("Cannot parse metadata TOML", 1)
}

fn read_embeddings(
    filename: &str,
    embedding_format: EmbeddingFormat,
    lossy: bool,
) -> Embeddings<VocabWrap, StorageWrap> {
    let f = File::open(filename).or_exit("Cannot open embeddings file", 1);
    let mut reader = BufReader::new(f);

    use self::EmbeddingFormat::*;
    match (embedding_format, lossy) {
        (FastText, true) => ReadFastText::read_fasttext_lossy(&mut reader).map(Embeddings::into),
        (FastText, false) => ReadFastText::read_fasttext(&mut reader).map(Embeddings::into),
        (FinalFusion, _) => ReadEmbeddings::read_embeddings(&mut reader),
        (FinalFusionMmap, _) => MmapEmbeddings::mmap_embeddings(&mut reader),
        (Word2Vec, true) => {
            ReadWord2Vec::read_word2vec_binary_lossy(&mut reader).map(Embeddings::into)
        }
        (Word2Vec, false) => ReadWord2Vec::read_word2vec_binary(&mut reader).map(Embeddings::into),
        (Text, true) => ReadText::read_text_lossy(&mut reader).map(Embeddings::into),
        (Text, false) => ReadText::read_text(&mut reader).map(Embeddings::into),
        (TextDims, true) => ReadTextDims::read_text_dims_lossy(&mut reader).map(Embeddings::into),
        (TextDims, false) => ReadTextDims::read_text_dims(&mut reader).map(Embeddings::into),
    }
    .or_exit("Cannot read embeddings", 1)
}

fn write_embeddings(
    embeddings: &Embeddings<VocabWrap, StorageWrap>,
    config: &ConvertApp,
) -> Fallible<()> {
    let f = File::create(&config.output_filename).or_exit("Cannot create embeddings file", 1);
    let mut writer = BufWriter::new(f);

    use self::EmbeddingFormat::*;
    match config.output_format {
        FastText => return Err(err_msg("Writing to the fastText format is not supported")),
        FinalFusion => embeddings.write_embeddings(&mut writer)?,
        FinalFusionMmap => return Err(err_msg("Writing to this format is not supported")),
        Word2Vec => embeddings.write_word2vec_binary(&mut writer, config.unnormalize)?,
        Text => embeddings.write_text(&mut writer, config.unnormalize)?,
        TextDims => embeddings.write_text_dims(&mut writer, config.unnormalize)?,
    };

    Ok(())
}
