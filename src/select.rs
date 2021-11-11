use std::collections::HashSet;
use std::convert::TryFrom;
use std::io::BufRead;

use anyhow::{bail, Context, Error, Result};
use clap::{App, Arg, ArgMatches};
use finalfusion::embeddings::Embeddings;
use finalfusion::norms::NdNorms;
use finalfusion::storage::{NdArray, StorageWrap};
use finalfusion::vocab::{SimpleVocab, Vocab, VocabWrap};
use ndarray::{Array1, Array2};
use stdinout::Input;

use super::FinalfusionApp;
use crate::io::{read_embeddings, write_embeddings, EmbeddingFormat};

const IGNORE_UNKNOWN: &str = "IGNORE_UNKNOWN";
const INPUT_EMBEDDINGS: &str = "INPUT_EMBEDDINGS";
const INPUT_FORMAT: &str = "INPUT_FORMAT";
const OUTPUT_EMBEDDINGS: &str = "OUTPUT_EMBEDDINGS";
const OUTPUT_FORMAT: &str = "OUTPUT_FORMAT";
const SELECT: &str = "SELECT";

pub struct SelectApp {
    ignore_unknown: bool,
    input_filename: String,
    input_format: EmbeddingFormat,
    output_filename: String,
    output_format: EmbeddingFormat,
    select_input: Input,
}

impl FinalfusionApp for SelectApp {
    fn app() -> App<'static, 'static> {
        App::new("select")
            .about("Select embeddings from an embeddings file")
            .arg(
                Arg::with_name(IGNORE_UNKNOWN)
                    .short("i")
                    .long("ignore-unknown")
                    .help("Ignore words for which no embedding is available"),
            )
            .arg(
                Arg::with_name(INPUT_FORMAT)
                    .short("f")
                    .value_name("FORMAT")
                    .takes_value(true)
                    .possible_values(&[
                        "fasttext",
                        "finalfusion",
                        "finalfusion_mmap",
                        "floret",
                        "text",
                        "textdims",
                        "word2vec",
                    ])
                    .default_value("finalfusion")
                    .help("Input format"),
            )
            .arg(
                Arg::with_name(OUTPUT_FORMAT)
                    .short("t")
                    .value_name("FORMAT")
                    .takes_value(true)
                    .possible_values(&["finalfusion", "text", "textdims", "word2vec"])
                    .default_value("finalfusion")
                    .help("Output format"),
            )
            .arg(
                Arg::with_name(INPUT_EMBEDDINGS)
                    .help("Input embeddings")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(OUTPUT_EMBEDDINGS)
                    .help("Output embeddings")
                    .index(2)
                    .required(true),
            )
            .arg(Arg::with_name(SELECT).help("Words to select").index(3))
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let input_filename = matches.value_of(INPUT_EMBEDDINGS).unwrap().to_owned();
        let output_filename = matches.value_of(OUTPUT_EMBEDDINGS).unwrap().to_owned();
        let select_input = Input::from(matches.value_of("SELECT"));

        let ignore_unknown = matches.is_present(IGNORE_UNKNOWN);

        let input_format = matches
            .value_of(INPUT_FORMAT)
            .map(|f| {
                EmbeddingFormat::try_from(f)
                    .context(format!("Cannot parse embedding format: {}", f))
            })
            .transpose()?
            .unwrap();

        let output_format = matches
            .value_of(OUTPUT_FORMAT)
            .map(|f| {
                EmbeddingFormat::try_from(f)
                    .context(format!("Cannot parse embedding format: {}", f))
            })
            .transpose()?
            .unwrap();

        Ok(SelectApp {
            ignore_unknown,
            input_filename,
            input_format,
            output_filename,
            output_format,
            select_input,
        })
    }

    fn run(&self) -> Result<()> {
        let embeddings = read_embeddings(&self.input_filename, self.input_format)
            .context("Cannot read embeddings")?;

        let select = self.read_words(&embeddings)?;

        let output_embeddings = copy_select_embeddings(&embeddings, select)?;

        write_embeddings(
            &output_embeddings,
            &self.output_filename,
            self.output_format,
            true,
        )
    }
}

impl SelectApp {
    fn read_words(
        &self,
        embeddings: &Embeddings<VocabWrap, StorageWrap>,
    ) -> Result<HashSet<String>, Error> {
        let mut words = HashSet::new();

        for word in self
            .select_input
            .buf_read()
            .context("Cannot open selection file")?
            .lines()
        {
            let word = word?;

            match embeddings.vocab().idx(&word) {
                Some(_) => {
                    words.insert(word);
                }
                None => {
                    if !self.ignore_unknown {
                        bail!("Cannot get embedding for: {}", word)
                    }
                }
            };
        }

        Ok(words)
    }
}

fn copy_select_embeddings(
    embeddings: &Embeddings<VocabWrap, StorageWrap>,
    select: HashSet<String>,
) -> Result<Embeddings<VocabWrap, StorageWrap>> {
    let mut selected_vocab = Vec::new();
    let mut selected_storage = Array2::zeros((select.len(), embeddings.dims()));
    let mut selected_norms = Array1::zeros((select.len(),));

    for (idx, word) in select.into_iter().enumerate() {
        match embeddings.embedding_with_norm(&word) {
            Some(embed_with_norm) => {
                selected_storage
                    .row_mut(idx)
                    .assign(&embed_with_norm.embedding);
                selected_norms[idx] = embed_with_norm.norm;
            }
            None => bail!("Cannot get embedding for: {}", word),
        }

        selected_vocab.push(word);
    }

    Ok(Embeddings::new(
        None,
        SimpleVocab::new(selected_vocab),
        NdArray::from(selected_storage),
        NdNorms::new(selected_norms),
    )
    .into())
}
