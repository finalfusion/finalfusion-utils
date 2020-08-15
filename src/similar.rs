use std::convert::TryFrom;
use std::io::BufRead;

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use finalfusion::similarity::WordSimilarity;
use stdinout::Input;

use super::FinalfusionApp;
use crate::io::{read_embeddings_view, EmbeddingFormat};
use crate::similarity::SimilarityMeasure;

pub struct SimilarApp {
    embeddings_filename: String,
    embedding_format: EmbeddingFormat,
    input: Option<String>,
    k: usize,
    similarity: SimilarityMeasure,
}

impl FinalfusionApp for SimilarApp {
    fn app() -> App<'static, 'static> {
        App::new("similar")
            .about("Find words that are similar to a given word")
            .arg(
                Arg::with_name("format")
                    .short("f")
                    .value_name("FORMAT")
                    .takes_value(true)
                    .possible_values(&[
                        "fasttext",
                        "finalfusion",
                        "finalfusion_mmap",
                        "text",
                        "textdims",
                        "word2vec",
                    ])
                    .default_value("finalfusion"),
            )
            .arg(
                Arg::with_name("neighbors")
                    .short("k")
                    .value_name("K")
                    .help("Return K nearest neighbors")
                    .takes_value(true)
                    .default_value("10"),
            )
            .arg(SimilarityMeasure::new_clap_arg())
            .arg(
                Arg::with_name("EMBEDDINGS")
                    .help("Embeddings file")
                    .index(1)
                    .required(true),
            )
            .arg(Arg::with_name("INPUT").help("Input words").index(2))
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let input = matches.value_of("INPUT").map(ToOwned::to_owned);

        let embeddings_filename = matches.value_of("EMBEDDINGS").unwrap().to_owned();

        let embedding_format = matches
            .value_of("format")
            .map(|f| {
                EmbeddingFormat::try_from(f)
                    .context(format!("Cannot parse embedding format: {}", f))
            })
            .transpose()?
            .unwrap();

        let k = matches
            .value_of("neighbors")
            .map(|k| {
                k.parse()
                    .context(format!("Cannot parse number of neighbors: {}", k))
            })
            .transpose()?
            .unwrap();

        let similarity = SimilarityMeasure::parse_clap_matches(&matches)?;

        Ok(SimilarApp {
            similarity,
            input,
            embeddings_filename,
            embedding_format,
            k,
        })
    }

    fn run(&self) -> Result<()> {
        let embeddings = read_embeddings_view(&self.embeddings_filename, self.embedding_format)
            .context("Cannot read embeddings")?;

        let input = Input::from(self.input.as_ref());
        let reader = input.buf_read().context("Cannot open input for reading")?;

        for line in reader.lines() {
            let line = line.context("Cannot read line")?.trim().to_owned();
            if line.is_empty() {
                continue;
            }

            let results = match embeddings.word_similarity(&line, self.k) {
                Some(results) => results,
                None => {
                    eprintln!("Could not compute embedding for: {}", line);
                    continue;
                }
            };

            for similar in results {
                println!("{}\t{}", similar.word(), self.similarity.to_f32(&similar));
            }
        }

        Ok(())
    }
}
