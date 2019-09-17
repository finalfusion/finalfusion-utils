use std::io::BufRead;

use clap::{App, Arg, ArgMatches};
use finalfusion::similarity::WordSimilarity;
use stdinout::{Input, OrExit};

use super::FinalfusionApp;
use crate::io::{read_embeddings_view, EmbeddingFormat};

pub struct SimilarApp {
    embeddings_filename: String,
    embedding_format: EmbeddingFormat,
    input: Option<String>,
    k: usize,
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
            .arg(
                Arg::with_name("EMBEDDINGS")
                    .help("Embeddings file")
                    .index(1)
                    .required(true),
            )
            .arg(Arg::with_name("INPUT").help("Input words").index(2))
    }

    fn parse(matches: &ArgMatches) -> Self {
        let input = matches.value_of("INPUT").map(ToOwned::to_owned);

        let embeddings_filename = matches.value_of("EMBEDDINGS").unwrap().to_owned();

        let embedding_format = matches
            .value_of("format")
            .map(|f| EmbeddingFormat::try_from(f).or_exit("Cannot parse embedding format", 1))
            .unwrap();

        let k = matches
            .value_of("neighbors")
            .map(|v| v.parse().or_exit("Cannot parse k", 1))
            .unwrap();

        SimilarApp {
            input,
            embeddings_filename,
            embedding_format,
            k,
        }
    }

    fn run(&self) {
        let embeddings = read_embeddings_view(&self.embeddings_filename, self.embedding_format)
            .or_exit("Cannot read embeddings", 1);

        let input = Input::from(self.input.as_ref());
        let reader = input.buf_read().or_exit("Cannot open input for reading", 1);

        for line in reader.lines() {
            let line = line.or_exit("Cannot read line", 1).trim().to_owned();
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
                println!("{}\t{}", similar.word, similar.similarity);
            }
        }
    }
}
