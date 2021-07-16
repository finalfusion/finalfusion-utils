use std::convert::TryFrom;
use std::fmt;

use anyhow::{anyhow, Context, Error, Result};
use clap::{Arg, ArgMatches};
use finalfusion::similarity::WordSimilarityResult;

const SIMILARITY: &str = "similarity";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimilarityMeasure {
    Angular,
    Cosine,
}

impl SimilarityMeasure {
    pub fn new_clap_arg() -> Arg<'static, 'static> {
        Arg::with_name(SIMILARITY)
            .short("s")
            .long("similarity")
            .value_name("SIMILARITY")
            .takes_value(true)
            .default_value("cosine")
            .possible_values(&["angular", "cosine"])
            .help("Similarity measure")
    }

    pub fn parse_clap_matches(matches: &ArgMatches) -> Result<Self> {
        let measure = matches
            .value_of(SIMILARITY)
            .map(|s| {
                SimilarityMeasure::try_from(s)
                    .context(format!("Cannot parse similarity measure: {}", s))
            })
            .transpose()?
            .unwrap();
        Ok(measure)
    }

    pub fn as_f32(&self, result: &WordSimilarityResult) -> f32 {
        use self::SimilarityMeasure::*;
        match self {
            Angular => result.angular_similarity(),
            Cosine => result.cosine_similarity(),
        }
    }
}

impl TryFrom<&str> for SimilarityMeasure {
    type Error = Error;

    fn try_from(format: &str) -> Result<Self> {
        use self::SimilarityMeasure::*;

        match format {
            "angular" => Ok(Angular),
            "cosine" => Ok(Cosine),
            unknown => Err(anyhow!("Unknown similarity measure: {}", unknown)),
        }
    }
}

impl fmt::Display for SimilarityMeasure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SimilarityMeasure::*;
        let s = match self {
            Angular => "angular",
            Cosine => "cosine",
        };

        f.write_str(s)
    }
}
