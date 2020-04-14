use std::fmt;
use std::fs::File;
use std::io::BufReader;

use anyhow::{anyhow, Context, Result};

use finalfusion::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingFormat {
    FastText,
    FinalFusion,
    FinalFusionMmap,
    Word2Vec,
    Text,
    TextDims,
}

impl EmbeddingFormat {
    pub fn try_from(format: impl AsRef<str>) -> Result<Self> {
        use self::EmbeddingFormat::*;

        match format.as_ref() {
            "fasttext" => Ok(FastText),
            "finalfusion" => Ok(FinalFusion),
            "finalfusion_mmap" => Ok(FinalFusionMmap),
            "word2vec" => Ok(Word2Vec),
            "text" => Ok(Text),
            "textdims" => Ok(TextDims),
            unknown => Err(anyhow!("Unknown embedding format: {}", unknown)),
        }
    }
}

impl fmt::Display for EmbeddingFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use EmbeddingFormat::*;
        let s = match self {
            FastText => "fasttext",
            FinalFusion => "finalfusion",
            FinalFusionMmap => "finalfusion_mmap",
            Word2Vec => "word2vec",
            Text => "text",
            TextDims => "textdims",
        };

        f.write_str(s)
    }
}

pub fn read_embeddings_view(
    filename: &str,
    embedding_format: EmbeddingFormat,
) -> Result<Embeddings<VocabWrap, StorageViewWrap>> {
    let f = File::open(filename).context("Cannot open embeddings file")?;
    let mut reader = BufReader::new(f);

    use self::EmbeddingFormat::*;
    let embeds = match embedding_format {
        FastText => ReadFastText::read_fasttext(&mut reader).map(Embeddings::into),
        FinalFusion => ReadEmbeddings::read_embeddings(&mut reader),
        FinalFusionMmap => MmapEmbeddings::mmap_embeddings(&mut reader),
        Word2Vec => ReadWord2Vec::read_word2vec_binary(&mut reader).map(Embeddings::into),
        Text => ReadText::read_text(&mut reader).map(Embeddings::into),
        TextDims => ReadTextDims::read_text_dims(&mut reader).map(Embeddings::into),
    };

    Ok(embeds?)
}
