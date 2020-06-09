use std::convert::TryFrom;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use anyhow::{anyhow, bail, Context, Error, Result};

use finalfusion::compat::text::{WriteText, WriteTextDims};
use finalfusion::compat::word2vec::WriteWord2Vec;
use finalfusion::io::WriteEmbeddings;
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

impl TryFrom<&str> for EmbeddingFormat {
    type Error = Error;

    fn try_from(format: &str) -> Result<Self> {
        use self::EmbeddingFormat::*;

        match format {
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

pub fn read_embeddings(
    filename: &str,
    embedding_format: EmbeddingFormat,
) -> Result<Embeddings<VocabWrap, StorageWrap>> {
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

pub fn write_embeddings(
    embeddings: &Embeddings<VocabWrap, StorageWrap>,
    filename: &str,
    format: EmbeddingFormat,
    unnormalize: bool,
) -> Result<()> {
    let f =
        File::create(filename).context(format!("Cannot create embeddings file: {}", filename))?;
    let mut writer = BufWriter::new(f);

    use self::EmbeddingFormat::*;
    match format {
        FastText => bail!("Writing to the fastText format is not supported"),
        FinalFusion => embeddings.write_embeddings(&mut writer)?,
        FinalFusionMmap => bail!("Writing to memory-mapped finalfusion file is not supported"),
        Word2Vec => embeddings.write_word2vec_binary(&mut writer, unnormalize)?,
        Text => embeddings.write_text(&mut writer, unnormalize)?,
        TextDims => embeddings.write_text_dims(&mut writer, unnormalize)?,
    };

    Ok(())
}
