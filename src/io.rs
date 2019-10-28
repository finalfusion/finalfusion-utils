use std::fs::File;
use std::io::BufReader;

use failure::{format_err, Error, ResultExt};

use finalfusion::prelude::*;
use finalfusion::storage::MmapQuantizedArray;
use finalfusion::storage::QuantizedArray;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingFormat {
    FastText,
    FinalFusion,
    FinalFusionMmap,
    Word2Vec,
    Text,
    TextDims,
}

#[derive(Clone, Copy)]
pub enum QuantizedEmbeddingFormat {
    FinalFusion,
    FinalFusionMmap,
}

impl EmbeddingFormat {
    pub fn try_from(format: impl AsRef<str>) -> Result<Self, Error> {
        use self::EmbeddingFormat::*;

        match format.as_ref() {
            "fasttext" => Ok(FastText),
            "finalfusion" => Ok(FinalFusion),
            "finalfusion_mmap" => Ok(FinalFusionMmap),
            "word2vec" => Ok(Word2Vec),
            "text" => Ok(Text),
            "textdims" => Ok(TextDims),
            unknown => Err(format_err!("Unknown embedding format: {}", unknown)),
        }
    }
}

impl QuantizedEmbeddingFormat {
    pub fn try_from(format: impl AsRef<str>) -> Result<Self, Error> {
        use self::QuantizedEmbeddingFormat::*;

        match format.as_ref() {
            "finalfusion" => Ok(FinalFusion),
            "finalfusion_mmap" => Ok(FinalFusionMmap),
            unknown => Err(format_err!("Unknown embedding format: {}", unknown)),
        }
    }
}

pub fn read_embeddings_view(
    filename: &str,
    embedding_format: EmbeddingFormat,
) -> Result<Embeddings<VocabWrap, StorageViewWrap>, Error> {
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

pub fn read_quantized_embeddings(
    filename: &str,
    embedding_format: QuantizedEmbeddingFormat,
) -> Result<Embeddings<VocabWrap, StorageWrap>, Error> {
    let f = File::open(filename).context("Cannot open embeddings file")?;
    let mut reader = BufReader::new(f);

    use self::QuantizedEmbeddingFormat::*;
    let embeds: Embeddings<VocabWrap, StorageWrap> = match embedding_format {
        FinalFusion => {
            let quantized_embeds: Embeddings<VocabWrap, QuantizedArray> =
                ReadEmbeddings::read_embeddings(&mut reader)?;
            quantized_embeds.into()
        }
        FinalFusionMmap => {
            let quantized_embeds: Embeddings<VocabWrap, MmapQuantizedArray> =
                MmapEmbeddings::mmap_embeddings(&mut reader)?;
            quantized_embeds.into()
        }
    };

    Ok(embeds)
}
