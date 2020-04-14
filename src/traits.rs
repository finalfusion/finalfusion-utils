use anyhow::Result;
use clap::{App, ArgMatches};

pub trait FinalfusionApp
where
    Self: Sized,
{
    fn app() -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Result<Self>;

    fn run(&self) -> Result<()>;
}
