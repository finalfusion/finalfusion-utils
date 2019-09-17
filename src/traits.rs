use clap::{App, ArgMatches};

pub trait FinalfusionApp {
    fn app() -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Self;

    fn run(&self);
}
