#[cfg(feature = "netlib")]
extern crate netlib_src;

#[cfg(feature = "openblas")]
extern crate openblas_src;

#[cfg(feature = "intel-mkl")]
extern crate intel_mkl_src;

use std::io::stdout;

use anyhow::Result;
use clap::{App, AppSettings, Arg, Shell, SubCommand};

mod analogy;

mod compute_accuracy;

mod convert;

pub mod io;

mod metadata;

mod quantize;

mod reconstruct;

mod similar;

mod traits;
pub use self::traits::FinalfusionApp;

pub mod util;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
    AppSettings::SubcommandRequiredElseHelp,
];

fn main() -> Result<()> {
    // Known subapplications.
    let apps = vec![
        analogy::AnalogyApp::app(),
        compute_accuracy::ComputeAccuracyApp::app(),
        convert::ConvertApp::app(),
        metadata::MetadataApp::app(),
        quantize::QuantizeApp::app(),
        reconstruct::ReconstructApp::app(),
        similar::SimilarApp::app(),
    ];

    let cli = App::new("finalfusion")
        .settings(DEFAULT_CLAP_SETTINGS)
        .subcommands(apps)
        .subcommand(
            SubCommand::with_name("completions")
                .about("Generate completion scripts for your shell")
                .setting(AppSettings::ArgRequiredElseHelp)
                .arg(Arg::with_name("shell").possible_values(&Shell::variants())),
        );
    let matches = cli.clone().get_matches();

    match matches.subcommand_name().unwrap() {
        "analogy" => {
            analogy::AnalogyApp::parse(matches.subcommand_matches("analogy").unwrap())?.run()
        }
        "completions" => {
            let shell = matches
                .subcommand_matches("completions")
                .unwrap()
                .value_of("shell")
                .unwrap();
            write_completion_script(cli, shell.parse::<Shell>().unwrap());
            Ok(())
        }
        "compute-accuracy" => compute_accuracy::ComputeAccuracyApp::parse(
            matches.subcommand_matches("compute-accuracy").unwrap(),
        )?
        .run(),
        "convert" => {
            convert::ConvertApp::parse(matches.subcommand_matches("convert").unwrap())?.run()
        }
        "metadata" => {
            metadata::MetadataApp::parse(matches.subcommand_matches("metadata").unwrap())?.run()
        }
        "quantize" => {
            quantize::QuantizeApp::parse(matches.subcommand_matches("quantize").unwrap())?.run()
        }
        "reconstruct" => {
            reconstruct::ReconstructApp::parse(matches.subcommand_matches("reconstruct").unwrap())?
                .run()
        }
        "similar" => {
            similar::SimilarApp::parse(matches.subcommand_matches("similar").unwrap())?.run()
        }
        _unknown => unreachable!(),
    }
}

fn write_completion_script(mut cli: App, shell: Shell) {
    cli.gen_completions_to("finalfusion", shell, &mut stdout());
}
