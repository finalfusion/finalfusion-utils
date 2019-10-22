use std::io::stdout;

use clap::{App, AppSettings, Arg, Shell, SubCommand};

mod analogy;

mod compute_accuracy;

mod convert;

pub mod io;

mod metadata;

mod quantize;

mod similar;

mod traits;
pub use self::traits::FinalfusionApp;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
    AppSettings::SubcommandRequiredElseHelp,
];

fn main() {
    // Known subapplications.
    let apps = vec![
        analogy::AnalogyApp::app(),
        compute_accuracy::ComputeAccuracyApp::app(),
        convert::ConvertApp::app(),
        metadata::MetadataApp::app(),
        quantize::QuantizeApp::app(),
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
            analogy::AnalogyApp::parse(matches.subcommand_matches("analogy").unwrap()).run()
        }
        "completions" => {
            let shell = matches
                .subcommand_matches("completions")
                .unwrap()
                .value_of("shell")
                .unwrap();
            write_completion_script(cli, shell.parse::<Shell>().unwrap());
        }
        "compute-accuracy" => compute_accuracy::ComputeAccuracyApp::parse(
            matches.subcommand_matches("compute-accuracy").unwrap(),
        )
        .run(),
        "convert" => {
            convert::ConvertApp::parse(matches.subcommand_matches("convert").unwrap()).run()
        }
        "metadata" => {
            metadata::MetadataApp::parse(matches.subcommand_matches("metadata").unwrap()).run()
        }
        "quantize" => {
            quantize::QuantizeApp::parse(matches.subcommand_matches("quantize").unwrap()).run()
        }
        "similar" => {
            similar::SimilarApp::parse(matches.subcommand_matches("similar").unwrap()).run()
        }
        _unknown => unreachable!(),
    }
}

fn write_completion_script(mut cli: App, shell: Shell) {
    cli.gen_completions_to("finalfusion", shell, &mut stdout());
}