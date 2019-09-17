use clap::{App, AppSettings};

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

    let matches = App::new("finalfusion")
        .settings(DEFAULT_CLAP_SETTINGS)
        .subcommands(apps)
        .get_matches();

    match matches.subcommand_name().unwrap() {
        "analogy" => {
            analogy::AnalogyApp::parse(matches.subcommand_matches("analogy").unwrap()).run()
        }
        "compute-accuracy" => {
            analogy::AnalogyApp::parse(matches.subcommand_matches("compute-accuracy").unwrap())
                .run()
        }
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
