{ pkgs ? import (import ./nix/sources.nix).nixpkgs {}

# Build finalfusion-utils with MKL or OpenBLAS support.
, withMkl ? false
, withOpenblas ? false
}:

assert !(withMkl && withOpenblas);

let
  sources = import ./nix/sources.nix;
  mkl = pkgs.callPackage nix/mkl.nix {};
  crateTools = pkgs.callPackage "${sources.crate2nix}/tools.nix" {};
  cargoNix = pkgs.callPackage (crateTools.generatedCargoNix {
    name = "finalfrontier";
    src = pkgs.nix-gitignore.gitignoreSource [ ".git/" "nix/" "*.nix" ] ./.;
  }) {
    inherit buildRustCrate;

    rootFeatures = with pkgs; [ "default" ]
      ++ lib.optional withOpenblas "openblas"
      ++ lib.optional withMkl "intel-mkl" ;
  };
  crateOverrides = with pkgs; defaultCrateOverrides // {
    finalfusion-utils = attr: rec {
      pname = "finalfusion-utils";
      name = "${pname}-${attr.version}";

      nativeBuildInputs = [ installShellFiles ];

      buildInputs = lib.optionals withOpenblas [ gfortran.cc.lib openblasCompat ]
        ++ stdenv.lib.optional stdenv.isDarwin darwin.Security;

      postBuild = ''
        for shell in bash fish zsh; do
          target/bin/finalfusion completions $shell > finalfusion.$shell
        done
      '';

      postInstall = ''
        # Install shell completions
        installShellCompletion finalfusion.{bash,fish,zsh}
      '';

      meta = with stdenv.lib; {
        description = "Utilities for finalfusion word embeddings";
        license = licenses.asl20;
        maintainers = with maintainers; [ danieldk ];
        platforms = platforms.all;
      };
    };

    intel-mkl-src = attr: rec {
      nativeBuildInputs = [ pkgconfig ];

      buildInputs = [ mkl ];
    };

    openblas-src = attr: {
      nativeBuildInputs = [ perl ];

      preConfigure = ''
        export CARGO_FEATURE_SYSTEM=1;
      '';
    };
  };
  buildRustCrate = pkgs.buildRustCrate.override {
    defaultCrateOverrides = crateOverrides;
  };
in cargoNix.rootCrate.build
