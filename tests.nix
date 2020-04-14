{ pkgs ? import (import ./nix/sources.nix).nixpkgs {} }:

let
  finalfrontier = import ./default.nix {
    inherit pkgs;
  };
  finalfrontierOpenblas = import ./default.nix {
    inherit pkgs;

    withOpenblas = true;
  };
in [
  finalfrontier
  finalfrontierOpenblas
]
