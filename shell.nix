with import <nixpkgs> {};

stdenv.mkDerivation rec {
  name = "finalfusion-utils-env";
  env = buildEnv { name = name; paths = buildInputs; };

  nativeBuildInputs = [
    latest.rustChannels.stable.rust
  ];

  buildInputs = [
    pkgconfig

    # OpenBLAS backend
    gfortran.cc.lib
    openblasCompat

    # MKL
    mkl
    openssl
    pkgconfig
  ];
}
