with import <nixpkgs> {};

mkShell {
  nativeBuildInputs = [
    cargo
    clippy
  ];

  buildInputs = [
    pkgconfig

    # OpenBLAS backend
    gfortran.cc.lib
    openblasCompat
  ] ++ lib.optional stdenv.isDarwin darwin.Security;
}
