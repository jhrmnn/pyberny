wget $MOPAC_DOWNLOAD_URL $MOPAC_PASSWORD_URL
unzip MOPAC2016_for_Linux_64_bit.zip
chmod +x MOPAC2016.exe
export PATH="$PATH:$PWD/.travis"
export MOPACDIR="$PWD"
