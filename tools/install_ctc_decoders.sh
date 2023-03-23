mkdir -p extras
cd ./extras || exit

# Install baidu's beamsearch_with_lm
if [ ! -d ctc_decoders ]; then
    git clone https://github.com/usimarit/ctc_decoders.git

    cd ./ctc_decoders || exit
    ./setup.sh

    cd ..
fi

cd ..