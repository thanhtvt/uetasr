mkdir -p extras
cd ./extras || exit

if [ ! -d g711-python ]; then
    git clone https://github.com/thanhtvt/g711-python.git

    cd ./g711-python || exit
    pip install -e .

    cd ../..
fi
