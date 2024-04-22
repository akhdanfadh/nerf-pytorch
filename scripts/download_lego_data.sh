echo "-> Creating directory data/nerf_synthetic"
mkdir -p data/nerf_synthetic

echo "-> Downloading lego.zip"
python scripts/download_gdrive.py 1EitqzKZLptJop82hdNqu1YCogxgNgN5u data/nerf_synthetic/lego.zip
cd data/nerf_synthetic

echo "-> Unzipping lego.zip, overwrite if exist"
unzip -oq lego.zip
rm -rf __MACOSX lego.zip