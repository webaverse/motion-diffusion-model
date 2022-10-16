echo "Updating APT"
sudo apt update
echo "Installing FFMPEG"
sudo apt install ffmpeg zip unzip

echo "Installing Anaconda."
echo "***** IMPORTANT *******"
echo "*** Please follow the prompts to install."
echo "*** Once you have installed, run setup.sh"
echo ""
echo "********************************************"
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash ./Anaconda3-2022.05-Linux-x86_64.sh