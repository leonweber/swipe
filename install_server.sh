cd ~

apt update
apt install -y Python3.11 tmux
pip install pipx
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
pipx install poetry

git clone https://github.com/leonweber/swipe.git
cd swipe
poetry install

poetry run wandb login
