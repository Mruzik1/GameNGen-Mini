# GameNGen-Mini
A DiT model based on Oasis implementation that generates the next frame conditioned on actions and the previous frames. The files are gonna be added soon.

## Script Descriptions
- **generate_dataset.py** *[implemented from scratch]*: This script is used to generate data for training (i.e. frames, actions, temporal indices). It utilizes gymnasium to simulate a game environment and stable_baselines3 to train an agent. I tried PPO and DQN, but decided to go with the second one since the results were slightly better. The game simulated was “ALE/Pong-v5“.
- **dataset.py** *[implemented from scratch]*: Contains a dataset class for loading the generated data.
- **generate.py** *[made by [1], adapted]*: Generates a video of an artificial gameplay using a single frame + encodings of actions in a “.pt“ format.
- **train.py** *[implemented from scratch]*: A training script; uses wandb for logging.
- **uvit_vae.py** *[made by [2], unchanged]*: Contains an architecture of VAE with attention (patching is 8, encodes into a latent space with 4 channels). I use it with the pre-trained weights: “autoencoder_kl.pth“.
- **attention.py** *[made by [3], unchanged]*: Contains two types of attentions: temporal axial attention and spatial axial attention. These attentions are then combined in **dit.py** into SpatioTemporalDiTBlock.
- **rotary_embedding_torch.py** *[made by [4], unchanged]*: Contains an implementation of RotaryEmbedding used in **dit.py**.
- **dit.py** *[made by [1], adapted]*: Contains an implementation of a diffusion model with a transformer backbone. It is a modification of a traditional DiT that is also temporal-aware.
- **utils.py** *[made by [1], adapted]*: Contains various utils for the project.

## References
- [1] Decart et al. Oasis: A Universe in a Transformer. 2024. url: https://oasis-model.github.io/.
- [2] Fan Bao et al. “All are worth words: A vit backbone for diffusion models”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023, pp. 22669–22679.
- [3] Boyuan Chen et al. Diffusion Forcing: Next-token Prediction Meets FullSequence Diffusion. 2024. arXiv: 2407 . 01392 [cs.LG]. url: https://arxiv.org/abs/2407.01392.
- [4] Jianlin Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. 2021. arXiv: 2104.09864 [cs.CL].
