# Generative AI Subject Assistant — Streamlit

Streamlit frontend for the fine-tuned LoRA-based Generative AI subject assistant for B.Tech course 1ADPC401.

**Created by Prof. Akshay Mane**  
Department of Artificial Intelligence & Data Science · ADCET, Ashta

## Files

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores local secrets, virtual environments, and model files

## Hugging Face setup

The app defaults to the public adapter repository `AIDSADCET/genai-subject-assistant-adapter`. To change it, set this in Streamlit Community Cloud → app Settings → Secrets:

```toml
HF_ADAPTER_REPO_ID = "AIDSADCET/genai-subject-assistant-adapter"
```

A public adapter does not normally need a Hugging Face access token. For a private repository only, add a read token to Secrets:

```toml
HF_TOKEN = "hf_your_read_token_here"
```

Never commit access tokens or other secrets to GitHub.

## Deploy

1. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore` to the root of your GitHub repository.
2. Open https://share.streamlit.io and create an app from that repository.
3. Set the branch to `main` and the main file path to `app.py`.
4. Add the model repository ID above in app Secrets if you need to override the default.
5. Deploy and test the app.

## Hosting limitation

The app downloads the base model named in the LoRA `adapter_config.json` and then loads the adapter. Free CPU hosts have limited memory and can be slow or fail to load a 0.5B model. The app uses CPU FP16 by default to reduce RAM. If CPU FP16 operations fail, add `GENAI_CPU_DTYPE = "float32"` to Streamlit Secrets/environment, but this uses more memory and may exceed the host limit. No free-hosting setup can guarantee that inference will fit its current resource allocation.

Fine-tuning does not guarantee factual correctness. Students should verify important answers against official course materials.
