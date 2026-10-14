from __future__ import annotations

"""Generative AI Subject Assistant (1ADPC401), implemented with Streamlit."""

import inspect
import os
import threading
import time
from typing import Any, Iterator

import streamlit as st
import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

# Must be the first Streamlit command in this script.
st.set_page_config(
    page_title="Generative AI Assistant — B.Tech (1ADPC401)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Generative AI Assistant — B.Tech (1ADPC401)"
DEFAULT_ADAPTER_REPO = "AIDSADCET/genai-subject-assistant-adapter"
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("GENAI_MAX_NEW_TOKENS", "192"))
MAX_INPUT_TOKENS = int(os.getenv("GENAI_MAX_INPUT_TOKENS", "1536"))

SYSTEM_PROMPT = """
You are the Generative AI Subject Assistant for B.Tech Semester VII,
course code 1ADPC401, at the Department of Artificial Intelligence and
Data Science, Annasaheb Dange College of Engineering and Technology, Ashta.

Help students understand concepts, prepare for examinations, implement
laboratory experiments, understand Python code, debug errors, and explore
applications. Explain at final-year engineering level, starting simply and
adding technical detail where needed.

COURSE COVERAGE
Unit 1: Generative and probabilistic models, GANs, VAEs, diffusion models,
Transformers, sampling, PyTorch, TensorFlow, and Hugging Face.
Unit 2: Advanced GANs, image/audio/text synthesis, deepfakes,
super-resolution, voice synthesis, text-to-image, training challenges.
Unit 3: Self-attention, multi-head attention, Transformer architecture,
GPT-2 to GPT-4, fine-tuning, chatbots, summarization, code generation.
Unit 4: Cross-domain generation, text-to-image, voice synthesis,
multimodal fusion, human-computer interaction, personalized responses,
and conversational AI.
Unit 5: RAG, external knowledge bases, VCA, cloud deployment, image
masking, inpainting, image editing, and background removal.
Unit 6: Fairness, explainability, accountability, transparency, safety,
deepfakes, AI-generated art, healthcare, autonomous vehicles, policy,
and regulation.

LAB EXPERIMENTS
1. Simple Autoencoders
2. Variational Autoencoders
3. Basic GAN
4. Conditional GAN for image-to-image translation
5. GPT-2/GPT-3 transformer-based text generation
6. RAG chatbot using Hugging Face Transformers
7. Image masking/editing using AI tools such as Stable Diffusion
8. Cloud deployment of a text-to-image model
9. AI art and deepfakes with GANs
10. AI bias and deepfake ethics case study

RESPONSE RULES
- Answer the student's question directly.
- For conceptual questions, explain clearly with an example.
- Use tables, equations, or headings when they genuinely help.
- Do not provide Python code unless requested or needed to solve a practical question.
- For debugging, request the relevant traceback/code if it is not supplied.
- Distinguish general technical explanation from official course rules.
- Never invent course deadlines, marks, policies, or experiment requirements.
- Acknowledge uncertainty and limits. Fine-tuning alone does not provide
  guaranteed access to all syllabus facts or lecture notes.
- For deepfake topics, emphasize consent, privacy, disclosure, and harm prevention.
""".strip()

ANSWER_STYLES: dict[str, str] = {
    "Balanced": "",
    "Simple": (
        "Explain for a beginner using plain language, one everyday analogy, "
        "and short paragraphs. Avoid heavy mathematics."
    ),
    "Exam-ready": (
        "Write a university exam answer. Start with a one-line definition, "
        "then use short headings and key points. Include the main equation or "
        "a diagram description where relevant, and end with a brief conclusion."
    ),
    "Deep technical": (
        "Include mathematical formulation, intuition, training or inference "
        "details, common failure modes, and trade-offs where relevant."
    ),
    "Lab & code help": (
        "Be practical. Give the experiment aim, required libraries, step-by-step "
        "procedure, expected output, and common errors. Include short Python code "
        "where helpful."
    ),
}

UNITS: list[tuple[str, list[str]]] = [
    ("Unit 1 · Foundations", [
        "Explain the difference between GANs and VAEs.",
        "What is the reparameterization trick in a VAE?",
        "How does a diffusion model generate an image step by step?",
    ]),
    ("Unit 2 · Advanced GANs", [
        "My GAN produces nearly identical images. How do I diagnose mode collapse?",
        "Compare DCGAN, WGAN and StyleGAN.",
        "How does super-resolution with GANs work?",
    ]),
    ("Unit 3 · Transformers", [
        "Explain self-attention with a simple example.",
        "Why do Transformers use multi-head attention?",
        "How did GPT-2 evolve into GPT-4? Distinguish known facts from uncertain details.",
    ]),
    ("Unit 4 · Multimodal", [
        "How does text-to-image generation work?",
        "Explain multimodal fusion with an example.",
        "How can a conversational AI give personalized responses responsibly?",
    ]),
    ("Unit 5 · RAG & Editing", [
        "How does a RAG chatbot retrieve and use documents?",
        "Explain inpainting with Stable Diffusion.",
        "What are the steps to deploy a model on the cloud?",
    ]),
    ("Unit 6 · Ethics", [
        "What are the main ethical risks of deepfakes?",
        "How can we make generative AI fair and explainable?",
        "Discuss responsible governance of AI-generated content.",
    ]),
]

LABS = [
    "1. Simple Autoencoders",
    "2. Variational Autoencoders",
    "3. Basic GAN",
    "4. Conditional GAN (image-to-image)",
    "5. GPT-2/GPT-3 text generation",
    "6. RAG chatbot with Hugging Face",
    "7. Image masking/editing with Stable Diffusion",
    "8. Cloud deployment of text-to-image model",
    "9. AI art and deepfakes with GANs",
    "10. AI bias and deepfake ethics case study",
]


def get_setting(key: str, default: str = "") -> str:
    """Read a setting from environment variables or Streamlit Secrets."""
    value = os.getenv(key)
    if value:
        return value
    try:
        value = st.secrets.get(key, default)
        return str(value) if value is not None else default
    except Exception:
        return default


ADAPTER_REPO_ID = get_setting("HF_ADAPTER_REPO_ID", DEFAULT_ADAPTER_REPO).strip()
HF_TOKEN = get_setting("HF_TOKEN", "").strip() or None
CPU_DTYPE_DEFAULT = get_setting("GENAI_CPU_DTYPE", "float16").strip().lower()


@st.cache_resource(show_spinner=False)
def load_model(adapter_repo_id: str, token: str | None, cpu_dtype_name: str):
    """Load base model and LoRA adapter once per app process."""
    if not adapter_repo_id or "/" not in adapter_repo_id:
        raise ValueError(
            "HF_ADAPTER_REPO_ID must be a Hugging Face model repository ID, "
            "for example 'AIDSADCET/genai-subject-assistant-adapter'."
        )

    peft_config = PeftConfig.from_pretrained(adapter_repo_id, token=token)
    base_model_name = peft_config.base_model_name_or_path
    if not base_model_name:
        raise ValueError("The adapter_config.json does not specify the base model name.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_repo_id, token=token)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_dtype = torch.float16
    elif cpu_dtype_name.lower() == "float32":
        model_dtype = torch.float32
    else:
        # Smaller footprint for the 0.5B base model on a constrained free host.
        # If this CPU does not support FP16 operations, set GENAI_CPU_DTYPE=float32.
        model_dtype = torch.float16

    load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    dtype_arg = "dtype" if "dtype" in params else "torch_dtype"
    load_kwargs[dtype_arg] = model_dtype

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=token,
        **load_kwargs,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_repo_id,
        token=token,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
    if use_cuda:
        model = model.to("cuda")
    return tokenizer, model, use_cuda, base_model_name


def build_inputs(
    question: str,
    history: list[dict[str, str]],
    style: str,
    tokenizer,
    max_input_tokens: int,
):
    """Build a chat-template prompt and trim older context if needed."""
    style_instruction = ANSWER_STYLES.get(style, "")
    system_text = SYSTEM_PROMPT
    if style_instruction:
        system_text += "\n\nANSWER STYLE\n" + style_instruction

    context = [
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in ("user", "assistant") and item.get("content", "").strip()
    ][-6:]
    current_question = question.strip()

    def encode() -> dict[str, torch.Tensor]:
        messages = [{"role": "system", "content": system_text}] + context + [
            {"role": "user", "content": current_question}
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)

    inputs = encode()
    while inputs["input_ids"].shape[1] > max_input_tokens and context:
        # Remove the oldest message(s), retaining the latest exchange where possible.
        context = context[2:] if len(context) >= 2 else context[1:]
        inputs = encode()

    while inputs["input_ids"].shape[1] > max_input_tokens and len(current_question) > 80:
        current_question = current_question[: int(len(current_question) * 0.8)]
        inputs = encode()

    if inputs["input_ids"].shape[1] > max_input_tokens:
        raise ValueError(
            "The system prompt and question exceed the configured context length. "
            "Please ask a shorter question or increase GENAI_MAX_INPUT_TOKENS."
        )
    return inputs


class StopOnEvent(StoppingCriteria):
    """Stop generation when the per-request stop event is set."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],), self.stop_event.is_set(),
            dtype=torch.bool, device=input_ids.device
        )


def stream_answer(
    question: str,
    history: list[dict[str, str]],
    style: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    tokenizer,
    model,
    use_cuda: bool,
) -> Iterator[str]:
    """Generate streamed text from the model."""
    inputs = build_inputs(question, history, style, tokenizer, MAX_INPUT_TOKENS)
    if use_cuda:
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
    )
    stopped = threading.Event()
    failures: list[BaseException] = []
    generation_args = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": float(temperature) > 0,
        "temperature": max(float(temperature), 0.05),
        "top_p": float(top_p),
        "repetition_penalty": 1.08,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "stopping_criteria": StoppingCriteriaList([StopOnEvent(stopped)]),
    }

    def worker() -> None:
        try:
            with torch.inference_mode():
                model.generate(**generation_args)
        except BaseException as exc:
            failures.append(exc)
            # Ensure a consumer waiting on the stream is released on generation errors.
            try:
                streamer.on_finalized_text("", stream_end=True)
            except Exception:
                pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    for piece in streamer:
        if piece:
            yield piece
    thread.join(timeout=5)
    if failures:
        err = failures[0]
        yield (
            "\n\n⚠️ Generation encountered an error ("
            f"{type(err).__name__}: {err}). If this is a CPU FP16 error, "
            "set `GENAI_CPU_DTYPE = \"float32\"` in the hosting Secrets/settings "
            "or environment, although this uses more memory."
        )


def queue_question(question: str) -> None:
    st.session_state.pending_question = question
    st.session_state.pending_history = list(st.session_state.get("messages", []))


def retry_last_answer() -> None:
    messages = list(st.session_state.get("messages", []))
    last_user = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"),
        None,
    )
    if last_user is None:
        return
    question = messages[last_user]["content"]
    history = messages[:last_user]
    st.session_state.messages = history
    st.session_state.pending_question = question
    st.session_state.pending_history = history


def undo_last_exchange() -> None:
    messages = list(st.session_state.get("messages", []))
    last_user = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"),
        None,
    )
    if last_user is not None:
        st.session_state.messages = messages[:last_user]


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pop("pending_question", None)
    st.session_state.pop("pending_history", None)


# --------------------------- Page setup and styling ---------------------------
st.markdown(
    """
    <style>
    .block-container {max-width: 1450px; padding-top: 1.4rem; padding-bottom: 1.4rem;}
    .genai-hero {
      border-radius: 20px; padding: 1.45rem 1.65rem; margin-bottom: 1rem;
      color: white; background: linear-gradient(120deg, #0e2a47, #3446a5, #0f8d99);
      box-shadow: 0 10px 28px rgba(14,42,71,.20);
    }
    .genai-hero h1 {color:white; margin:0 0 .55rem 0; font-size:2rem; line-height:1.2;}
    .genai-hero p {color:white; margin:0; opacity:.93; line-height:1.6;}
    .meta-pills {display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem;}
    .meta-pill {border:1px solid rgba(255,255,255,.36); border-radius:999px;
      padding:.25rem .65rem; background:rgba(255,255,255,.12); font-size:.82rem;}
    .small-muted {font-size:.85rem; opacity:.78;}
    .app-footer {border-top:1px solid rgba(128,128,128,.3); margin-top:2rem;
      padding-top:1rem; text-align:center; font-size:.9rem; opacity:.82; line-height:1.7;}
    </style>
    <div class="genai-hero">
      <h1>Generative AI Subject Assistant</h1>
      <p>B.Tech Semester VII · 1ADPC401 · Ask about concepts, laboratory experiments,
      Python implementations, debugging, and examination preparation across all six units.</p>
      <div class="meta-pills">
        <span class="meta-pill">6 syllabus units</span>
        <span class="meta-pill">10 lab experiments</span>
        <span class="meta-pill">Fine-tuned base model + LoRA</span>
        <span class="meta-pill">Streamlit Community Cloud ready</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "pending_history" not in st.session_state:
    st.session_state.pending_history = None

# ------------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.subheader("🎓 Course assistant")
    st.markdown("**Course:** Generative AI (1ADPC401)")
    st.markdown("**Programme:** B.Tech — AI & Data Science")
    st.markdown("**Institute:** ADCET, Ashta")
    st.caption(f"Adapter repository: `{ADAPTER_REPO_ID}`")
    st.divider()

    st.markdown("#### Answer style")
    answer_style = st.selectbox(
        "Choose how the answer should be written",
        list(ANSWER_STYLES.keys()),
        index=0,
        label_visibility="collapsed",
    )
    with st.expander("Generation settings", expanded=False):
        temperature = st.slider("Creativity (temperature)", 0.1, 1.2, 0.7, 0.05)
        top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05)
        max_new_tokens = st.slider(
            "Maximum answer length (tokens)", 64, 512,
            min(max(MAX_NEW_TOKENS_DEFAULT, 64), 512), 32,
        )
        cpu_dtype = st.selectbox(
            "CPU precision (cloud hosting)",
            options=["float16", "float32"],
            index=1 if CPU_DTYPE_DEFAULT == "float32" else 0,
            help="float16 uses less RAM but some CPUs may not support every operation. float32 is more compatible but uses more RAM.",
        )

    st.divider()
    st.markdown("#### Browse by unit")
    for unit_index, (unit_title, questions) in enumerate(UNITS):
        with st.expander(unit_title, expanded=(unit_index == 0)):
            for question_index, question in enumerate(questions):
                st.button(
                    question,
                    key=f"unit_{unit_index}_{question_index}",
                    use_container_width=True,
                    on_click=queue_question,
                    args=(question,),
                )

    st.markdown("#### Laboratory support")
    selected_lab = st.selectbox("Select an experiment", LABS, index=0)
    selected_lab_name = selected_lab.split(". ", 1)[-1]
    st.button(
        "Generate complete lab guide",
        key="lab_guide",
        type="primary",
        use_container_width=True,
        on_click=queue_question,
        args=(f"Give me a complete guide for the lab experiment '{selected_lab_name}': aim, theory, required libraries, step-by-step procedure, expected output and conclusion.",),
    )
    st.button(
        "Prepare viva questions",
        key="lab_viva",
        use_container_width=True,
        on_click=queue_question,
        args=(f"Give me 8 likely viva questions with short model answers for the lab experiment '{selected_lab_name}'.",),
    )
    st.button(
        "Troubleshoot common errors",
        key="lab_errors",
        use_container_width=True,
        on_click=queue_question,
        args=(f"What are common errors students face in the lab experiment '{selected_lab_name}', and how can they fix them? Include practical examples.",),
    )

    st.markdown("#### Quiz mode")
    quiz_unit = st.selectbox("Choose a unit", [unit for unit, _ in UNITS], key="quiz_unit")
    st.button(
        "Ask me a quiz question",
        key="quiz_start",
        use_container_width=True,
        on_click=queue_question,
        args=((f"Quiz me on {quiz_unit}. Ask exactly one question suitable for a university exam. "
               "Do not give the answer yet; wait for my response."),),
    )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.button("↶ Undo", use_container_width=True, on_click=undo_last_exchange)
    with col_b:
        st.button("↻ Retry", use_container_width=True, on_click=retry_last_answer)
    st.button("Clear conversation", use_container_width=True, on_click=clear_conversation)
    st.caption(
        "AI-generated answers may contain errors. Verify important details against official course materials. "
        "Do not enter private or confidential information."
    )

# ------------------------------- Main chat -----------------------------------
st.markdown("### Conversation")
if not st.session_state.messages:
    st.info(
        "Welcome! Ask a question below, or choose a suggested question, lab guide, "
        "or quiz from the sidebar."
    )

chat_prompt = st.chat_input(
    "Ask about GANs, VAEs, Transformers, RAG, a lab experiment, or a Python error..."
)

# Show the saved conversation before appending a new answer.
for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

pending_prompt = st.session_state.get("pending_question")
if pending_prompt:
    question = str(pending_prompt)
    history_for_question = list(st.session_state.get("pending_history") or [])
    st.session_state.pending_question = None
    st.session_state.pending_history = None
elif chat_prompt:
    question = chat_prompt.strip()
    history_for_question = list(st.session_state.messages)
else:
    question = ""
    history_for_question = []

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Loading the model on first use. This may take several minutes..."):
                tokenizer, model, use_cuda, base_model_name = load_model(
                    ADAPTER_REPO_ID, HF_TOKEN, cpu_dtype
                )
            stream = stream_answer(
                question=question,
                history=history_for_question,
                style=answer_style,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
                model=model,
                use_cuda=use_cuda,
            )
            answer = st.write_stream(stream)
            if not isinstance(answer, str):
                answer = str(answer or "")
            if not answer.strip():
                answer = "I couldn't generate an answer. Please rephrase your question and try again."
                st.markdown(answer)
        except Exception as exc:
            answer = (
                "The assistant could not load the model or generate a response.\n\n"
                "Confirm that the Hugging Face model repository contains `adapter_config.json` "
                "and `adapter_model.safetensors`, and that `HF_ADAPTER_REPO_ID` is correct. "
                "Free cloud hosting may also run out of RAM while loading or using the base model.\n\n"
                f"**Technical detail:** `{type(exc).__name__}: {exc}`"
            )
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    """
    <div class="app-footer">
      <strong>Created by Prof. Akshay Mane</strong><br>
      Department of Artificial Intelligence &amp; Data Science · ADCET, Ashta<br>
      For learning support. Verify important answers against official course materials.
    </div>
    """,
    unsafe_allow_html=True,
)
