"""Streamlit demo: classify a pasted contract clause with the fine-tuned
Phi-3 + LoRA adapter, showing the zero-shot base model's prediction
alongside it via peft's disable_adapter() (same trick used in the eval
notebook) rather than loading two separate model copies.

Deliberately self-contained (no contract_clause_qlora import) and runs on
CPU (device_map="cpu" below) for public deployment, where there's no GPU.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_ID = "DLCS/contract-clause-phi3-lora"
MAX_CLAUSE_LENGTH = 2000

# --- Copied from src/contract_clause_qlora/cuad.py -- keep in sync if that file changes. ---

INSTRUCTION_TEMPLATE = """Classify the following contract clause into its category, or respond with "None" if it does not match any category. Respond with only the category name and nothing else.

Clause:
{clause_text}"""

VALID_CATEGORIES = {
    "Affiliate License-Licensee",
    "Affiliate License-Licensor",
    "Agreement Date",
    "Anti-Assignment",
    "Audit Rights",
    "Cap On Liability",
    "Change Of Control",
    "Competitive Restriction Exception",
    "Covenant Not To Sue",
    "Document Name",
    "Effective Date",
    "Exclusivity",
    "Expiration Date",
    "Governing Law",
    "Insurance",
    "Ip Ownership Assignment",
    "Irrevocable Or Perpetual License",
    "Joint Ip Ownership",
    "License Grant",
    "Liquidated Damages",
    "Minimum Commitment",
    "Most Favored Nation",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Compete",
    "Non-Disparagement",
    "Non-Transferable License",
    "None",
    "Notice Period To Terminate Renewal",
    "Parties",
    "Post-Termination Services",
    "Price Restrictions",
    "Renewal Term",
    "Revenue/Profit Sharing",
    "Rofr/Rofo/Rofn",
    "Source Code Escrow",
    "Termination For Convenience",
    "Third Party Beneficiary",
    "Uncapped Liability",
    "Unlimited/All-You-Can-Eat-License",
    "Volume Restriction",
    "Warranty Duration",
}


def build_inference_prompt(clause_text, tokenizer):
    """Build an inference-ready prompt, ending at <|assistant|> with no completion.

    See cuad.py for full docs -- this is a straight copy.
    """
    messages = [
        {"role": "user", "content": INSTRUCTION_TEMPLATE.format(clause_text=clause_text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_prediction(raw_text, valid_categories):
    """Parse raw generated text into a valid category, or "INVALID".

    Cascading exact -> case-insensitive -> punctuation-stripped match; see
    cuad.py for the full rationale (also a straight copy).
    """
    stripped = raw_text.strip()

    if stripped in valid_categories:
        return stripped

    lowercase_lookup = {cat.lower(): cat for cat in valid_categories}
    if stripped.lower() in lowercase_lookup:
        return lowercase_lookup[stripped.lower()]

    trailing_stripped = stripped.rstrip('.\'"')
    if trailing_stripped in valid_categories:
        return trailing_stripped
    if trailing_stripped.lower() in lowercase_lookup:
        return lowercase_lookup[trailing_stripped.lower()]

    return "INVALID"

# --- End copied section ---

EXAMPLE_CLAUSES = {
    "Governing Law": "This Agreement shall be governed by and construed under the laws of the State of Delaware, without regard to its conflict of laws principles.",
    "Document Name": "DISTRIBUTOR AGREEMENT",
    "Parties": "This Agreement is entered into by and between Acme Corporation, a Delaware corporation (\"Company\"), and Widget Industries LLC (\"Distributor\").",
    "None / Boilerplate": "The waiver by either party of any breach of this Agreement by the other party in a particular instance shall not operate as a waiver of subsequent breaches of the same or different kind.",
}

@st.cache_resource
def load_model():
    """Load the base model + LoRA adapter once and cache across reruns.

    Streamlit reruns this whole script on every widget interaction --
    without @st.cache_resource, the ~2GB 4-bit model would reload from
    scratch on every button click. device_map="cpu" is deliberate: public
    deployment (Streamlit Community Cloud / HF Spaces free tier) has no
    GPU, and this has been verified to work correctly on CPU-only.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()

    return model, tokenizer


def classify(clause_text, model, tokenizer, use_adapter=True):
    """Generate and parse a category prediction for one clause.

    use_adapter=False generates inside peft's disable_adapter() context,
    giving the zero-shot base model's prediction from the same loaded
    model -- no second model copy needed, same trick as the eval notebook.
    """
    prompt = build_inference_prompt(clause_text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        # 18 = longest category name's token count + small buffer (same
        # value used for eval generation, derived empirically there).
        if use_adapter:
            output = model.generate(**inputs, max_new_tokens=18)
        else:
            with model.disable_adapter():
                output = model.generate(**inputs, max_new_tokens=18)

    raw_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    parsed = parse_prediction(raw_text, VALID_CATEGORIES)
    return parsed, raw_text


st.title("Contract Clause Classifier")
st.write(
    "Paste a contract clause below to see how a fine-tuned Phi-3 model classifies it, "
    "compared to the same model without fine-tuning."
)

with st.expander("About this project"):
    st.write(
        "A QLoRA fine-tune of Phi-3-mini-4k-instruct on the CUAD (Contract Understanding "
        "Atticus Dataset), classifying contract clauses into 41 legal categories (or 'None'). "
        "[View the code on GitHub](https://github.com/D-L-C-S/contract-clause-qlora)."
    )

model, tokenizer = load_model()

st.write("**Try an example:**")
# Setting session_state["clause_text"] here populates the text_area below,
# since it shares that same widget key -- Streamlit's way of letting one
# widget's interaction update another's value.
example_cols = st.columns(len(EXAMPLE_CLAUSES))
for col, (label, text) in zip(example_cols, EXAMPLE_CLAUSES.items()):
    if col.button(label):
        st.session_state["clause_text"] = text

clause_text = st.text_area(
    "Contract clause text",
    height=150,
    key="clause_text",
)

if st.button("Classify"):
    stripped = clause_text.strip()
    if not stripped:
        st.warning("Please paste some clause text before classifying.")
    elif len(stripped) > MAX_CLAUSE_LENGTH:
        st.warning(
            f"That's a bit long ({len(stripped)} characters). "
            f"Please paste a single clause, not a whole contract (max {MAX_CLAUSE_LENGTH} characters)."
        )
    else:
        with st.spinner("Classifying... this may take up to 30 seconds on CPU."):
            finetuned_pred, finetuned_raw = classify(stripped, model, tokenizer, use_adapter=True)
            baseline_pred, baseline_raw = classify(stripped, model, tokenizer, use_adapter=False)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fine-tuned")
            st.write(finetuned_pred)
        with col2:
            st.subheader("Baseline (zero-shot)")
            st.write(baseline_pred)

        with st.expander("Show raw model output"):
            st.write("**Fine-tuned raw output:**", repr(finetuned_raw))
            st.write("**Baseline raw output:**", repr(baseline_raw))